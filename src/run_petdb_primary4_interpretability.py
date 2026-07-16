from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache" / "matplotlib"
CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(CACHE))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import yaml
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline


DATA_PATH = ROOT / "data" / "processed" / "petdb_primary4_v0_1.parquet"
CONFIG_PATH = ROOT / "config" / "study_config.yaml"
OUT_DIR = ROOT / "reports" / "modeling"
FIG_DIR = ROOT / "figures"
SEED = 20260714
LABELS = ["ARC", "CIB", "MORB", "OIB"]


def make_model(n_estimators: int = 160) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    min_samples_leaf=4,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=SEED,
                ),
            ),
        ]
    )


def stratified_subset(index: np.ndarray, y: pd.Series, size: int, seed: int) -> np.ndarray:
    if len(index) <= size:
        return index
    chosen, _ = train_test_split(
        index,
        train_size=size,
        stratify=y.iloc[index],
        random_state=seed,
    )
    return np.sort(chosen)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    configured = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "feature_sets"
    ]["broad_v1"]
    data = pd.read_parquet(DATA_PATH)
    data = data.loc[data["model_ready_broad"]].reset_index(drop=True)
    y = data["label_primary4"].astype(str)
    class_non_null = data.groupby("label_primary4")[configured].count()
    features = [feature for feature in configured if class_non_null[feature].min() > 0]
    excluded = [feature for feature in configured if feature not in features]
    if excluded != ["TH(PPM)"]:
        raise ValueError(f"Unexpected structural exclusions: {excluded}")
    X = np.log10(data[features].where(data[features] > 0))
    groups = data["citation_overlap_component"]

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    permutation_rows: list[dict[str, object]] = []
    for fold, (train, test) in enumerate(splitter.split(X, y, groups), start=1):
        model = make_model()
        model.fit(X.iloc[train], y.iloc[train])
        result = permutation_importance(
            model,
            X.iloc[test],
            y.iloc[test],
            scoring="balanced_accuracy",
            n_repeats=6,
            random_state=SEED + fold,
            n_jobs=-1,
        )
        for feature, mean, sd in zip(
            features, result.importances_mean, result.importances_std, strict=True
        ):
            permutation_rows.append(
                {
                    "fold": fold,
                    "feature": feature,
                    "evaluation_n": len(test),
                    "importance_mean": mean,
                    "importance_sd_within_fold": sd,
                }
            )
    permutation_folds = pd.DataFrame(permutation_rows)
    permutation_summary = (
        permutation_folds.groupby("feature")
        .agg(
            importance_mean=("importance_mean", "mean"),
            importance_sd_between_folds=("importance_mean", "std"),
            min_fold=("importance_mean", "min"),
            max_fold=("importance_mean", "max"),
        )
        .reset_index()
        .sort_values("importance_mean", ascending=False)
    )
    permutation_folds.to_csv(
        OUT_DIR / "petdb_primary4_permutation_importance_folds.csv", index=False
    )
    permutation_summary.to_csv(
        OUT_DIR / "petdb_primary4_permutation_importance_summary.csv", index=False
    )

    full_model = make_model(n_estimators=160)
    full_model.fit(X, y)
    imputer = full_model.named_steps["imputer"]
    forest = full_model.named_steps["model"]
    imputed = imputer.transform(X)
    shap_index = stratified_subset(np.arange(len(data)), y, 800, SEED)
    shap_input = imputed[shap_index]
    explainer = shap.TreeExplainer(forest)
    raw_shap = explainer.shap_values(shap_input, check_additivity=False)
    if isinstance(raw_shap, list):
        shap_values = np.stack(raw_shap, axis=-1)
    else:
        shap_values = np.asarray(raw_shap)
    if shap_values.ndim == 2:
        shap_values = shap_values[:, :, np.newaxis]
    if shap_values.shape != (len(shap_index), len(features), len(LABELS)):
        raise RuntimeError(f"Unexpected SHAP shape: {shap_values.shape}")

    shap_rows: list[dict[str, object]] = []
    direction_rows: list[dict[str, object]] = []
    mean_abs = np.abs(shap_values).mean(axis=0)
    for class_index, label in enumerate(forest.classes_):
        for feature_index, feature in enumerate(features):
            shap_rows.append(
                {
                    "feature": feature,
                    "label": label,
                    "mean_abs_shap": mean_abs[feature_index, class_index],
                }
            )
            direction_rows.append(
                {
                    "feature": feature,
                    "label": label,
                    "spearman_log_value_vs_shap": spearmanr(
                        shap_input[:, feature_index],
                        shap_values[:, feature_index, class_index],
                    ).statistic,
                }
            )
    shap_by_class = pd.DataFrame(shap_rows)
    shap_direction = pd.DataFrame(direction_rows)
    shap_overall = (
        shap_by_class.groupby("feature")["mean_abs_shap"]
        .mean()
        .rename("mean_abs_shap_over_classes")
        .sort_values(ascending=False)
        .reset_index()
    )
    shap_by_class.to_csv(
        OUT_DIR / "petdb_primary4_shap_global_by_class.csv", index=False
    )
    shap_overall.to_csv(
        OUT_DIR / "petdb_primary4_shap_global_overall.csv", index=False
    )
    shap_direction.to_csv(
        OUT_DIR / "petdb_primary4_shap_direction_spearman.csv", index=False
    )

    sns.set_theme(style="whitegrid", context="notebook")
    top = permutation_summary.head(15).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(9.0, 7.4))
    ax.barh(
        top["feature"],
        top["importance_mean"],
        xerr=top["importance_sd_between_folds"],
        color="#35618F",
        edgecolor="#233F5E",
        linewidth=0.7,
        capsize=2.5,
    )
    ax.axvline(0, color="#3F444A", linewidth=0.9)
    ax.set_xlabel("Decrease in balanced accuracy after permutation")
    ax.set_ylabel("")
    ax.set_title("PetDB four-class permutation importance")
    fig.text(
        0.5,
        0.935,
        "Mean ± between-fold SD; five citation-overlap-held-out folds; n = 6,148; 21 shared features",
        ha="center",
        fontsize=10,
        color="#4A4F55",
    )
    fig.subplots_adjust(top=0.88, left=0.25, bottom=0.11)
    fig.savefig(FIG_DIR / "petdb_primary4_permutation_importance.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "petdb_primary4_permutation_importance.pdf", bbox_inches="tight")
    plt.close(fig)

    top_features = shap_overall.head(15)["feature"].tolist()
    shap_matrix = (
        shap_by_class[shap_by_class["feature"].isin(top_features)]
        .pivot(index="feature", columns="label", values="mean_abs_shap")
        .reindex(top_features[::-1])
        .reindex(columns=LABELS)
    )
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    sns.heatmap(
        shap_matrix,
        cmap="Blues",
        annot=True,
        fmt=".3f",
        linewidths=0.4,
        ax=ax,
    )
    ax.set_xlabel("Class")
    ax.set_ylabel("")
    ax.set_title("PetDB four-class mean absolute SHAP value")
    fig.text(
        0.5,
        0.935,
        "Descriptive full-cohort model; stratified SHAP sample n = 800; log10 inputs",
        ha="center",
        fontsize=10,
        color="#4A4F55",
    )
    fig.subplots_adjust(top=0.88, left=0.24, bottom=0.10)
    fig.savefig(FIG_DIR / "petdb_primary4_shap_global_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "petdb_primary4_shap_global_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)

    direction_matrix = (
        shap_direction[shap_direction["feature"].isin(top_features)]
        .pivot(index="feature", columns="label", values="spearman_log_value_vs_shap")
        .reindex(top_features[::-1])
        .reindex(columns=LABELS)
    )
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    sns.heatmap(
        direction_matrix,
        cmap=sns.diverging_palette(25, 240, as_cmap=True),
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        ax=ax,
    )
    ax.set_xlabel("Class")
    ax.set_ylabel("")
    ax.set_title("PetDB four-class SHAP direction")
    fig.text(
        0.5,
        0.935,
        "Spearman association between log10 concentration and class SHAP contribution; associative, not causal",
        ha="center",
        fontsize=9.7,
        color="#4A4F55",
    )
    fig.subplots_adjust(top=0.88, left=0.24, bottom=0.10)
    fig.savefig(FIG_DIR / "petdb_primary4_shap_direction_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "petdb_primary4_shap_direction_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "dataset": str(DATA_PATH.relative_to(ROOT)),
        "cohort_rows": len(data),
        "labels": LABELS,
        "features": features,
        "excluded_structurally_absent_features": excluded,
        "permutation_validation": (
            "five-fold citation-overlap-component grouped; all held-out rows; "
            "six repeats; balanced-accuracy scoring"
        ),
        "shap_model": "full-cohort descriptive balanced random forest, 160 trees",
        "shap_sample_rows": len(shap_index),
        "caveat": (
            "Permutation importance is leakage-aware but correlated features share or mask "
            "importance. SHAP is descriptive full-cohort attribution and is not causal."
        ),
    }
    (OUT_DIR / "petdb_primary4_interpretability_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(permutation_summary.head(15).to_string(index=False))
    print("\nSHAP\n", shap_overall.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
