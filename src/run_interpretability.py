from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATPLOTLIB_CACHE = PROJECT_ROOT / ".cache" / "matplotlib"
MATPLOTLIB_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE))

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


ROOT = PROJECT_ROOT
DATA_PATH = ROOT / "data" / "processed" / "georoc_primary3_v0_1.parquet"
CONFIG_PATH = ROOT / "config" / "study_config.yaml"
OUT_DIR = ROOT / "reports" / "modeling"
FIG_DIR = ROOT / "figures"
RANDOM_SEED = 20260714
LABELS = ["ARC", "CIB", "OIB"]


def model_pipeline(n_estimators: int = 120) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=False)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    min_samples_leaf=4,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def stratified_subset(index: np.ndarray, y: pd.Series, size: int, seed: int) -> np.ndarray:
    if len(index) <= size:
        return index
    selected, _ = train_test_split(
        index,
        train_size=size,
        stratify=y.iloc[index],
        random_state=seed,
    )
    return np.sort(selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reuse-permutation",
        action="store_true",
        help="Reuse existing fold-level permutation outputs from the same pinned dataset and configuration.",
    )
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    features = config["feature_sets"]["broad_v1"]
    data = pd.read_parquet(DATA_PATH)
    data = data.loc[data["model_ready_broad"] & data["model_ready_immobile"]].reset_index(drop=True)
    X = np.log10(data[features].where(data[features] > 0))
    y = data["label_primary3"].astype(str)
    groups = data["citation_set"]

    permutation_fold_path = OUT_DIR / "permutation_importance_folds.csv"
    permutation_summary_path = OUT_DIR / "permutation_importance_summary.csv"
    if args.reuse_permutation:
        if not permutation_fold_path.exists() or not permutation_summary_path.exists():
            raise FileNotFoundError("Permutation outputs requested for reuse but not found")
        permutation_fold = pd.read_csv(permutation_fold_path)
        permutation_summary = pd.read_csv(permutation_summary_path)
    else:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        permutation_rows: list[dict[str, object]] = []
        for fold, (train_index, test_index) in enumerate(splitter.split(X, y, groups), start=1):
            model = model_pipeline(n_estimators=120)
            model.fit(X.iloc[train_index], y.iloc[train_index])
            evaluation_index = stratified_subset(test_index, y, size=4000, seed=RANDOM_SEED + fold)
            result = permutation_importance(
                model,
                X.iloc[evaluation_index],
                y.iloc[evaluation_index],
                scoring="balanced_accuracy",
                n_repeats=4,
                random_state=RANDOM_SEED + fold,
                n_jobs=-1,
            )
            for feature, mean, std in zip(features, result.importances_mean, result.importances_std):
                permutation_rows.append(
                    {
                        "fold": fold,
                        "feature": feature,
                        "evaluation_n": len(evaluation_index),
                        "importance_mean": mean,
                        "importance_std_within_fold": std,
                    }
                )

        permutation_fold = pd.DataFrame(permutation_rows)
        permutation_fold.to_csv(permutation_fold_path, index=False)
        permutation_summary = (
            permutation_fold.groupby("feature")
            .agg(
                importance_mean=("importance_mean", "mean"),
                importance_sd_between_folds=("importance_mean", "std"),
                min_fold=("importance_mean", "min"),
                max_fold=("importance_mean", "max"),
            )
            .reset_index()
            .sort_values("importance_mean", ascending=False)
        )
        permutation_summary.to_csv(permutation_summary_path, index=False)

    full_model = model_pipeline(n_estimators=100)
    full_model.fit(X, y)
    imputer = full_model.named_steps["imputer"]
    forest = full_model.named_steps["model"]
    all_imputed = imputer.transform(X)
    all_index = np.arange(len(data))
    shap_index = stratified_subset(all_index, y, size=500, seed=RANDOM_SEED)
    shap_input = all_imputed[shap_index]
    explainer = shap.TreeExplainer(forest)
    shap_values_raw = explainer.shap_values(shap_input, check_additivity=False)
    if isinstance(shap_values_raw, list):
        shap_values = np.stack(shap_values_raw, axis=-1)
    else:
        shap_values = np.asarray(shap_values_raw)
    if shap_values.ndim == 2:
        shap_values = shap_values[:, :, np.newaxis]
    if shap_values.shape[1] != len(features):
        raise RuntimeError(f"Unexpected SHAP shape: {shap_values.shape}")
    if shap_values.shape[2] != len(forest.classes_):
        raise RuntimeError(f"Unexpected SHAP class dimension: {shap_values.shape}")

    mean_abs = np.abs(shap_values).mean(axis=0)
    shap_rows: list[dict[str, object]] = []
    direction_rows: list[dict[str, object]] = []
    for class_index, label in enumerate(forest.classes_):
        for feature_index, feature in enumerate(features):
            shap_rows.append(
                {
                    "feature": feature,
                    "label": label,
                    "mean_abs_shap": mean_abs[feature_index, class_index],
                }
            )
            correlation = spearmanr(
                shap_input[:, feature_index],
                shap_values[:, feature_index, class_index],
                nan_policy="omit",
            ).statistic
            direction_rows.append(
                {
                    "feature": feature,
                    "label": label,
                    "spearman_log_value_vs_shap": correlation,
                }
            )
    shap_table = pd.DataFrame(shap_rows)
    direction_table = pd.DataFrame(direction_rows)
    shap_overall = (
        shap_table.groupby("feature")["mean_abs_shap"]
        .mean()
        .rename("mean_abs_shap_over_classes")
        .sort_values(ascending=False)
        .reset_index()
    )
    shap_table.to_csv(OUT_DIR / "shap_global_by_class.csv", index=False)
    shap_overall.to_csv(OUT_DIR / "shap_global_overall.csv", index=False)
    direction_table.to_csv(OUT_DIR / "shap_direction_spearman.csv", index=False)

    sns.set_theme(style="whitegrid", context="notebook")
    top_permutation = permutation_summary.head(15).sort_values("importance_mean")
    figure, axis = plt.subplots(figsize=(8.8, 7.3))
    axis.barh(
        top_permutation["feature"],
        top_permutation["importance_mean"],
        xerr=top_permutation["importance_sd_between_folds"],
        color="#35618F",
        edgecolor="#233F5E",
        linewidth=0.6,
        capsize=2.5,
    )
    axis.axvline(0, color="#4A4F55", linewidth=0.9)
    axis.set_xlabel("Decrease in balanced accuracy after permutation")
    axis.set_ylabel("")
    figure.suptitle("Feature importance under citation-grouped validation", y=0.98, fontsize=16)
    figure.text(
        0.5,
        0.925,
        "Mean ± between-fold SD; common cohort; up to 4,000 held-out records per fold",
        ha="center",
        fontsize=10.2,
        color="#4A4F55",
    )
    figure.subplots_adjust(top=0.85, left=0.25, bottom=0.11)
    figure.savefig(FIG_DIR / "permutation_importance.png", dpi=300, bbox_inches="tight")
    figure.savefig(FIG_DIR / "permutation_importance.pdf", bbox_inches="tight")
    plt.close(figure)

    top_features = shap_overall.head(15)["feature"].tolist()
    shap_matrix = (
        shap_table[shap_table["feature"].isin(top_features)]
        .pivot(index="feature", columns="label", values="mean_abs_shap")
        .reindex(top_features[::-1])
        .reindex(columns=LABELS)
    )
    figure, axis = plt.subplots(figsize=(7.2, 7.0))
    sns.heatmap(shap_matrix, cmap="Blues", annot=True, fmt=".3f", linewidths=0.4, ax=axis)
    axis.set_title("Mean absolute SHAP value by class")
    axis.set_xlabel("Class")
    axis.set_ylabel("")
    figure.suptitle("Global random-forest attribution", y=0.985, fontsize=16)
    figure.text(
        0.5,
        0.944,
        "Descriptive full-cohort model; stratified SHAP sample n = 500; log10 inputs",
        ha="center",
        fontsize=10.2,
        color="#4A4F55",
    )
    figure.subplots_adjust(top=0.86, left=0.24, bottom=0.10)
    figure.savefig(FIG_DIR / "shap_global_heatmap.png", dpi=300, bbox_inches="tight")
    figure.savefig(FIG_DIR / "shap_global_heatmap.pdf", bbox_inches="tight")
    plt.close(figure)

    direction_matrix = (
        direction_table[direction_table["feature"].isin(top_features)]
        .pivot(index="feature", columns="label", values="spearman_log_value_vs_shap")
        .reindex(top_features[::-1])
        .reindex(columns=LABELS)
    )
    figure, axis = plt.subplots(figsize=(7.2, 7.0))
    sns.heatmap(
        direction_matrix,
        cmap=sns.diverging_palette(25, 240, as_cmap=True),
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        ax=axis,
    )
    axis.set_title("Spearman association: log value vs. SHAP contribution")
    axis.set_xlabel("Class")
    axis.set_ylabel("")
    figure.suptitle("Direction of model attribution", y=0.985, fontsize=16)
    figure.text(
        0.5,
        0.944,
        "Positive values indicate higher concentration tends to raise predicted class probability; not a causal effect",
        ha="center",
        fontsize=10.0,
        color="#4A4F55",
    )
    figure.subplots_adjust(top=0.86, left=0.24, bottom=0.10)
    figure.savefig(FIG_DIR / "shap_direction_heatmap.png", dpi=300, bbox_inches="tight")
    figure.savefig(FIG_DIR / "shap_direction_heatmap.pdf", bbox_inches="tight")
    plt.close(figure)

    manifest = {
        "dataset": str(DATA_PATH.relative_to(ROOT)),
        "cohort_rows": len(data),
        "features": features,
        "permutation_validation": "five-fold citation-set grouped; up to 4000 held-out records per fold; four repeats",
        "shap_model": "full-cohort descriptive balanced random forest, 100 trees",
        "shap_sample_rows": len(shap_index),
        "caveat": "Correlated elements share or mask importance; SHAP direction is associative, not causal.",
    }
    (OUT_DIR / "interpretability_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Permutation importance:\n", permutation_summary.head(15).to_string(index=False))
    print("\nSHAP overall:\n", shap_overall.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
