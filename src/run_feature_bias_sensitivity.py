from __future__ import annotations

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
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline


ROOT = PROJECT_ROOT
DATA_PATH = ROOT / "data" / "processed" / "georoc_primary3_v0_1.parquet"
CONFIG_PATH = ROOT / "config" / "study_config.yaml"
OUT_DIR = ROOT / "reports" / "modeling"
FIG_DIR = ROOT / "figures"
RANDOM_SEED = 20260714
LABELS = ["ARC", "CIB", "OIB"]


def rf_pipeline(add_indicator: bool) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=add_indicator)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=80,
                    min_samples_leaf=4,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    broad = config["feature_sets"]["broad_v1"]
    immobile = config["feature_sets"]["immobile_v1"]

    data = pd.read_parquet(DATA_PATH)
    data = data.loc[data["model_ready_broad"] & data["model_ready_immobile"]].reset_index(drop=True)
    y = data["label_primary3"].astype(str)

    inputs = {
        "broad_values": (np.log10(data[broad].where(data[broad] > 0)), False),
        "broad_values_plus_missing": (np.log10(data[broad].where(data[broad] > 0)), True),
        "immobile_values": (np.log10(data[immobile].where(data[immobile] > 0)), False),
        "broad_missingness_only": (data[broad].isna().astype(float), False),
    }
    strategies = {
        "citation_set_grouped": data["citation_set"],
        "location_root_grouped": data["location_root"],
    }

    rows: list[dict[str, object]] = []
    for strategy_name, groups in strategies.items():
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        splits = list(splitter.split(data, y, groups))
        for input_name, (X, add_indicator) in inputs.items():
            for fold, (train_index, test_index) in enumerate(splits, start=1):
                model = rf_pipeline(add_indicator=add_indicator)
                model.fit(X.iloc[train_index], y.iloc[train_index])
                predicted = model.predict(X.iloc[test_index])
                probabilities = model.predict_proba(X.iloc[test_index])
                classes = list(model.classes_)
                probabilities = np.column_stack(
                    [probabilities[:, classes.index(label)] for label in LABELS]
                )
                y_test = y.iloc[test_index]
                rows.append(
                    {
                        "cv_strategy": strategy_name,
                        "input_spec": input_name,
                        "fold": fold,
                        "train_n": len(train_index),
                        "test_n": len(test_index),
                        "accuracy": accuracy_score(y_test, predicted),
                        "balanced_accuracy": balanced_accuracy_score(y_test, predicted),
                        "macro_f1": f1_score(y_test, predicted, average="macro"),
                        "weighted_f1": f1_score(y_test, predicted, average="weighted"),
                        "log_loss": log_loss(y_test, probabilities, labels=LABELS),
                    }
                )

    fold_metrics = pd.DataFrame(rows)
    fold_metrics.to_csv(OUT_DIR / "feature_bias_sensitivity_fold_metrics.csv", index=False)
    summary = (
        fold_metrics.groupby(["cv_strategy", "input_spec"])
        .agg(
            n_folds=("fold", "size"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_sd=("accuracy", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_sd=("balanced_accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_sd=("macro_f1", "std"),
            log_loss_mean=("log_loss", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(OUT_DIR / "feature_bias_sensitivity_summary.csv", index=False)

    display_names = {
        "broad_values": "Broad values",
        "broad_values_plus_missing": "Broad values + missingness",
        "immobile_values": "Immobile values",
        "broad_missingness_only": "Missingness only",
    }
    strategy_names = {
        "citation_set_grouped": "Citation set",
        "location_root_grouped": "Location root",
    }
    palette = {
        "Broad values": "#35618F",
        "Broad values + missingness": "#6E8FB0",
        "Immobile values": "#7A8B45",
        "Missingness only": "#C9794A",
    }
    plot_data = summary.copy()
    plot_data["Input"] = plot_data["input_spec"].map(display_names)
    plot_data["Validation"] = plot_data["cv_strategy"].map(strategy_names)

    sns.set_theme(style="whitegrid", context="notebook")
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.7), sharey=True)
    for axis, metric, title in zip(
        axes,
        ["balanced_accuracy_mean", "macro_f1_mean"],
        ["Balanced Accuracy", "Macro F1"],
    ):
        sns.barplot(
            data=plot_data,
            x="Validation",
            y=metric,
            hue="Input",
            hue_order=list(palette),
            palette=palette,
            ax=axis,
        )
        axis.set_title(title)
        axis.set_xlabel("")
        axis.set_ylabel("Score" if axis is axes[0] else "")
        axis.set_ylim(0, 1)
        axis.legend_.remove()
    handles, labels = axes[0].get_legend_handles_labels()
    figure.suptitle("Feature-set and missingness sensitivity", y=0.985, fontsize=16)
    figure.text(
        0.5,
        0.935,
        "Common cohort; n = 43,005; balanced random forest; five grouped folds",
        ha="center",
        fontsize=10.5,
        color="#4A4F55",
    )
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.895), ncol=4, frameon=False)
    figure.subplots_adjust(top=0.78, bottom=0.12, wspace=0.15)
    figure.savefig(FIG_DIR / "feature_bias_sensitivity.png", dpi=300, bbox_inches="tight")
    figure.savefig(FIG_DIR / "feature_bias_sensitivity.pdf", bbox_inches="tight")
    plt.close(figure)

    manifest = {
        "dataset": str(DATA_PATH.relative_to(ROOT)),
        "common_cohort_rows": len(data),
        "labels": LABELS,
        "broad_feature_count": len(broad),
        "immobile_feature_count": len(immobile),
        "random_seed": RANDOM_SEED,
        "note": "Values-only models omit explicit missingness indicators; missingness-only is a bias diagnostic.",
    }
    (OUT_DIR / "feature_bias_sensitivity_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

