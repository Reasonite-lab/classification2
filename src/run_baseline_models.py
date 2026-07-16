from __future__ import annotations

import json
import os
import time
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
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = PROJECT_ROOT
DATA_PATH = ROOT / "data" / "processed" / "georoc_primary3_v0_1.parquet"
CONFIG_PATH = ROOT / "config" / "study_config.yaml"
OUT_DIR = ROOT / "reports" / "modeling"
FIG_DIR = ROOT / "figures"
RANDOM_SEED = 20260714
LABELS = ["ARC", "CIB", "OIB"]


def make_models() -> dict[str, Pipeline]:
    return {
        "dummy_prior": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", DummyClassifier(strategy="prior", random_state=RANDOM_SEED)),
            ]
        ),
        "logistic_balanced": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        solver="lbfgs",
                        max_iter=1500,
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "random_forest_balanced": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=120,
                        min_samples_leaf=4,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
    }


def cv_strategies(data: pd.DataFrame, y: pd.Series):
    return {
        "random_stratified": (
            StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED),
            None,
        ),
        "citation_set_grouped": (
            StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED),
            data["citation_set"],
        ),
        "location_root_grouped": (
            StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED),
            data["location_root"],
        ),
        "citation_overlap_conservative": (
            StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED),
            data["citation_overlap_component"],
        ),
    }


def evaluate() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    features = config["feature_sets"]["broad_v1"]
    data = pd.read_parquet(DATA_PATH)
    data = data.loc[data["model_ready_broad"]].reset_index(drop=True)
    X = data[features].copy()
    X = np.log10(X.where(X > 0))
    y = data["label_primary3"].astype(str)

    if sorted(y.unique()) != LABELS:
        raise RuntimeError(f"Unexpected labels: {sorted(y.unique())}")

    models = make_models()
    fold_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for strategy_name, (splitter, groups) in cv_strategies(data, y).items():
        split_iter = splitter.split(X, y, groups) if groups is not None else splitter.split(X, y)
        for fold, (train_index, test_index) in enumerate(split_iter, start=1):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            for model_name, model_template in models.items():
                model = clone(model_template)
                start = time.perf_counter()
                model.fit(X_train, y_train)
                elapsed = time.perf_counter() - start
                predicted = model.predict(X_test)
                probabilities = model.predict_proba(X_test)
                model_classes = list(model.classes_)
                reordered_probabilities = np.column_stack(
                    [probabilities[:, model_classes.index(label)] for label in LABELS]
                )

                fold_rows.append(
                    {
                        "cv_strategy": strategy_name,
                        "fold": fold,
                        "model": model_name,
                        "train_n": len(train_index),
                        "test_n": len(test_index),
                        "fit_seconds": round(elapsed, 4),
                        "accuracy": accuracy_score(y_test, predicted),
                        "balanced_accuracy": balanced_accuracy_score(y_test, predicted),
                        "macro_f1": f1_score(y_test, predicted, average="macro"),
                        "weighted_f1": f1_score(y_test, predicted, average="weighted"),
                        "log_loss": log_loss(y_test, reordered_probabilities, labels=LABELS),
                    }
                )

                prediction_frame = pd.DataFrame(
                    {
                        "record_id": data.iloc[test_index]["record_id"].to_numpy(),
                        "cv_strategy": strategy_name,
                        "fold": fold,
                        "model": model_name,
                        "actual": y_test.to_numpy(),
                        "predicted": predicted,
                        "prob_arc": reordered_probabilities[:, 0],
                        "prob_cib": reordered_probabilities[:, 1],
                        "prob_oib": reordered_probabilities[:, 2],
                    }
                )
                prediction_frames.append(prediction_frame)

    fold_metrics = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    fold_metrics.to_csv(OUT_DIR / "baseline_fold_metrics.csv", index=False)
    predictions.to_parquet(OUT_DIR / "baseline_oof_predictions.parquet", index=False)

    overall_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    for (strategy, model_name), frame in predictions.groupby(["cv_strategy", "model"]):
        y_true = frame["actual"]
        y_pred = frame["predicted"]
        probability = frame[["prob_arc", "prob_cib", "prob_oib"]].to_numpy()
        overall_rows.append(
            {
                "cv_strategy": strategy,
                "model": model_name,
                "n": len(frame),
                "accuracy": accuracy_score(y_true, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                "macro_f1": f1_score(y_true, y_pred, average="macro"),
                "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
                "log_loss": log_loss(y_true, probability, labels=LABELS),
            }
        )
        report = classification_report(
            y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
        )
        for label in LABELS:
            class_rows.append(
                {
                    "cv_strategy": strategy,
                    "model": model_name,
                    "label": label,
                    **{key: report[label][key] for key in ("precision", "recall", "f1-score", "support")},
                }
            )

    overall = pd.DataFrame(overall_rows).sort_values(["cv_strategy", "macro_f1"], ascending=[True, False])
    per_class = pd.DataFrame(class_rows)
    overall.to_csv(OUT_DIR / "baseline_overall_metrics.csv", index=False)
    per_class.to_csv(OUT_DIR / "baseline_per_class_metrics.csv", index=False)

    sns.set_theme(style="whitegrid", context="notebook")
    strategy_labels = {
        "citation_overlap_conservative": "Citation overlap\n(conservative)",
        "citation_set_grouped": "Citation set",
        "location_root_grouped": "Location root",
        "random_stratified": "Random rows",
    }
    model_labels = {
        "random_forest_balanced": "Random forest",
        "logistic_balanced": "Logistic regression",
        "dummy_prior": "Class prior",
    }
    palette = {
        "Random forest": "#35618F",
        "Logistic regression": "#C9794A",
        "Class prior": "#9AA1A8",
    }
    plot_metrics = overall.melt(
        id_vars=["cv_strategy", "model"],
        value_vars=["balanced_accuracy", "macro_f1"],
        var_name="metric",
        value_name="score",
    )
    plot_metrics["strategy_label"] = plot_metrics["cv_strategy"].map(strategy_labels)
    plot_metrics["model_label"] = plot_metrics["model"].map(model_labels)
    strategy_order = list(strategy_labels.values())
    model_order = list(model_labels.values())
    figure, axes = plt.subplots(1, 2, figsize=(14, 6.2), sharey=True)
    for axis, metric in zip(axes, ["balanced_accuracy", "macro_f1"]):
        subset = plot_metrics[plot_metrics["metric"] == metric]
        sns.barplot(
            data=subset,
            x="strategy_label",
            y="score",
            hue="model_label",
            order=strategy_order,
            hue_order=model_order,
            palette=palette,
            ax=axis,
        )
        axis.set_title(metric.replace("_", " ").title())
        axis.set_xlabel("")
        axis.set_ylabel("Score" if axis is axes[0] else "")
        axis.set_ylim(0, 1)
        axis.legend_.remove()
    handles, labels = axes[0].get_legend_handles_labels()
    figure.suptitle("Baseline performance across cross-validation strategies", y=0.985, fontsize=16)
    figure.text(
        0.5,
        0.938,
        "GEOROC primary-3; n = 54,012; five-fold out-of-fold evaluation; 22 broad geochemical features",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#4A4F55",
    )
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=3, frameon=False)
    figure.subplots_adjust(top=0.80, bottom=0.18, wspace=0.14)
    figure.savefig(FIG_DIR / "baseline_cv_performance.png", dpi=300, bbox_inches="tight")
    figure.savefig(FIG_DIR / "baseline_cv_performance.pdf", bbox_inches="tight")
    plt.close(figure)

    citation_predictions = predictions[predictions["cv_strategy"] == "citation_set_grouped"]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for axis, model_name in zip(axes, models):
        frame = citation_predictions[citation_predictions["model"] == model_name]
        matrix = confusion_matrix(frame["actual"], frame["predicted"], labels=LABELS, normalize="true")
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            vmin=0,
            vmax=1,
            xticklabels=LABELS,
            yticklabels=LABELS,
            cbar=model_name == list(models)[-1],
            ax=axis,
        )
        axis.set_title(model_name.replace("_", " ").title())
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual" if axis is axes[0] else "")
    figure.suptitle("Citation-grouped out-of-fold confusion matrices", y=0.99, fontsize=16)
    figure.text(
        0.5,
        0.935,
        "Row-normalized recall; GEOROC primary-3; n = 54,012; five folds",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#4A4F55",
    )
    figure.subplots_adjust(top=0.80, wspace=0.12)
    figure.savefig(FIG_DIR / "baseline_confusion_citation.png", dpi=300, bbox_inches="tight")
    figure.savefig(FIG_DIR / "baseline_confusion_citation.pdf", bbox_inches="tight")
    plt.close(figure)

    summary = {
        "dataset": str(DATA_PATH.relative_to(ROOT)),
        "rows": len(data),
        "features": features,
        "labels": LABELS,
        "random_seed": RANDOM_SEED,
        "models": list(models),
        "cv_strategies": list(cv_strategies(data, y)),
        "interpretation": "Provisional baseline only; MORB is not present and hyperparameters are not tuned.",
    }
    (OUT_DIR / "baseline_run_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(overall.to_string(index=False))


if __name__ == "__main__":
    evaluate()
