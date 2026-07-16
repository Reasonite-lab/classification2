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
from sklearn.metrics import confusion_matrix


OUT_DIR = ROOT / "reports" / "modeling"
FIG_DIR = ROOT / "figures"
LABELS = ["ARC", "CIB", "MORB", "OIB"]
PROB_COLUMNS = [f"prob_{label.lower()}" for label in LABELS]
BLUE = "#35618F"
ORANGE = "#C9794A"
OLIVE = "#7C8438"
PINK = "#C46A8A"
INK = "#34383D"
GRID = "#D9DCE0"


def calibration_metrics(frame: pd.DataFrame) -> tuple[float, float, float]:
    probability = frame[PROB_COLUMNS].to_numpy(float)
    actual_index = np.array([LABELS.index(value) for value in frame["actual"]])
    one_hot = np.eye(len(LABELS))[actual_index]
    brier = float(np.mean(np.sum((probability - one_hot) ** 2, axis=1)))
    predicted_index = probability.argmax(axis=1)
    confidence = probability.max(axis=1)
    correct = predicted_index == actual_index
    bins = np.linspace(0, 1, 11)
    bin_id = np.clip(np.digitize(confidence, bins, right=True) - 1, 0, 9)
    ece = 0.0
    for index in range(10):
        mask = bin_id == index
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return brier, float(ece), float(confidence.mean())


def class_calibration(frame: pd.DataFrame) -> pd.DataFrame:
    bins = np.linspace(0, 1, 11)
    rows: list[dict[str, object]] = []
    for label, column in zip(LABELS, PROB_COLUMNS, strict=True):
        probability = frame[column].to_numpy(float)
        observed = frame["actual"].eq(label).to_numpy(float)
        bin_id = np.clip(np.digitize(probability, bins, right=True) - 1, 0, 9)
        for index in range(10):
            mask = bin_id == index
            if not mask.any():
                continue
            rows.append(
                {
                    "label": label,
                    "bin": index,
                    "n": int(mask.sum()),
                    "mean_predicted_probability": float(probability[mask].mean()),
                    "observed_frequency": float(observed[mask].mean()),
                }
            )
    return pd.DataFrame(rows)


def save_validation_performance(overall: pd.DataFrame) -> None:
    strategies = [
        "random_stratified",
        "citation_set_grouped",
        "citation_overlap_conservative",
        "location_root_grouped",
    ]
    strategy_labels = ["Random", "Citation set", "Citation overlap", "Location"]
    models = ["random_forest_balanced", "logistic_balanced"]
    model_labels = ["Random forest", "Logistic regression"]
    colors = [BLUE, ORANGE]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.7), sharey=True)
    x = np.arange(len(strategies))
    width = 0.36
    for axis, metric, metric_label in zip(
        axes,
        ["macro_f1", "balanced_accuracy"],
        ["Macro-F1", "Balanced accuracy"],
        strict=True,
    ):
        for offset, model, model_label, color in zip(
            [-width / 2, width / 2], models, model_labels, colors, strict=True
        ):
            values = [
                float(
                    overall.loc[
                        overall["cv_strategy"].eq(strategy)
                        & overall["model"].eq(model),
                        metric,
                    ].iloc[0]
                )
                for strategy in strategies
            ]
            bars = axis.bar(
                x + offset,
                values,
                width,
                label=model_label,
                color=color,
                edgecolor=INK,
                linewidth=0.6,
            )
            axis.bar_label(bars, fmt="%.2f", fontsize=8.5, padding=2)
        axis.set_xticks(x, strategy_labels, rotation=18, ha="right")
        axis.set_ylim(0, 1.04)
        axis.set_ylabel(metric_label if axis is axes[0] else "")
        axis.set_title(metric_label)
        axis.grid(axis="y", color=GRID, linewidth=0.7)
        axis.grid(axis="x", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("PetDB four-class validation performance", fontsize=16, y=0.99)
    fig.text(
        0.5,
        0.935,
        "Five-fold out-of-fold evaluation; n = 6,148; 21 features observed in every class",
        ha="center",
        fontsize=10,
        color="#4A4F55",
    )
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.905), ncol=2, frameon=False)
    fig.subplots_adjust(top=0.80, bottom=0.22, wspace=0.12)
    fig.savefig(FIG_DIR / "petdb_primary4_validation_performance.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "petdb_primary4_validation_performance.pdf", bbox_inches="tight")
    plt.close(fig)


def save_confusion(predictions: pd.DataFrame) -> None:
    strategies = ["citation_overlap_conservative", "location_root_grouped"]
    titles = ["Citation-overlap grouped", "Location grouped"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.9))
    for axis, strategy, title in zip(axes, strategies, titles, strict=True):
        frame = predictions[
            predictions["cv_strategy"].eq(strategy)
            & predictions["model"].eq("random_forest_balanced")
        ]
        matrix = confusion_matrix(
            frame["actual"], frame["predicted"], labels=LABELS, normalize="true"
        )
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            vmin=0,
            vmax=1,
            xticklabels=LABELS,
            yticklabels=LABELS,
            linewidths=0.4,
            cbar=axis is axes[-1],
            ax=axis,
        )
        axis.set_title(title)
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("Actual class" if axis is axes[0] else "")
    fig.suptitle("PetDB four-class random-forest confusion matrices", fontsize=16, y=0.99)
    fig.text(
        0.5,
        0.93,
        "Row-normalized held-out recall; n = 6,148; five folds",
        ha="center",
        fontsize=10,
        color="#4A4F55",
    )
    fig.subplots_adjust(top=0.82, wspace=0.16)
    fig.savefig(FIG_DIR / "petdb_primary4_grouped_confusion.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "petdb_primary4_grouped_confusion.pdf", bbox_inches="tight")
    plt.close(fig)


def save_calibration(calibration: pd.DataFrame, summary_row: pd.Series) -> None:
    styles = {
        "ARC": (BLUE, "o", "-"),
        "CIB": (ORANGE, "s", "--"),
        "MORB": (OLIVE, "^", "-."),
        "OIB": (PINK, "D", ":"),
    }
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    ax.plot([0, 1], [0, 1], color=INK, linewidth=1.0, linestyle="--", label="Ideal")
    for label, (color, marker, linestyle) in styles.items():
        frame = calibration[calibration["label"].eq(label)]
        ax.plot(
            frame["mean_predicted_probability"],
            frame["observed_frequency"],
            color=color,
            marker=marker,
            markerfacecolor="white",
            markeredgewidth=1.0,
            linestyle=linestyle,
            linewidth=1.8,
            label=label,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed class frequency")
    ax.set_title("PetDB four-class reliability curves")
    ax.grid(color=GRID, linewidth=0.7)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.text(
        0.5,
        0.935,
        (
            "Citation-overlap-held-out random forest; 10 equal-width bins; "
            f"multiclass Brier = {summary_row['multiclass_brier']:.3f}; "
            f"top-label ECE = {summary_row['top_label_ece']:.3f}"
        ),
        ha="center",
        fontsize=9.7,
        color="#4A4F55",
    )
    fig.subplots_adjust(top=0.87, bottom=0.12)
    fig.savefig(FIG_DIR / "petdb_primary4_calibration.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "petdb_primary4_calibration.pdf", bbox_inches="tight")
    plt.close(fig)


def save_cross_database(cross: pd.DataFrame) -> None:
    cross = cross[cross["model"].eq("random_forest_balanced")].copy()
    directions = ["GEOROC_to_PetDB", "PetDB_to_GEOROC"]
    direction_labels = ["GEOROC → PetDB", "PetDB → GEOROC"]
    variants = ["broad_cib", "rift_aligned_cib"]
    variant_labels = ["Broad CIB", "Rift-aligned CIB"]
    colors = [BLUE, ORANGE]
    x = np.arange(len(directions))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.5), sharey=True)
    for axis, metric, title in zip(
        axes,
        ["macro_f1", "balanced_accuracy"],
        ["Macro-F1", "Balanced accuracy"],
        strict=True,
    ):
        for offset, variant, label, color in zip(
            [-width / 2, width / 2], variants, variant_labels, colors, strict=True
        ):
            values = [
                float(
                    cross.loc[
                        cross["direction"].eq(direction)
                        & cross["ontology_variant"].eq(variant),
                        metric,
                    ].iloc[0]
                )
                for direction in directions
            ]
            bars = axis.bar(
                x + offset,
                values,
                width,
                color=color,
                edgecolor=INK,
                linewidth=0.6,
                label=label,
            )
            axis.bar_label(bars, fmt="%.2f", fontsize=9, padding=2)
        axis.set_xticks(x, direction_labels)
        axis.set_ylim(0, 1.02)
        axis.set_title(title)
        axis.set_ylabel(title if axis is axes[0] else "")
        axis.grid(axis="y", color=GRID, linewidth=0.7)
        axis.grid(axis="x", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Cross-database three-class transfer", fontsize=16, y=0.99)
    fig.text(
        0.5,
        0.935,
        "ARC/CIB/OIB only; 21 shared features; target database is never used for fitting",
        ha="center",
        fontsize=10,
        color="#4A4F55",
    )
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.90), ncol=2, frameon=False)
    fig.subplots_adjust(top=0.79, bottom=0.17, wspace=0.12)
    fig.savefig(FIG_DIR / "cross_database_transfer.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "cross_database_transfer.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_parquet(OUT_DIR / "petdb_primary4_oof_predictions.parquet")
    overall = pd.read_csv(OUT_DIR / "petdb_primary4_overall_metrics.csv")
    rows: list[dict[str, object]] = []
    for (strategy, model), frame in predictions.groupby(["cv_strategy", "model"]):
        brier, ece, mean_confidence = calibration_metrics(frame)
        rows.append(
            {
                "cv_strategy": strategy,
                "model": model,
                "n": len(frame),
                "multiclass_brier": brier,
                "top_label_ece": ece,
                "mean_top_label_confidence": mean_confidence,
            }
        )
    calibration_summary = pd.DataFrame(rows)
    calibration_summary.to_csv(
        OUT_DIR / "petdb_primary4_calibration_summary.csv", index=False
    )
    calibration_frame = predictions[
        predictions["cv_strategy"].eq("citation_overlap_conservative")
        & predictions["model"].eq("random_forest_balanced")
    ]
    by_class = class_calibration(calibration_frame)
    by_class.to_csv(
        OUT_DIR / "petdb_primary4_calibration_by_class.csv", index=False
    )
    selected_summary = calibration_summary[
        calibration_summary["cv_strategy"].eq("citation_overlap_conservative")
        & calibration_summary["model"].eq("random_forest_balanced")
    ].iloc[0]

    sns.set_theme(style="whitegrid", context="notebook")
    save_validation_performance(overall)
    save_confusion(predictions)
    save_calibration(by_class, selected_summary)
    cross = pd.read_csv(OUT_DIR / "cross_database_overall_metrics.csv")
    save_cross_database(cross)

    manifest = {
        "calibration_cohort": (
            "PetDB primary4 common-21 out-of-fold predictions under each validation strategy"
        ),
        "primary_calibration": {
            "cv_strategy": "citation_overlap_conservative",
            "model": "random_forest_balanced",
            "multiclass_brier": float(selected_summary["multiclass_brier"]),
            "top_label_ece": float(selected_summary["top_label_ece"]),
            "mean_top_label_confidence": float(
                selected_summary["mean_top_label_confidence"]
            ),
        },
        "figure_contracts": {
            "validation_performance": "grouped bars from zero; exact labels; two model colors",
            "grouped_confusion": "two row-normalized heatmaps on identical 0–1 scale",
            "calibration": "four class reliability curves plus dark-neutral ideal line",
            "cross_database_transfer": "grouped bars from zero; direction × ontology variant",
        },
        "caveat": (
            "Calibration is descriptive out-of-fold reliability without post-hoc recalibration. "
            "Cross-database transfer excludes MORB and mixes database with ontology shift."
        ),
    }
    (OUT_DIR / "petdb_primary4_figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(calibration_summary.to_string(index=False))


if __name__ == "__main__":
    main()
