from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODELING = ROOT / "reports" / "modeling"
OUT = ROOT / "figures"
LABELS = ["ARC", "CIB", "MORB", "OIB"]
MODELS = ["logistic_balanced", "random_forest_balanced"]
MODEL_LABELS = {
    "logistic_balanced": "Balanced logistic",
    "random_forest_balanced": "Balanced random forest",
}
COLORS = {"logistic_balanced": "#4C78A8", "random_forest_balanced": "#F58518"}


def heatmap(ax: plt.Axes, data: pd.DataFrame, model: str, panel: str) -> None:
    selected = data.loc[data["model"].eq(model)]
    matrix = (
        selected.pivot(index="actual", columns="predicted", values="row_share")
        .reindex(index=LABELS, columns=LABELS)
        .fillna(0)
        .to_numpy()
    )
    counts = (
        selected.pivot(index="actual", columns="predicted", values="n")
        .reindex(index=LABELS, columns=LABELS)
        .fillna(0)
        .to_numpy()
    )
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    for row in range(len(LABELS)):
        for column in range(len(LABELS)):
            color = "white" if matrix[row, column] > 0.55 else "#202124"
            ax.text(
                column,
                row,
                f"{matrix[row, column]:.2f}\n(n={int(counts[row, column])})",
                ha="center",
                va="center",
                fontsize=8,
                color=color,
            )
    ax.set_xticks(range(len(LABELS)), LABELS)
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Tectonic setting")
    ax.set_title(f"{panel}  {MODEL_LABELS[model]}", loc="left", weight="bold")
    return image


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    confusion = pd.read_csv(MODELING / "multiclass_external_confusion.csv")
    per_class = pd.read_csv(MODELING / "multiclass_external_per_class.csv")
    transition = pd.read_csv(MODELING / "multiclass_external_backarc_summary.csv")
    predictions = pd.read_csv(MODELING / "multiclass_external_predictions.csv")
    one_model = predictions.loc[
        predictions["model"].eq("random_forest_balanced")
    ]
    strict_n = len(one_model)
    province_counts = one_model.groupby("actual_label")["province_proxy"].nunique()

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})
    figure, axes = plt.subplots(2, 2, figsize=(10.4, 8.0), constrained_layout=True)
    image = heatmap(axes[0, 0], confusion, MODELS[0], "A")
    heatmap(axes[0, 1], confusion, MODELS[1], "B")
    colorbar = figure.colorbar(image, ax=axes[0, :], shrink=0.72, pad=0.02)
    colorbar.set_label("Row-normalized share")

    x = np.arange(len(LABELS))
    width = 0.35
    for offset, model in zip([-width / 2, width / 2], MODELS):
        values = (
            per_class.loc[per_class["model"].eq(model)]
            .set_index("label")
            .reindex(LABELS)["recall"]
        )
        axes[1, 0].bar(
            x + offset,
            values,
            width,
            label=MODEL_LABELS[model],
            color=COLORS[model],
        )
        for xpos, value in zip(x + offset, values):
            axes[1, 0].text(xpos, value + 0.025, f"{value:.2f}", ha="center", fontsize=8)
    axes[1, 0].set_xticks(x, LABELS)
    axes[1, 0].set_ylim(0, 1.12)
    axes[1, 0].set_ylabel("External recall")
    axes[1, 0].set_title("C  Recall by setting", loc="left", weight="bold")
    axes[1, 0].grid(axis="y", alpha=0.2)

    for offset, model in zip([-width / 2, width / 2], MODELS):
        row = transition.loc[transition["model"].eq(model)].iloc[0]
        values = [row[f"predicted_share_{label.lower()}"] for label in LABELS]
        axes[1, 1].bar(
            x + offset,
            values,
            width,
            label=MODEL_LABELS[model],
            color=COLORS[model],
        )
    axes[1, 1].set_xticks(x, LABELS)
    axes[1, 1].set_ylim(0, 1.02)
    axes[1, 1].set_ylabel("Predicted share (unscored)")
    axes[1, 1].set_title(
        "D  Vate Trough initial back-arc rift", loc="left", weight="bold"
    )
    axes[1, 1].legend(frameon=False, fontsize=8, loc="upper right")
    axes[1, 1].grid(axis="y", alpha=0.2)

    figure.suptitle(
        "Four-setting external transfer and a back-arc mechanism pressure test",
        fontsize=13,
        weight="bold",
    )
    figure.text(
        0.5,
        -0.015,
        f"Common 18-element model; {strict_n} strict rows. Province proxies: "
        + ", ".join(f"{label}={int(province_counts[label])}" for label in LABELS)
        + ". Vate Trough is not scored as ARC or MORB.",
        ha="center",
        fontsize=8,
        color="#4A4A4A",
    )
    figure.savefig(OUT / "multiclass_external_validation.png", dpi=300, bbox_inches="tight")
    figure.savefig(OUT / "multiclass_external_validation.pdf", bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
