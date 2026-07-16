from __future__ import annotations

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


BLUE = "#35618F"
ORANGE = "#C9794A"
INK = "#34383D"
GRID = "#D9DCE0"
SOURCE_ORDER = ["PANGAEA.707361", "PANGAEA.727391"]
SOURCE_LABELS = ["DSDP 24-238\n(n = 2)", "MAR Ascension\n(n = 16)"]
MODEL_ORDER = ["random_forest_balanced", "logistic_balanced"]
MODEL_LABELS = ["Random forest", "Logistic regression"]


def main() -> None:
    predictions = pd.read_csv(
        ROOT / "reports" / "modeling" / "pangaea_morb_external_predictions.csv"
    )
    summary = pd.read_csv(
        ROOT / "reports" / "modeling" / "pangaea_morb_external_summary.csv"
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.4), sharey=True)
    rng = np.random.default_rng(20260715)
    for axis, model_name, model_label in zip(axes, MODEL_ORDER, MODEL_LABELS, strict=True):
        frame = predictions[predictions["model"].eq(model_name)]
        for source_index, source in enumerate(SOURCE_ORDER):
            source_frame = frame[frame["source_dataset"].eq(source)].copy()
            jitter = rng.uniform(-0.13, 0.13, len(source_frame))
            correct = source_frame["predicted"].eq("MORB").to_numpy()
            axis.scatter(
                source_index + jitter[correct],
                source_frame.loc[correct, "prob_morb"],
                s=46,
                color=BLUE,
                edgecolor="white",
                linewidth=0.6,
                alpha=0.88,
                zorder=3,
            )
            axis.scatter(
                source_index + jitter[~correct],
                source_frame.loc[~correct, "prob_morb"],
                s=62,
                marker="X",
                color=ORANGE,
                edgecolor=INK,
                linewidth=0.7,
                zorder=4,
            )
            for x_value, row in zip(jitter[~correct], source_frame.loc[~correct].itertuples(), strict=True):
                axis.annotate(
                    f"{row.sample_name} → {row.predicted}",
                    (source_index + x_value, row.prob_morb),
                    xytext=(5, -13),
                    textcoords="offset points",
                    fontsize=7.8,
                    color=INK,
                )
        overall = summary[
            summary["model"].eq(model_name)
            & summary["source_dataset"].eq("ALL_ACCEPTED")
        ].iloc[0]
        axis.set_title(f"{model_label}\nMORB recall = {overall['morb_recall']:.3f}")
        axis.set_xticks(range(len(SOURCE_ORDER)), SOURCE_LABELS)
        axis.set_ylim(0, 1.03)
        axis.set_xlim(-0.45, 1.45)
        axis.set_ylabel("Predicted MORB probability" if axis is axes[0] else "")
        axis.grid(axis="y", color=GRID, linewidth=0.7)
        axis.grid(axis="x", visible=False)
    fig.suptitle("Pilot external MORB stress test", fontsize=16, y=0.985)
    fig.text(
        0.5,
        0.89,
        "PetDB-trained 10-major-element models; independent PANGAEA glass samples; one true class",
        ha="center",
        fontsize=9.7,
        color="#4A4F55",
    )
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=BLUE, markersize=7, label="Predicted MORB"),
        plt.Line2D([], [], marker="X", linestyle="", color=ORANGE, markeredgecolor=INK, markersize=8, label="Predicted other class"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.845), ncol=2, frameon=False)
    fig.subplots_adjust(top=0.72, bottom=0.16, wspace=0.12)
    figure_dir = ROOT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "pangaea_morb_external_probabilities.png", dpi=300, bbox_inches="tight")
    fig.savefig(figure_dir / "pangaea_morb_external_probabilities.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved PANGAEA external MORB probability figure.")


if __name__ == "__main__":
    main()
