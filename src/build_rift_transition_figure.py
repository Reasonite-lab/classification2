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


COLORS = {
    "ARC": "#A35D52",
    "CIB": "#D7A84B",
    "MORB": "#3E7C91",
    "OIB": "#80558C",
}
INK = "#34383D"
GRID = "#D9DCE0"
ORANGE = "#C9794A"


def main() -> None:
    geography = pd.read_csv(
        ROOT / "reports" / "modeling" / "east_african_rift_stress_geography.csv"
    )
    feature_shift = pd.read_csv(
        ROOT / "reports" / "modeling" / "east_african_rift_feature_shift.csv"
    )
    country = geography[
        geography["model"].eq("random_forest_balanced")
        & geography["actual"].eq("CIB")
        & ~geography["country"].eq("<UNRESOLVED>")
        & geography["n"].ge(4)
    ].copy()
    country["country"] = country["country"].replace(
        {"DEMOCRATIC REPUBLIC OF THE CONGO": "DR CONGO"}
    )
    country = country.sort_values("mean_prob_oib", ascending=True)

    shift = feature_shift[
        feature_shift["ears_closer_to_oib_than_other_cib"].astype(bool)
        & feature_shift["oib_proximity_gain_global_mad"].gt(0)
    ].nlargest(10, "oib_proximity_gain_global_mad").sort_values(
        "oib_proximity_gain_global_mad"
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.8), gridspec_kw={"wspace": 0.34})
    axis = axes[0]
    y = np.arange(len(country))
    left = np.zeros(len(country))
    for label in ["ARC", "CIB", "MORB", "OIB"]:
        values = country[f"mean_prob_{label.lower()}"].to_numpy()
        axis.barh(
            y,
            values,
            left=left,
            color=COLORS[label],
            edgecolor="white",
            linewidth=0.5,
            label=label,
        )
        left += values
    axis.set_yticks(y, [f"{row.country.title()}  (n={row.n})" for row in country.itertuples()])
    axis.set_xlim(0, 1)
    axis.set_xlabel("Mean predicted class probability")
    axis.set_title("A  East African Rift CIB affinity by country")
    axis.grid(axis="x", color=GRID, linewidth=0.7)
    axis.grid(axis="y", visible=False)
    axis.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.20), frameon=False)

    axis = axes[1]
    bars = axis.barh(
        shift["feature"],
        shift["oib_proximity_gain_global_mad"],
        color=ORANGE,
        edgecolor=INK,
        linewidth=0.5,
    )
    for bar, value in zip(bars, shift["oib_proximity_gain_global_mad"], strict=True):
        axis.text(
            value + 0.03,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            fontsize=8.5,
            color=INK,
        )
    axis.set_xlim(0, max(shift["oib_proximity_gain_global_mad"].max() * 1.22, 1))
    axis.set_xlabel("Gain in proximity to OIB median (global MAD units)")
    axis.set_title("B  Elements shifting from other CIB toward OIB")
    axis.grid(axis="x", color=GRID, linewidth=0.7)
    axis.grid(axis="y", visible=False)

    fig.suptitle("East African Rift as a rift–plume geochemical transition", fontsize=16, y=0.99)
    fig.text(
        0.5,
        0.942,
        "Random forest trained with all East African Rift samples withheld; probabilities are affinities, not relabeling",
        ha="center",
        fontsize=9.8,
        color="#4A4F55",
    )
    fig.text(
        0.5,
        0.035,
        "Feature shifts use log10 medians; positive values indicate that East African Rift CIB is closer to the OIB median than other CIB.",
        ha="center",
        fontsize=8.6,
        color="#4A4F55",
    )
    fig.subplots_adjust(top=0.85, bottom=0.20, left=0.17, right=0.98)
    figure_dir = ROOT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "east_african_rift_transition.png", dpi=300, bbox_inches="tight")
    fig.savefig(figure_dir / "east_african_rift_transition.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved East African Rift transition figure.")


if __name__ == "__main__":
    main()
