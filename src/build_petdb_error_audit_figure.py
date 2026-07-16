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


def short_location(value: str, limit: int = 42) -> str:
    cleaned = value.replace('"', "").replace("COUNTRY | ", "").replace("OCEAN | ", "")
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1] + "…"


def main() -> None:
    group_risk = pd.read_csv(ROOT / "reports" / "modeling" / "petdb_primary4_error_group_risk.csv")
    pair = pd.read_csv(ROOT / "reports" / "modeling" / "petdb_primary4_error_pair_summary.csv")

    locations = group_risk[
        group_risk["cv_strategy"].eq("location_root_grouped") & group_risk["n"].ge(10)
    ].nlargest(12, ["error_rate", "n"]).sort_values("error_rate")
    locations = locations.assign(label=locations["group_value"].astype(str).map(short_location))

    errors = pair[pair["actual"].ne(pair["predicted"])].copy()
    errors["pair"] = errors["actual"] + "→" + errors["predicted"]
    top_pairs = (
        errors.groupby("pair")["share_of_actual"].max().nlargest(8).index.tolist()
    )
    pivot = errors[errors["pair"].isin(top_pairs)].pivot(
        index="pair", columns="cv_strategy", values="share_of_actual"
    ).fillna(0)
    pivot["max_share"] = pivot.max(axis=1)
    pivot = pivot.sort_values("max_share").drop(columns="max_share")

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 7.2), gridspec_kw={"wspace": 0.43})
    axis = axes[0]
    colors = [ORANGE if value == "EAST AFRICAN RIFT" else BLUE for value in locations["group_value"]]
    bars = axis.barh(
        locations["label"], locations["error_rate"], color=colors, edgecolor=INK, linewidth=0.5
    )
    for bar, rate, n in zip(bars, locations["error_rate"], locations["n"], strict=True):
        axis.text(
            min(rate + 0.018, 1.015),
            bar.get_y() + bar.get_height() / 2,
            f"{rate:.0%}  (n={n})",
            va="center",
            fontsize=8.4,
            color=INK,
        )
    axis.set_xlim(0, 1.18)
    axis.set_xlabel("Out-of-fold error rate")
    axis.set_title("A  Highest-risk held-out locations")
    axis.grid(axis="x", color=GRID, linewidth=0.7)
    axis.grid(axis="y", visible=False)

    axis = axes[1]
    y = np.arange(len(pivot))
    height = 0.36
    citation_values = pivot.get("citation_overlap_conservative", pd.Series(0, index=pivot.index))
    location_values = pivot.get("location_root_grouped", pd.Series(0, index=pivot.index))
    citation_bars = axis.barh(
        y - height / 2,
        citation_values,
        height,
        color=BLUE,
        edgecolor=INK,
        linewidth=0.5,
        label="Citation-overlap grouped",
    )
    location_bars = axis.barh(
        y + height / 2,
        location_values,
        height,
        color=ORANGE,
        edgecolor=INK,
        linewidth=0.5,
        label="Location grouped",
    )
    for bars, values in [(citation_bars, citation_values), (location_bars, location_values)]:
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                value + 0.012,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.0%}",
                va="center",
                fontsize=8.2,
                color=INK,
            )
    axis.set_yticks(y, pivot.index)
    axis.set_xlim(0, 0.86)
    axis.set_xlabel("Share of the actual class assigned to predicted class")
    axis.set_title("B  Largest directed class errors")
    axis.grid(axis="x", color=GRID, linewidth=0.7)
    axis.grid(axis="y", visible=False)
    axis.legend(loc="lower right", frameon=False, fontsize=8.8)

    fig.suptitle("PetDB four-class error audit", fontsize=16, y=0.99)
    fig.text(
        0.5,
        0.945,
        "Balanced random forest; 6,148 samples; 21 features; held-out predictions only",
        ha="center",
        fontsize=10,
        color="#4A4F55",
    )
    fig.subplots_adjust(top=0.86, bottom=0.13, left=0.18, right=0.98)
    figure_dir = ROOT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "petdb_primary4_error_audit.png", dpi=300, bbox_inches="tight")
    fig.savefig(figure_dir / "petdb_primary4_error_audit.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved PetDB error-audit figure.")


if __name__ == "__main__":
    main()
