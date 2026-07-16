from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
SEED = 20260714
COLORS = {
    "ARC": "#D55E00",
    "CIB": "#CC79A7",
    "MORB": "#0072B2",
    "OIB": "#009E73",
}
MODEL_LABELS = {
    "logistic_balanced": "Compositional logistic",
    "random_forest_balanced": "Random forest",
}


def finite_ratio(frame: pd.DataFrame, numerator: str, denominator: str) -> pd.Series:
    ratio = frame[numerator] / frame[denominator]
    return np.log10(ratio.where(ratio > 0))


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(
        ROOT / "reports" / "modeling" / "figshare_morb_external_predictions.csv"
    )
    predictions = predictions.loc[predictions["strict_external_candidate"]].copy()
    external = pd.read_parquet(
        ROOT / "data" / "processed" / "figshare_morb_external_v0_1.parquet"
    )
    external = external.loc[external["strict_external_candidate"]].copy()
    train = pd.read_parquet(
        ROOT / "data" / "processed" / "petdb_primary4_v0_1.parquet"
    )

    train["log_nb_y"] = finite_ratio(train, "NB(PPM)", "Y(PPM)")
    train["log_sr_y"] = finite_ratio(train, "SR(PPM)", "Y(PPM)")
    external["log_nb_y"] = finite_ratio(external, "NB(PPM)", "Y(PPM)")
    external["log_sr_y"] = finite_ratio(external, "SR(PPM)", "Y(PPM)")

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "figure.dpi": 160,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.15), constrained_layout=True)

    ax = axes[0]
    rng = np.random.default_rng(SEED)
    for index, model_name in enumerate(MODEL_LABELS):
        frame = predictions.loc[predictions["model"].eq(model_name)].sort_values(
            "prob_morb"
        )
        jitter = rng.normal(0, 0.045, len(frame))
        correct = frame["predicted"].eq("MORB").to_numpy()
        ax.scatter(
            np.full(len(frame), index) + jitter,
            frame["prob_morb"],
            s=28,
            c=np.where(correct, COLORS["MORB"], COLORS["ARC"]),
            edgecolor="white",
            linewidth=0.45,
            alpha=0.85,
            zorder=3,
        )
        ax.plot(
            [index - 0.18, index + 0.18],
            [frame["prob_morb"].median()] * 2,
            color="black",
            linewidth=1.8,
            zorder=4,
        )
    ax.axhline(0.5, color="#666666", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(MODEL_LABELS)), MODEL_LABELS.values())
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Predicted MORB probability")
    ax.set_title(
        "a  Provisional external SWIR cohort (n = 36)",
        loc="left",
        fontweight="bold",
    )
    ax.text(
        0.02,
        0.04,
        "Blue: MORB prediction\nOrange: ARC prediction\nBlack bar: median",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
    )
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)

    ax = axes[1]
    for label in ["ARC", "CIB", "MORB", "OIB"]:
        frame = train.loc[train["label_primary4"].eq(label), ["log_nb_y", "log_sr_y"]]
        frame = frame.dropna()
        if len(frame) > 450:
            frame = frame.sample(450, random_state=SEED)
        ax.scatter(
            frame["log_nb_y"],
            frame["log_sr_y"],
            s=8,
            color=COLORS[label],
            alpha=0.15,
            linewidth=0,
            label=f"PetDB {label}",
        )
    ax.scatter(
        external["log_nb_y"],
        external["log_sr_y"],
        marker="D",
        s=32,
        facecolor="black",
        edgecolor="white",
        linewidth=0.5,
        alpha=0.82,
        label="External SWIR",
        zorder=4,
    )
    misclassified_names = set(
        predictions.loc[
            predictions["model"].eq("logistic_balanced")
            & predictions["predicted"].ne("MORB"),
            "Sample",
        ]
    )
    misclassified = external.loc[external["Sample"].isin(misclassified_names)]
    ax.scatter(
        misclassified["log_nb_y"],
        misclassified["log_sr_y"],
        marker="o",
        s=90,
        facecolor="none",
        edgecolor=COLORS["ARC"],
        linewidth=1.5,
        label="Logistic ARC error",
        zorder=5,
    )
    ax.set_xlabel(r"log$_{10}$(Nb/Y)")
    ax.set_ylabel(r"log$_{10}$(Sr/Y)")
    ax.set_title(
        "b  Depleted MORB–arc ambiguity in ratio space", loc="left", fontweight="bold"
    )
    ax.set_xlim(-2.2, 1.0)
    ax.set_ylim(-0.4, 2.3)
    ax.grid(color="#EEEEEE", linewidth=0.6)
    ax.legend(
        loc="upper left",
        fontsize=7.4,
        frameon=True,
        framealpha=0.92,
        ncol=2,
    )

    fig.suptitle(
        "External whole-rock MORB validation and a geochemical failure mode",
        fontsize=12,
        fontweight="bold",
    )
    for suffix in ("png", "pdf"):
        fig.savefig(FIG / f"figshare_morb_external_validation.{suffix}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
