from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from run_petdb_primary4_validation import LABELS, models


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "reports" / "modeling"
FEATURES = [
    "SIO2(WT%)",
    "TIO2(WT%)",
    "AL2O3(WT%)",
    "FEOT(WT%)",
    "CAO(WT%)",
    "MGO(WT%)",
    "MNO(WT%)",
    "K2O(WT%)",
    "NA2O(WT%)",
    "P2O5(WT%)",
    "RB(PPM)",
    "SR(PPM)",
    "Y(PPM)",
    "ZR(PPM)",
    "NB(PPM)",
    "LA(PPM)",
    "CE(PPM)",
    "ND(PPM)",
]
MODEL_NAMES = ["logistic_balanced", "random_forest_balanced"]


def log_features(frame: pd.DataFrame) -> pd.DataFrame:
    return np.log10(frame[FEATURES].where(frame[FEATURES] > 0))


def probability_matrix(model: object, X: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(X)
    classes = list(model.classes_)
    return np.column_stack([raw[:, classes.index(label)] for label in LABELS])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(PROCESSED / "petdb_primary4_v0_1.parquet")
    train = train.loc[train["model_ready_broad"]].reset_index(drop=True)
    external = pd.read_parquet(
        PROCESSED / "la_palma_2021_oib_external_v0_1.parquet"
    )
    external = external.loc[external["strict_external_candidate"]].reset_index(
        drop=True
    )
    external["eruption_stage"] = np.select(
        [external["eruption_day"].le(20), external["eruption_day"].gt(20)],
        ["day_1_20_fractionated_tephrite", "post_day_20_primitive_basanite"],
        default="day_unreported",
    )

    y = train["label_primary4"].astype(str)
    X_train = log_features(train)
    X_external = log_features(external)
    lower = X_train.quantile(0.01)
    upper = X_train.quantile(0.99)
    external["outside_train_1_99_n"] = (
        (X_external.lt(lower, axis="columns") | X_external.gt(upper, axis="columns"))
        & X_external.notna()
    ).sum(axis=1)

    predictions: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    templates = models()
    for model_name in MODEL_NAMES:
        model = clone(templates[model_name])
        model.fit(X_train, y)
        predicted = model.predict(X_external)
        probability = probability_matrix(model, X_external)
        frame = external[
            [
                "record_id",
                "sample_name",
                "source_area",
                "actual_label",
                "eruption_day",
                "eruption_stage",
                "outside_train_1_99_n",
                "SIO2(WT%)",
                "MGO(WT%)",
                "NB(PPM)",
                "Y(PPM)",
                "TH(PPM)",
            ]
        ].copy()
        frame["model"] = model_name
        frame["predicted"] = predicted
        for index, label in enumerate(LABELS):
            frame[f"prob_{label.lower()}"] = probability[:, index]
        predictions.append(frame)

        for cohort, mask in {
            "ALL_LA_PALMA_2021": np.ones(len(frame), dtype=bool),
            "DAY_1_20_FRACTIONATED_TEPHRITE": frame["eruption_stage"].eq(
                "day_1_20_fractionated_tephrite"
            ),
            "POST_DAY_20_PRIMITIVE_BASANITE": frame["eruption_stage"].eq(
                "post_day_20_primitive_basanite"
            ),
        }.items():
            selected = frame.loc[mask]
            row: dict[str, object] = {
                "model": model_name,
                "cohort": cohort,
                "n": len(selected),
                "oib_recall": float(selected["predicted"].eq("OIB").mean()),
                "mean_prob_oib": float(selected["prob_oib"].mean()),
                "median_prob_oib": float(selected["prob_oib"].median()),
                "mean_outside_train_1_99_n": float(
                    selected["outside_train_1_99_n"].mean()
                ),
                "share_with_any_outside_train_1_99": float(
                    selected["outside_train_1_99_n"].gt(0).mean()
                ),
            }
            for label in LABELS:
                row[f"predicted_share_{label.lower()}"] = float(
                    selected["predicted"].eq(label).mean()
                )
                row[f"mean_prob_{label.lower()}"] = float(
                    selected[f"prob_{label.lower()}"].mean()
                )
            summaries.append(row)

    prediction_table = pd.concat(predictions, ignore_index=True)
    prediction_table.to_csv(OUT / "la_palma_oib_external_predictions.csv", index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "la_palma_oib_external_summary.csv", index=False)
    manifest = {
        "training_dataset": "data/processed/petdb_primary4_v0_1.parquet",
        "external_dataset": "data/processed/la_palma_2021_oib_external_v0_1.parquet",
        "publication_doi": "10.1016/j.epsl.2022.117793",
        "features": FEATURES,
        "models": MODEL_NAMES,
        "external_n": len(external),
        "stage_policy": (
            "Day 1-20 and post-day-20 groups follow the publication's interpreted "
            "transition from fractionated tephrite to more primitive basanite."
        ),
        "scope_caveat": (
            "Samples are a temporally resolved single-eruption cohort. Stage contrasts "
            "are mechanism diagnostics, not independent province replication."
        ),
    }
    (OUT / "la_palma_oib_external_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
