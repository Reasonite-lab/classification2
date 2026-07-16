from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold

from run_petdb_primary4_validation import LABELS, SEED, models


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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(PROCESSED / "petdb_primary4_v0_1.parquet")
    train = train.loc[train["model_ready_broad"]].reset_index(drop=True)
    external = pd.read_parquet(PROCESSED / "figshare_morb_external_v0_1.parquet")
    external = external.loc[external["model_ready_common18"]].reset_index(drop=True)
    y = train["label_primary4"].astype(str)
    X = log_features(train)
    X_external = log_features(external)

    lower = X.quantile(0.01)
    upper = X.quantile(0.99)
    external["outside_train_1_99_n"] = (
        (X_external.lt(lower, axis="columns") | X_external.gt(upper, axis="columns"))
        & X_external.notna()
    ).sum(axis=1)

    templates = models()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    internal_rows: list[dict[str, object]] = []
    for model_name in MODEL_NAMES:
        predicted = pd.Series(index=train.index, dtype=object)
        probabilities = np.full((len(train), len(LABELS)), np.nan)
        for train_index, test_index in splitter.split(
            X, y, train["citation_overlap_component"]
        ):
            model = clone(templates[model_name])
            model.fit(X.iloc[train_index], y.iloc[train_index])
            predicted.iloc[test_index] = model.predict(X.iloc[test_index])
            raw_probability = model.predict_proba(X.iloc[test_index])
            classes = list(model.classes_)
            probabilities[test_index] = np.column_stack(
                [raw_probability[:, classes.index(label)] for label in LABELS]
            )
        internal_rows.append(
            {
                "model": model_name,
                "feature_set": "common18_figshare",
                "validation": "citation_overlap_conservative",
                "n": len(train),
                "accuracy": accuracy_score(y, predicted),
                "balanced_accuracy": balanced_accuracy_score(y, predicted),
                "macro_f1": f1_score(y, predicted, average="macro"),
                "log_loss": log_loss(y, probabilities, labels=LABELS),
            }
        )
    pd.DataFrame(internal_rows).to_csv(
        OUT / "petdb_common18_figshare_internal_metrics.csv", index=False
    )

    prediction_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for model_name in MODEL_NAMES:
        model = clone(templates[model_name])
        model.fit(X, y)
        predicted = model.predict(X_external)
        raw_probability = model.predict_proba(X_external)
        classes = list(model.classes_)
        probability = np.column_stack(
            [raw_probability[:, classes.index(label)] for label in LABELS]
        )
        frame = external[
            [
                "source_region",
                "source_excel_row",
                "Sample",
                "provenance_tier",
                "strict_external_candidate",
                "observed_feature_n",
                "outside_train_1_99_n",
            ]
        ].copy()
        frame["record_id"] = (
            "figshare:25295671:" + frame["Sample"].astype(str)
        )
        frame["model"] = model_name
        frame["actual"] = "MORB"
        frame["predicted"] = predicted
        for index, label in enumerate(LABELS):
            frame[f"prob_{label.lower()}"] = probability[:, index]
        prediction_frames.append(frame)

        cohorts = {
            "SWIR_STRICT_PROVISIONAL": frame["strict_external_candidate"],
            "EPR_SECONDARY_SENSITIVITY": frame["source_region"].eq(
                "East Pacific Rise"
            ),
            "ALL_REPOSITORY_SENSITIVITY": pd.Series(True, index=frame.index),
        }
        for cohort_name, mask in cohorts.items():
            cohort = frame.loc[mask]
            summary_rows.append(
                {
                    "model": model_name,
                    "cohort": cohort_name,
                    "n": len(cohort),
                    "morb_recall": cohort["predicted"].eq("MORB").mean(),
                    "mean_prob_morb": cohort["prob_morb"].mean(),
                    "median_prob_morb": cohort["prob_morb"].median(),
                    "mean_outside_train_1_99_n": cohort[
                        "outside_train_1_99_n"
                    ].mean(),
                    "share_with_any_outside_train_1_99": cohort[
                        "outside_train_1_99_n"
                    ].gt(0).mean(),
                }
            )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(OUT / "figshare_morb_external_predictions.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "figshare_morb_external_summary.csv", index=False)
    manifest = {
        "training_dataset": "data/processed/petdb_primary4_v0_1.parquet",
        "training_rows": len(train),
        "training_class_counts": y.value_counts().to_dict(),
        "external_dataset": "data/processed/figshare_morb_external_v0_1.parquet",
        "external_rows": len(external),
        "strict_swir_rows": int(external["strict_external_candidate"].sum()),
        "features": FEATURES,
        "models": MODEL_NAMES,
        "domain_check": (
            "For each external sample, count observed feature values below the PetDB "
            "training 1st percentile or above the 99th percentile in log10 space."
        ),
        "provenance_policy": {
            "East Pacific Rise": (
                "Sensitivity only: workbook says these data are from Li et al. (2019)."
            ),
            "Southwest Indian Ridge": (
                "Strict provisional cohort: no exact normalized PetDB sample-name overlap, "
                "but the repository record has no linked journal DOI."
            ),
        },
        "scope_caveat": (
            "This is a one-class, one-study MORB province test. It estimates MORB recall "
            "and confidence, not external four-class macro-performance. The 36 SWIR rows "
            "are not 36 independent geological provinces."
        ),
    }
    (OUT / "figshare_morb_external_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(internal_rows).to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
