from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from audit_pangaea_morb_candidates import FEATURE_ALIASES, read_pangaea, resolve_column
from run_petdb_primary4_validation import LABELS, SEED, models


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "modeling"
PROCESSED = ROOT / "data" / "processed"
MAJOR_FEATURES = [
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
]
ACCEPTED = {
    "PANGAEA.727391": ROOT / "data" / "raw" / "pangaea_727391.tsv",
    "PANGAEA.707361": ROOT / "data" / "raw" / "pangaea_707361.tsv",
}


def build_external_samples() -> tuple[pd.DataFrame, dict[str, object]]:
    frames: list[pd.DataFrame] = []
    profile: dict[str, object] = {}
    for dataset_id, path in ACCEPTED.items():
        raw, _ = read_pangaea(path)
        name_column = "Event" if "Event" in raw and raw["Event"].notna().sum() else "Sample label"
        valid = raw[name_column].notna()
        feature_frame = pd.DataFrame(index=raw.index)
        for feature in MAJOR_FEATURES:
            column = resolve_column(raw.columns, FEATURE_ALIASES[feature])
            feature_frame[feature] = (
                pd.to_numeric(raw[column], errors="coerce")
                if column is not None
                else pd.Series(index=raw.index, dtype=float)
            )
        feature_frame["sample_name"] = raw[name_column]
        sample = feature_frame.loc[valid].groupby("sample_name", as_index=False).median(numeric_only=True)
        sample["source_dataset"] = dataset_id
        sample["major_observed_n"] = sample[MAJOR_FEATURES].notna().sum(axis=1)
        sample["major_total_calc"] = sample[MAJOR_FEATURES].sum(axis=1, min_count=len(MAJOR_FEATURES))
        sample["flag_complete_major10"] = sample["major_observed_n"].eq(len(MAJOR_FEATURES))
        sample["flag_silica_40_55"] = sample["SIO2(WT%)"].between(40, 55)
        sample["flag_total_85_105"] = sample["major_total_calc"].between(85, 105)
        sample["model_ready_external"] = (
            sample["flag_complete_major10"]
            & sample["flag_silica_40_55"]
            & sample["flag_total_85_105"]
        )
        profile[dataset_id] = {
            "aggregated_samples": len(sample),
            "complete_major10": int(sample["flag_complete_major10"].sum()),
            "silica_40_55": int(sample["flag_silica_40_55"].sum()),
            "total_85_105": int(sample["flag_total_85_105"].sum()),
            "model_ready_external": int(sample["model_ready_external"].sum()),
        }
        frames.append(sample)
    external = pd.concat(frames, ignore_index=True)
    external["label_primary4"] = "MORB"
    external["record_id"] = (
        "pangaea:" + external["source_dataset"] + ":" + external["sample_name"].astype(str)
    )
    return external, profile


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    external, external_profile = build_external_samples()
    external.to_parquet(PROCESSED / "pangaea_morb_external_v0_1.parquet", index=False)
    test = external.loc[external["model_ready_external"]].reset_index(drop=True)
    if len(test) < 10:
        raise ValueError(f"External cohort unexpectedly small: {len(test)}")

    train = pd.read_parquet(PROCESSED / "petdb_primary4_v0_1.parquet")
    train = train.loc[train["model_ready_broad"]].reset_index(drop=True)
    y = train["label_primary4"].astype(str)
    X = np.log10(train[MAJOR_FEATURES].where(train[MAJOR_FEATURES] > 0))
    X_external = np.log10(test[MAJOR_FEATURES].where(test[MAJOR_FEATURES] > 0))

    templates = models()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    internal_rows: list[dict[str, object]] = []
    for model_name in ["logistic_balanced", "random_forest_balanced"]:
        predicted = pd.Series(index=train.index, dtype=object)
        for train_index, test_index in splitter.split(X, y, train["citation_overlap_component"]):
            model = clone(templates[model_name])
            model.fit(X.iloc[train_index], y.iloc[train_index])
            predicted.iloc[test_index] = model.predict(X.iloc[test_index])
        internal_rows.append(
            {
                "model": model_name,
                "feature_set": "major10",
                "validation": "citation_overlap_conservative",
                "n": len(train),
                "accuracy": accuracy_score(y, predicted),
                "balanced_accuracy": balanced_accuracy_score(y, predicted),
                "macro_f1": f1_score(y, predicted, average="macro"),
            }
        )
    pd.DataFrame(internal_rows).to_csv(OUT / "petdb_major10_internal_metrics.csv", index=False)

    prediction_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for model_name in ["logistic_balanced", "random_forest_balanced"]:
        model = clone(templates[model_name])
        model.fit(X, y)
        predicted = model.predict(X_external)
        raw_probability = model.predict_proba(X_external)
        classes = list(model.classes_)
        probability = np.column_stack(
            [raw_probability[:, classes.index(label)] for label in LABELS]
        )
        frame = test[["record_id", "source_dataset", "sample_name"]].copy()
        frame["model"] = model_name
        frame["actual"] = "MORB"
        frame["predicted"] = predicted
        for index, label in enumerate(LABELS):
            frame[f"prob_{label.lower()}"] = probability[:, index]
        prediction_frames.append(frame)
        for source, source_frame in frame.groupby("source_dataset"):
            summary_rows.append(
                {
                    "model": model_name,
                    "source_dataset": source,
                    "n": len(source_frame),
                    "morb_recall": source_frame["predicted"].eq("MORB").mean(),
                    "mean_prob_morb": source_frame["prob_morb"].mean(),
                    "median_prob_morb": source_frame["prob_morb"].median(),
                }
            )
        summary_rows.append(
            {
                "model": model_name,
                "source_dataset": "ALL_ACCEPTED",
                "n": len(frame),
                "morb_recall": frame["predicted"].eq("MORB").mean(),
                "mean_prob_morb": frame["prob_morb"].mean(),
                "median_prob_morb": frame["prob_morb"].median(),
            }
        )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(OUT / "pangaea_morb_external_predictions.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(OUT / "pangaea_morb_external_summary.csv", index=False)
    manifest = {
        "training_dataset": "data/processed/petdb_primary4_v0_1.parquet",
        "training_rows": len(train),
        "training_class_counts": y.value_counts().to_dict(),
        "external_dataset": "data/processed/pangaea_morb_external_v0_1.parquet",
        "external_profile": external_profile,
        "external_test_rows": len(test),
        "features": MAJOR_FEATURES,
        "accepted_sources": list(ACCEPTED),
        "excluded_source": {
            "PANGAEA.956077": "Publication citation is present in the PetDB MORB export; excluded to avoid source overlap."
        },
        "scope_caveat": (
            "Pilot one-class external MORB stress test on glass/glass-inclusion data. "
            "It estimates MORB recall only and is not a four-class external macro-performance estimate."
        ),
    }
    (OUT / "pangaea_morb_external_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
