from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score

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
    "V(PPM)",
    "CR(PPM)",
    "NI(PPM)",
    "RB(PPM)",
    "SR(PPM)",
    "Y(PPM)",
    "ZR(PPM)",
    "NB(PPM)",
    "LA(PPM)",
    "CE(PPM)",
    "ND(PPM)",
    "TH(PPM)",
]
MODEL_NAMES = ["logistic_balanced", "random_forest_balanced"]


def log_features(frame: pd.DataFrame) -> pd.DataFrame:
    return np.log10(frame[FEATURES].where(frame[FEATURES] > 0))


def load_external() -> pd.DataFrame:
    pangaea = pd.read_parquet(
        PROCESSED / "pangaea_922011_arc_backarc_external_v0_1.parquet"
    )
    pangaea["external_source"] = "PANGAEA.922011"
    rgr = pd.read_parquet(PROCESSED / "rio_grande_rift_cib_external_v0_1.parquet")
    rgr["external_source"] = "Rio Grande Rift Table S1"
    return pd.concat([pangaea, rgr], ignore_index=True, sort=False)


def probability_matrix(model: object, X: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(X)
    classes = list(model.classes_)
    return np.column_stack([raw[:, classes.index(label)] for label in LABELS])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(PROCESSED / "petdb_primary4_v0_1.parquet")
    train = train.loc[train["model_ready_broad"]].reset_index(drop=True)
    external = load_external()
    external = external.loc[external["model_ready_common22"]].reset_index(drop=True)
    y = train["label_primary4"].astype(str)
    X = log_features(train)
    X_external = log_features(external)

    lower = X.quantile(0.01)
    upper = X.quantile(0.99)
    external["outside_train_1_99_n"] = (
        (X_external.lt(lower, axis="columns") | X_external.gt(upper, axis="columns"))
        & X_external.notna()
    ).sum(axis=1)
    external["ratio_nb_y"] = external["NB(PPM)"] / external["Y(PPM)"]
    external["ratio_sr_y"] = external["SR(PPM)"] / external["Y(PPM)"]
    external["ratio_th_nb"] = external["TH(PPM)"] / external["NB(PPM)"]

    prediction_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    templates = models()
    for model_name in MODEL_NAMES:
        model = clone(templates[model_name])
        model.fit(X, y)
        predicted = model.predict(X_external)
        probability = probability_matrix(model, X_external)
        frame = external[
            [
                "record_id",
                "external_source",
                "sample_name",
                "source_area",
                "mechanism_cohort",
                "actual_label",
                "strict_external_candidate",
                "observed_feature_n",
                "outside_train_1_99_n",
                "SIO2(WT%)",
                "MGO(WT%)",
                "K2O(WT%)",
                "NB(PPM)",
                "SR(PPM)",
                "Y(PPM)",
                "TH(PPM)",
                "ratio_nb_y",
                "ratio_sr_y",
                "ratio_th_nb",
            ]
        ].copy()
        frame["model"] = model_name
        frame["predicted"] = predicted
        for index, label in enumerate(LABELS):
            frame[f"prob_{label.lower()}"] = probability[:, index]
        prediction_frames.append(frame)

        cohorts = {
            "ARC_FRONT_STRICT": frame["mechanism_cohort"].eq("ARC_FRONT")
            & frame["strict_external_candidate"],
            "RGR_CIB_STRICT": frame["mechanism_cohort"].eq("CONTINENTAL_RIFT")
            & frame["strict_external_candidate"],
            "BACK_ARC_TRANSITION": frame["mechanism_cohort"].eq(
                "BACK_ARC_TRANSITION"
            ),
        }
        for cohort_name, mask in cohorts.items():
            cohort = frame.loc[mask]
            target = {
                "ARC_FRONT_STRICT": "ARC",
                "RGR_CIB_STRICT": "CIB",
                "BACK_ARC_TRANSITION": None,
            }[cohort_name]
            row: dict[str, object] = {
                "model": model_name,
                "cohort": cohort_name,
                "target": target or "UNSCORED_TRANSITION",
                "n": len(cohort),
                "target_recall": (
                    float(cohort["predicted"].eq(target).mean()) if target else np.nan
                ),
                "mean_target_probability": (
                    float(cohort[f"prob_{target.lower()}"].mean())
                    if target
                    else np.nan
                ),
                "median_target_probability": (
                    float(cohort[f"prob_{target.lower()}"].median())
                    if target
                    else np.nan
                ),
                "mean_outside_train_1_99_n": float(
                    cohort["outside_train_1_99_n"].mean()
                ),
                "share_with_any_outside_train_1_99": float(
                    cohort["outside_train_1_99_n"].gt(0).mean()
                ),
            }
            for label in LABELS:
                row[f"predicted_share_{label.lower()}"] = float(
                    cohort["predicted"].eq(label).mean()
                )
                row[f"mean_prob_{label.lower()}"] = float(
                    cohort[f"prob_{label.lower()}"].mean()
                )
            summary_rows.append(row)

        strict = frame.loc[frame["strict_external_candidate"]].copy()
        summary_rows.append(
            {
                "model": model_name,
                "cohort": "ARC_CIB_COMBINED_STRICT",
                "target": "TWO_CLASS_SUBSET_OF_FOUR_CLASS_MODEL",
                "n": len(strict),
                "target_recall": accuracy_score(
                    strict["actual_label"], strict["predicted"]
                ),
                "mean_target_probability": np.mean(
                    [
                        row[f"prob_{actual.lower()}"]
                        for (_, row), actual in zip(
                            strict.iterrows(), strict["actual_label"]
                        )
                    ]
                ),
                "median_target_probability": np.median(
                    [
                        row[f"prob_{actual.lower()}"]
                        for (_, row), actual in zip(
                            strict.iterrows(), strict["actual_label"]
                        )
                    ]
                ),
                "balanced_accuracy_two_observed_classes": balanced_accuracy_score(
                    strict["actual_label"], strict["predicted"]
                ),
                "mean_outside_train_1_99_n": float(
                    strict["outside_train_1_99_n"].mean()
                ),
                "share_with_any_outside_train_1_99": float(
                    strict["outside_train_1_99_n"].gt(0).mean()
                ),
            }
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(OUT / "arc_cib_external_predictions.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "arc_cib_external_summary.csv", index=False)

    internal = pd.read_csv(OUT / "petdb_primary4_all22_overall_metrics.csv")
    internal = internal.loc[
        internal["cv_strategy"].eq("citation_overlap_conservative")
        & internal["model"].isin(MODEL_NAMES)
    ]
    internal.to_csv(OUT / "petdb_common22_external_reference_metrics.csv", index=False)
    manifest = {
        "training_dataset": "data/processed/petdb_primary4_v0_1.parquet",
        "training_rows": len(train),
        "training_class_counts": y.value_counts().to_dict(),
        "features": FEATURES,
        "models": MODEL_NAMES,
        "external_sources": [
            "data/processed/pangaea_922011_arc_backarc_external_v0_1.parquet",
            "data/processed/rio_grande_rift_cib_external_v0_1.parquet",
        ],
        "strict_arc_n": int(
            external["mechanism_cohort"].eq("ARC_FRONT")
            .where(external["strict_external_candidate"], False)
            .sum()
        ),
        "strict_cib_n": int(
            external["mechanism_cohort"].eq("CONTINENTAL_RIFT")
            .where(external["strict_external_candidate"], False)
            .sum()
        ),
        "back_arc_transition_n": int(
            external["mechanism_cohort"].eq("BACK_ARC_TRANSITION").sum()
        ),
        "domain_check": (
            "Count each external sample's observed log10 features outside the PetDB "
            "training 1st-99th percentile envelope."
        ),
        "scoring_policy": (
            "Erromango and Vulcan Seamount are scored as ARC; Rio Grande Rift and "
            "Jemez Lineament basalt are scored as CIB. Vate Trough is an unscored "
            "back-arc transition pressure test."
        ),
        "scope_caveat": (
            "The strict external set contains only ARC and CIB and only two studies. "
            "Combined accuracy is not external four-class balanced accuracy, and samples "
            "within each study are not independent geological provinces."
        ),
    }
    (OUT / "arc_cib_external_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(internal.to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
