from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
GEOROC_DATA = ROOT / "data" / "processed" / "georoc_primary3_v0_1.parquet"
PETDB_DATA = ROOT / "data" / "processed" / "petdb_morb_v0_1.parquet"
CONFIG = ROOT / "config" / "study_config.yaml"
OUT_DIR = ROOT / "reports" / "modeling"
RANDOM_SEED = 20260714


def make_pipeline(input_spec: str) -> Pipeline:
    if input_spec == "missingness_only":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        )
    if input_spec == "values_only":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        )
    if input_spec == "values_plus_missing":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        )
    raise ValueError(input_spec)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    features = config["feature_sets"]["broad_v1"]

    georoc = pd.read_parquet(GEOROC_DATA)
    georoc = georoc.loc[georoc["model_ready_broad"]].copy()
    # Use the same total-iron-as-FeO-equivalent convention as the PetDB cohort.
    # The published broad_v1 baseline used reported FEOT; retaining that here
    # would turn a harmonization choice into a trivial source indicator.
    georoc["FEOT(WT%)"] = georoc["FEO_EQ(WT%)"]
    georoc["source"] = "GEOROC"
    georoc["source_target"] = 0
    georoc["source_group"] = (
        "GEOROC:" + georoc["citation_overlap_component"].astype(str)
    )

    petdb = pd.read_parquet(PETDB_DATA)
    petdb = petdb.loc[petdb["model_ready_broad"]].copy()
    petdb["source"] = "PETDB"
    petdb["source_target"] = 1
    petdb["source_group"] = (
        "PETDB:" + petdb["citation_overlap_component"].astype(str)
    )

    common_columns = features + ["source", "source_target", "source_group", "record_id"]
    combined = pd.concat(
        [georoc[common_columns], petdb[common_columns]], ignore_index=True
    )
    values = combined[features].astype(float).where(combined[features].astype(float) > 0)
    log_values = np.log10(values)
    missingness = values.isna().astype(float)
    target = combined["source_target"].astype(int)
    groups = combined["source_group"].astype(str)

    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_SEED
    )
    split_indices = list(splitter.split(log_values, target, groups))
    fold_rows: list[dict[str, object]] = []
    input_matrices = {
        "missingness_only": missingness,
        "values_only": log_values,
        "values_plus_missing": log_values,
    }
    for input_spec, matrix in input_matrices.items():
        for fold, (train_index, test_index) in enumerate(split_indices, start=1):
            pipeline = make_pipeline(input_spec)
            pipeline.fit(matrix.iloc[train_index], target.iloc[train_index])
            predicted = pipeline.predict(matrix.iloc[test_index])
            probability = pipeline.predict_proba(matrix.iloc[test_index])[:, 1]
            actual = target.iloc[test_index]
            fold_rows.append(
                {
                    "input_spec": input_spec,
                    "fold": fold,
                    "train_n": len(train_index),
                    "test_n": len(test_index),
                    "test_georoc_n": int((actual == 0).sum()),
                    "test_petdb_n": int((actual == 1).sum()),
                    "accuracy": accuracy_score(actual, predicted),
                    "balanced_accuracy": balanced_accuracy_score(actual, predicted),
                    "macro_f1": f1_score(actual, predicted, average="macro"),
                    "georoc_recall": recall_score(actual, predicted, pos_label=0),
                    "petdb_recall": recall_score(actual, predicted, pos_label=1),
                    "roc_auc": roc_auc_score(actual, probability),
                }
            )
    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(
        OUT_DIR / "petdb_source_confounding_fold_metrics.csv", index=False
    )
    summary = (
        fold_metrics.groupby("input_spec", as_index=False)
        .agg(
            n_folds=("fold", "nunique"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_sd=("macro_f1", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_sd=("balanced_accuracy", "std"),
            georoc_recall_mean=("georoc_recall", "mean"),
            petdb_recall_mean=("petdb_recall", "mean"),
            roc_auc_mean=("roc_auc", "mean"),
        )
        .sort_values("input_spec")
    )
    summary.to_csv(OUT_DIR / "petdb_source_confounding_summary.csv", index=False)

    prevalence = []
    for source, source_frame in combined.groupby("source"):
        source_values = source_frame[features].where(source_frame[features] > 0)
        for feature in features:
            prevalence.append(
                {
                    "source": source,
                    "feature": feature,
                    "n": len(source_frame),
                    "missing_pct": 100 * source_values[feature].isna().mean(),
                }
            )
    prevalence = pd.DataFrame(prevalence)
    prevalence_wide = prevalence.pivot(
        index="feature", columns="source", values="missing_pct"
    ).reset_index()
    prevalence_wide["petdb_minus_georoc_missing_pct"] = (
        prevalence_wide["PETDB"] - prevalence_wide["GEOROC"]
    )
    prevalence_wide.sort_values(
        "petdb_minus_georoc_missing_pct", key=lambda x: x.abs(), ascending=False
    ).to_csv(OUT_DIR / "petdb_source_missingness_prevalence.csv", index=False)

    missingness_model = make_pipeline("missingness_only")
    missingness_model.fit(missingness, target)
    coefficients = pd.DataFrame(
        {
            "feature": features,
            "coefficient_toward_petdb_when_missing": missingness_model.named_steps[
                "model"
            ].coef_[0],
        }
    ).sort_values(
        "coefficient_toward_petdb_when_missing", key=lambda x: x.abs(), ascending=False
    )
    coefficients.to_csv(
        OUT_DIR / "petdb_source_missingness_coefficients.csv", index=False
    )

    manifest = {
        "purpose": "Diagnose database-source separability before any pooled four-class model",
        "georoc_rows": len(georoc),
        "petdb_rows": len(petdb),
        "features": features,
        "validation": "5-fold StratifiedGroupKFold using source-prefixed citation-overlap components",
        "model": "class-balanced logistic regression",
        "value_transform": "log10 positive values; fold-local median imputation",
        "iron_harmonization": "GEOROC FEO_EQ and PetDB harmonized FEOT, both total iron as FeO equivalent",
        "interpretation": (
            "Missingness-only predictability is direct evidence that measurement coverage "
            "contains database/source signal. Value-based predictability cannot separate "
            "geologic MORB-vs-non-MORB differences from database effects because class and "
            "source are perfectly confounded in the current pooled design."
        ),
        "claim_gate": (
            "Do not report a naive GEOROC ARC/CIB/OIB plus PetDB MORB four-class score as "
            "source-robust performance. Acquire overlapping tectonic classes across sources."
        ),
    }
    (OUT_DIR / "petdb_source_confounding_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
