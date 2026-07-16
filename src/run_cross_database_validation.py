from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
GEOROC_PATH = ROOT / "data" / "processed" / "georoc_primary3_v0_1.parquet"
PETDB_PATH = ROOT / "data" / "processed" / "petdb_primary4_v0_1.parquet"
CONFIG_PATH = ROOT / "config" / "study_config.yaml"
OUT_DIR = ROOT / "reports" / "modeling"
LABELS = ["ARC", "CIB", "OIB"]
SEED = 20260714


def make_models() -> dict[str, Pipeline]:
    return {
        "logistic_balanced": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "random_forest_balanced": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        min_samples_leaf=4,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
    }


def evaluate_direction(
    ontology_variant: str,
    direction: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[pd.DataFrame]]:
    X_train = np.log10(train[features].where(train[features] > 0))
    X_test = np.log10(test[features].where(test[features] > 0))
    y_train = train["label"].astype(str)
    y_test = test["label"].astype(str)
    overall_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    for model_name, model in make_models().items():
        start = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - start
        predicted = model.predict(X_test)
        raw_probability = model.predict_proba(X_test)
        classes = list(model.classes_)
        probability = np.column_stack(
            [raw_probability[:, classes.index(label)] for label in LABELS]
        )
        overall_rows.append(
            {
                "direction": direction,
                "ontology_variant": ontology_variant,
                "model": model_name,
                "train_n": len(train),
                "test_n": len(test),
                "fit_seconds": round(elapsed, 4),
                "accuracy": accuracy_score(y_test, predicted),
                "balanced_accuracy": balanced_accuracy_score(y_test, predicted),
                "macro_f1": f1_score(y_test, predicted, average="macro"),
                "weighted_f1": f1_score(y_test, predicted, average="weighted"),
                "log_loss": log_loss(y_test, probability, labels=LABELS),
            }
        )
        report = classification_report(
            y_test, predicted, labels=LABELS, output_dict=True, zero_division=0
        )
        for label in LABELS:
            class_rows.append(
                {
                    "direction": direction,
                    "ontology_variant": ontology_variant,
                    "model": model_name,
                    "label": label,
                    **{
                        key: report[label][key]
                        for key in ("precision", "recall", "f1-score", "support")
                    },
                }
            )
        predictions.append(
            pd.DataFrame(
                {
                    "record_id": test["record_id"].to_numpy(),
                    "direction": direction,
                    "ontology_variant": ontology_variant,
                    "model": model_name,
                    "actual": y_test.to_numpy(),
                    "predicted": predicted,
                    **{
                        f"prob_{label.lower()}": probability[:, index]
                        for index, label in enumerate(LABELS)
                    },
                }
            )
        )
    return overall_rows, class_rows, predictions


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    configured_features = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "feature_sets"
    ]["broad_v1"]
    georoc = pd.read_parquet(GEOROC_PATH)
    georoc = georoc.loc[georoc["model_ready_broad"]].copy()
    georoc["label"] = georoc["label_primary3"].astype(str)
    petdb = pd.read_parquet(PETDB_PATH)
    petdb = petdb.loc[
        petdb["model_ready_broad"] & petdb["label_primary4"].isin(LABELS)
    ].copy()
    petdb["label"] = petdb["label_primary4"].astype(str)
    if sorted(georoc["label"].unique()) != LABELS:
        raise ValueError("Unexpected GEOROC labels")
    if sorted(petdb["label"].unique()) != LABELS:
        raise ValueError("Unexpected PetDB labels")

    combined = pd.concat(
        [
            georoc.assign(database="GEOROC"),
            petdb.assign(database="PetDB"),
        ],
        ignore_index=True,
        sort=False,
    )
    non_null = combined.groupby(["database", "label"])[configured_features].count()
    features = [feature for feature in configured_features if non_null[feature].min() > 0]
    excluded = [feature for feature in configured_features if feature not in features]

    ontology_variants = {
        "broad_cib": georoc,
        "rift_aligned_cib": georoc.loc[
            georoc["label"].ne("CIB")
            | georoc["label_fine"].eq("RIFT_VOLCANICS")
        ].copy(),
    }
    overall_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for ontology_variant, georoc_variant in ontology_variants.items():
        for direction, train, test in [
            ("GEOROC_to_PetDB", georoc_variant, petdb),
            ("PetDB_to_GEOROC", petdb, georoc_variant),
        ]:
            overall, by_class, predictions = evaluate_direction(
                ontology_variant, direction, train, test, features
            )
            overall_rows.extend(overall)
            class_rows.extend(by_class)
            prediction_frames.extend(predictions)

    overall = pd.DataFrame(overall_rows)
    per_class = pd.DataFrame(class_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    overall.to_csv(OUT_DIR / "cross_database_overall_metrics.csv", index=False)
    per_class.to_csv(OUT_DIR / "cross_database_per_class_metrics.csv", index=False)
    predictions.to_parquet(
        OUT_DIR / "cross_database_predictions.parquet", index=False
    )
    manifest = {
        "georoc_rows": len(georoc),
        "petdb_rows": len(petdb),
        "georoc_class_counts": georoc["label"].value_counts().to_dict(),
        "petdb_class_counts": petdb["label"].value_counts().to_dict(),
        "georoc_ontology_variants": {
            name: {
                "rows": len(frame),
                "class_counts": frame["label"].value_counts().to_dict(),
            }
            for name, frame in ontology_variants.items()
        },
        "configured_features": configured_features,
        "shared_features": features,
        "excluded_structurally_absent_features": excluded,
        "label_ontology_caveat": (
            "GEOROC CIB combines intraplate volcanics, continental flood basalt, and "
            "rift volcanics, whereas the PetDB CIB export is restricted to CONTINENTAL RIFT. "
            "The two transfer directions are database/ontology-shift diagnostics, not final "
            "external performance claims."
        ),
    }
    (OUT_DIR / "cross_database_validation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(overall.to_string(index=False))
    print("\nPer-class recall")
    print(
        per_class.pivot_table(
            index=["ontology_variant", "direction", "model"],
            columns="label",
            values="recall",
        ).to_string()
    )


if __name__ == "__main__":
    main()
