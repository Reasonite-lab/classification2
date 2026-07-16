from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
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
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "petdb_primary4_v0_1.parquet"
CONFIG_PATH = ROOT / "config" / "study_config.yaml"
OUT_DIR = ROOT / "reports" / "modeling"
SEED = 20260714
LABELS = ["ARC", "CIB", "MORB", "OIB"]


def models() -> dict[str, Pipeline]:
    return {
        "dummy_prior": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", DummyClassifier(strategy="prior", random_state=SEED)),
            ]
        ),
        "logistic_balanced": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=1,
                        solver="lbfgs",
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "random_forest_balanced": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=160,
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


def strategies(data: pd.DataFrame) -> dict[str, tuple[object, pd.Series | None]]:
    return {
        "random_stratified": (
            StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
            None,
        ),
        "citation_set_grouped": (
            StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED),
            data["citation_set"],
        ),
        "citation_overlap_conservative": (
            StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED),
            data["citation_overlap_component"],
        ),
        "location_root_grouped": (
            StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED),
            data["location_root"],
        ),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    configured_features = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "feature_sets"
    ]["broad_v1"]
    data = pd.read_parquet(DATA_PATH)
    data = data.loc[data["model_ready_broad"]].reset_index(drop=True)
    y = data["label_primary4"].astype(str)
    if sorted(y.unique()) != LABELS:
        raise ValueError(f"Unexpected labels: {sorted(y.unique())}")
    class_non_null = data.groupby("label_primary4")[configured_features].count()
    features = [
        feature
        for feature in configured_features
        if class_non_null[feature].min() > 0
    ]
    excluded_structural_features = [
        feature for feature in configured_features if feature not in features
    ]
    X = np.log10(data[features].where(data[features] > 0))

    fold_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    model_templates = models()
    strategy_map = strategies(data)
    for strategy_name, (splitter, groups) in strategy_map.items():
        split_iter = (
            splitter.split(X, y, groups)
            if groups is not None
            else splitter.split(X, y)
        )
        for fold, (train_index, test_index) in enumerate(split_iter, start=1):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            for model_name, template in model_templates.items():
                model = clone(template)
                start = time.perf_counter()
                model.fit(X_train, y_train)
                fit_seconds = time.perf_counter() - start
                predicted = model.predict(X_test)
                raw_probability = model.predict_proba(X_test)
                classes = list(model.classes_)
                probability = np.column_stack(
                    [raw_probability[:, classes.index(label)] for label in LABELS]
                )
                fold_rows.append(
                    {
                        "cv_strategy": strategy_name,
                        "fold": fold,
                        "model": model_name,
                        "train_n": len(train_index),
                        "test_n": len(test_index),
                        "fit_seconds": round(fit_seconds, 4),
                        "accuracy": accuracy_score(y_test, predicted),
                        "balanced_accuracy": balanced_accuracy_score(
                            y_test, predicted
                        ),
                        "macro_f1": f1_score(y_test, predicted, average="macro"),
                        "weighted_f1": f1_score(
                            y_test, predicted, average="weighted"
                        ),
                        "log_loss": log_loss(y_test, probability, labels=LABELS),
                    }
                )
                prediction_frames.append(
                    pd.DataFrame(
                        {
                            "record_id": data.iloc[test_index][
                                "record_id"
                            ].to_numpy(),
                            "cv_strategy": strategy_name,
                            "fold": fold,
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

    folds = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    folds.to_csv(OUT_DIR / "petdb_primary4_fold_metrics.csv", index=False)
    predictions.to_parquet(
        OUT_DIR / "petdb_primary4_oof_predictions.parquet", index=False
    )

    overall_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    probability_columns = [f"prob_{label.lower()}" for label in LABELS]
    for (strategy_name, model_name), frame in predictions.groupby(
        ["cv_strategy", "model"]
    ):
        y_true, y_pred = frame["actual"], frame["predicted"]
        probability = frame[probability_columns].to_numpy()
        overall_rows.append(
            {
                "cv_strategy": strategy_name,
                "model": model_name,
                "n": len(frame),
                "accuracy": accuracy_score(y_true, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                "macro_f1": f1_score(y_true, y_pred, average="macro"),
                "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
                "log_loss": log_loss(y_true, probability, labels=LABELS),
            }
        )
        report = classification_report(
            y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
        )
        for label in LABELS:
            class_rows.append(
                {
                    "cv_strategy": strategy_name,
                    "model": model_name,
                    "label": label,
                    **{
                        key: report[label][key]
                        for key in ("precision", "recall", "f1-score", "support")
                    },
                }
            )
    overall = pd.DataFrame(overall_rows).sort_values(
        ["cv_strategy", "macro_f1"], ascending=[True, False]
    )
    per_class = pd.DataFrame(class_rows)
    overall.to_csv(OUT_DIR / "petdb_primary4_overall_metrics.csv", index=False)
    per_class.to_csv(OUT_DIR / "petdb_primary4_per_class_metrics.csv", index=False)

    manifest = {
        "dataset": str(DATA_PATH.relative_to(ROOT)),
        "rows": len(data),
        "class_counts": y.value_counts().to_dict(),
        "configured_features": configured_features,
        "features": features,
        "excluded_structurally_absent_features": excluded_structural_features,
        "feature_variant": "broad features observed at least once in every PetDB class",
        "labels": LABELS,
        "random_seed": SEED,
        "models": list(model_templates),
        "cv_strategies": list(strategy_map),
        "interpretation": (
            "PetDB now contains all four labels, so class is no longer deterministically "
            "mapped to database source. Features absent from an entire class export are "
            "excluded from this primary validation. Citation- and location-grouped estimates "
            "remain the appropriate primary evidence; random-row scores are optimistic "
            "sensitivity checks."
        ),
    }
    (OUT_DIR / "petdb_primary4_validation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
