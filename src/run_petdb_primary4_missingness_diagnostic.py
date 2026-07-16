from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "petdb_primary4_v0_1.parquet"
CONFIG_PATH = ROOT / "config" / "study_config.yaml"
OUT_DIR = ROOT / "reports" / "modeling"
LABELS = ["ARC", "CIB", "MORB", "OIB"]
SEED = 20260714


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    broad = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "feature_sets"
    ]["broad_v1"]
    data = pd.read_parquet(DATA_PATH)
    data = data.loc[data["model_ready_broad"]].reset_index(drop=True)
    y = data["label_primary4"].astype(str)
    class_non_null = data.groupby("label_primary4")[broad].count()
    common = [feature for feature in broad if class_non_null[feature].min() > 0]
    structural = [feature for feature in broad if feature not in common]
    prevalence = (
        data.groupby("label_primary4")[broad]
        .agg(lambda values: values.isna().mean())
        .T.reset_index(names="feature")
    )
    prevalence.to_csv(
        OUT_DIR / "petdb_primary4_missingness_prevalence.csv", index=False
    )

    variants = {"all_features": broad, "common_features": common}
    groups_map = {
        "citation_overlap_conservative": data["citation_overlap_component"],
        "location_root_grouped": data["location_root"],
    }
    rows: list[dict[str, object]] = []
    coefficients: list[dict[str, object]] = []
    for variant, features in variants.items():
        X_missing = data[features].isna().astype(float)
        X_values = np.log10(data[features].where(data[features] > 0))
        for representation, X in {
            "missingness_only": X_missing,
            "values_no_indicator": X_values,
        }.items():
            for strategy, groups in groups_map.items():
                splitter = StratifiedGroupKFold(
                    n_splits=5, shuffle=True, random_state=SEED
                )
                for fold, (train, test) in enumerate(
                    splitter.split(X, y, groups), start=1
                ):
                    if representation == "missingness_only":
                        model = LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced",
                            random_state=SEED,
                        )
                    else:
                        model = Pipeline(
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
                        )
                    model.fit(X.iloc[train], y.iloc[train])
                    predicted = model.predict(X.iloc[test])
                    raw_probability = model.predict_proba(X.iloc[test])
                    classes = list(model.classes_)
                    probability = np.column_stack(
                        [raw_probability[:, classes.index(label)] for label in LABELS]
                    )
                    rows.append(
                        {
                            "variant": variant,
                            "representation": representation,
                            "cv_strategy": strategy,
                            "fold": fold,
                            "train_n": len(train),
                            "test_n": len(test),
                            "balanced_accuracy": balanced_accuracy_score(
                                y.iloc[test], predicted
                            ),
                            "macro_f1": f1_score(
                                y.iloc[test], predicted, average="macro"
                            ),
                            "log_loss": log_loss(
                                y.iloc[test], probability, labels=LABELS
                            ),
                        }
                    )
                    estimator = model if representation == "missingness_only" else model[-1]
                    for class_index, label in enumerate(estimator.classes_):
                        for feature, coefficient in zip(
                            features, estimator.coef_[class_index], strict=True
                        ):
                            coefficients.append(
                                {
                                    "variant": variant,
                                    "representation": representation,
                                    "cv_strategy": strategy,
                                    "fold": fold,
                                    "label": label,
                                    "feature": feature,
                                    "coefficient": coefficient,
                                }
                            )
    folds = pd.DataFrame(rows)
    folds.to_csv(
        OUT_DIR / "petdb_primary4_missingness_fold_metrics.csv", index=False
    )
    summary = (
        folds.groupby(["variant", "representation", "cv_strategy"])[
            ["balanced_accuracy", "macro_f1", "log_loss"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(part for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    summary.to_csv(
        OUT_DIR / "petdb_primary4_missingness_summary.csv", index=False
    )
    pd.DataFrame(coefficients).to_csv(
        OUT_DIR / "petdb_primary4_missingness_coefficients.csv", index=False
    )
    manifest = {
        "rows": len(data),
        "all_features": broad,
        "features_present_in_every_class": common,
        "structurally_absent_in_at_least_one_class": structural,
        "purpose": (
            "Quantify label signal carried by reporting missingness and compare it with "
            "a value-only logistic model without explicit missingness indicators."
        ),
    }
    (OUT_DIR / "petdb_primary4_missingness_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
