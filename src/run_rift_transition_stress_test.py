from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "petdb_primary4_v0_1.parquet"
CONFIG_PATH = ROOT / "config" / "study_config.yaml"
MODEL_MANIFEST = ROOT / "reports" / "modeling" / "petdb_primary4_validation_manifest.json"
OUT_DIR = ROOT / "reports" / "modeling"
SEED = 20260715
LABELS = ["ARC", "CIB", "MORB", "OIB"]


def extract_country(value: object) -> str:
    matches = re.findall(r"COUNTRY\s*\|\s*([^,\"|]+)", str(value), flags=re.IGNORECASE)
    normalized = sorted({re.sub(r"\s+", " ", item).strip().upper() for item in matches})
    return " | ".join(normalized) if normalized else "<UNRESOLVED>"


def model_templates() -> dict[str, Pipeline]:
    return {
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
                        n_estimators=500,
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


def citation_cluster_bootstrap(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    repeats: int = 2000,
) -> dict[str, tuple[float, float, float]]:
    working = frame.assign(
        citation_cluster=frame["citation_set"].fillna("<MISSING>"),
        oib_gt_cib=frame["prob_oib"].gt(frame["prob_cib"]).astype(float),
    )
    cluster = working.groupby("citation_cluster", observed=True).agg(
        n=("record_id", "size"),
        sum_prob_cib=("prob_cib", "sum"),
        sum_prob_oib=("prob_oib", "sum"),
        sum_transition_index=("transition_index", "sum"),
        sum_oib_gt_cib=("oib_gt_cib", "sum"),
    )
    values = cluster.to_numpy(dtype=float)
    draws = rng.integers(0, len(cluster), size=(repeats, len(cluster)))
    sampled_sums = values[draws].sum(axis=1)
    sampled_n = sampled_sums[:, 0]
    metrics = {
        "mean_prob_cib": sampled_sums[:, 1] / sampled_n,
        "mean_prob_oib": sampled_sums[:, 2] / sampled_n,
        "mean_transition_index": sampled_sums[:, 3] / sampled_n,
        "share_prob_oib_gt_cib": sampled_sums[:, 4] / sampled_n,
    }
    output: dict[str, tuple[float, float, float]] = {}
    for name, bootstrap_values in metrics.items():
        estimate = {
            "mean_prob_cib": frame["prob_cib"].mean(),
            "mean_prob_oib": frame["prob_oib"].mean(),
            "mean_transition_index": frame["transition_index"].mean(),
            "share_prob_oib_gt_cib": frame["prob_oib"].gt(frame["prob_cib"]).mean(),
        }[name]
        low, high = np.quantile(bootstrap_values, [0.025, 0.975])
        output[name] = (float(estimate), float(low), float(high))
    return output


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    configured = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["feature_sets"][
        "broad_v1"
    ]
    manifest_features = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))["features"]
    data = pd.read_parquet(DATA_PATH)
    data = data.loc[data["model_ready_broad"]].reset_index(drop=True)
    features = [feature for feature in configured if feature in manifest_features]
    X = np.log10(data[features].where(data[features] > 0))
    y = data["label_primary4"].astype(str)
    ears_mask = data["location_root"].eq("EAST AFRICAN RIFT")
    train_index = np.flatnonzero(~ears_mask)
    test_index = np.flatnonzero(ears_mask)
    if len(test_index) != 361:
        raise ValueError(f"Unexpected East African Rift cohort: {len(test_index)}")

    metadata_columns = [
        "record_id",
        "label_primary4",
        "citation_set",
        "citations",
        "sample_names",
        "geographic_features",
        "location_root",
    ]
    prediction_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(SEED)
    for model_name, model in model_templates().items():
        model.fit(X.iloc[train_index], y.iloc[train_index])
        raw_probability = model.predict_proba(X.iloc[test_index])
        classes = list(model.classes_)
        probability = np.column_stack(
            [raw_probability[:, classes.index(label)] for label in LABELS]
        )
        predicted = np.asarray(LABELS)[np.argmax(probability, axis=1)]
        frame = data.iloc[test_index][metadata_columns].reset_index(drop=True).copy()
        frame = frame.rename(columns={"label_primary4": "actual"})
        frame.insert(1, "model", model_name)
        frame["country"] = frame["geographic_features"].map(extract_country)
        frame["predicted"] = predicted
        for index, label in enumerate(LABELS):
            frame[f"prob_{label.lower()}"] = probability[:, index]
        denominator = frame["prob_oib"] + frame["prob_cib"]
        frame["transition_index"] = frame["prob_oib"] / denominator.replace(0, np.nan)
        prediction_frames.append(frame)

        for actual, cohort in [("ALL", frame), ("CIB", frame.loc[frame["actual"].eq("CIB")])]:
            intervals = citation_cluster_bootstrap(cohort, rng)
            row: dict[str, object] = {
                "model": model_name,
                "actual_cohort": actual,
                "n": len(cohort),
                "citation_groups": cohort["citation_set"].nunique(),
                "accuracy_against_database_label": cohort["predicted"].eq(cohort["actual"]).mean(),
                "predicted_arc": int(cohort["predicted"].eq("ARC").sum()),
                "predicted_cib": int(cohort["predicted"].eq("CIB").sum()),
                "predicted_morb": int(cohort["predicted"].eq("MORB").sum()),
                "predicted_oib": int(cohort["predicted"].eq("OIB").sum()),
            }
            for name, (estimate, low, high) in intervals.items():
                row[name] = estimate
                row[f"{name}_ci_low"] = low
                row[f"{name}_ci_high"] = high
            summary_rows.append(row)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    predictions.to_parquet(OUT_DIR / "east_african_rift_stress_predictions.parquet", index=False)
    summary.to_csv(OUT_DIR / "east_african_rift_stress_summary.csv", index=False)

    country = (
        predictions.groupby(["model", "country", "actual"], observed=True)
        .agg(
            n=("record_id", "size"),
            mean_prob_arc=("prob_arc", "mean"),
            mean_prob_cib=("prob_cib", "mean"),
            mean_prob_morb=("prob_morb", "mean"),
            mean_prob_oib=("prob_oib", "mean"),
            mean_transition_index=("transition_index", "mean"),
        )
        .reset_index()
    )
    country.to_csv(OUT_DIR / "east_african_rift_stress_geography.csv", index=False)

    audit_manifest = {
        "purpose": (
            "Geology-led external province stress test. All East African Rift samples are excluded "
            "from training; class probabilities are treated as geochemical affinities rather than "
            "proof that the database tectonic label is wrong."
        ),
        "training_rows": int(len(train_index)),
        "test_rows": int(len(test_index)),
        "test_actual_counts": y.iloc[test_index].value_counts().to_dict(),
        "features": features,
        "models": list(model_templates()),
        "uncertainty": (
            "95% percentile intervals from 2,000 cluster bootstrap resamples of citation sets; "
            "this preserves within-publication dependence better than sample-wise bootstrap."
        ),
        "transition_index": "p(OIB) / [p(OIB) + p(CIB)]",
        "interpretation_caveat": (
            "High OIB affinity is consistent with a rift-plume/source-mixing continuum, but the "
            "classifier and bulk-rock medians cannot uniquely separate mantle source, melt fraction, "
            "fractionation, crustal interaction, alteration, or analytical/literature selection."
        ),
    }
    (OUT_DIR / "east_african_rift_stress_manifest.json").write_text(
        json.dumps(audit_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
