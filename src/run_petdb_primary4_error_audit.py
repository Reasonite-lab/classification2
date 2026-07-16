from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "petdb_primary4_v0_1.parquet"
PREDICTIONS = ROOT / "reports" / "modeling" / "petdb_primary4_oof_predictions.parquet"
VALIDATION_MANIFEST = ROOT / "reports" / "modeling" / "petdb_primary4_validation_manifest.json"
OUT = ROOT / "reports" / "modeling"
LABELS = ["ARC", "CIB", "MORB", "OIB"]
PROBABILITY_COLUMNS = [f"prob_{label.lower()}" for label in LABELS]
STRATEGIES = {
    "citation_overlap_conservative": "citation_overlap_component",
    "location_root_grouped": "location_root",
}


def robust_profiles(log_values: pd.DataFrame, labels: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    medians = log_values.groupby(labels).median()
    mad_rows: list[pd.Series] = []
    for label in LABELS:
        frame = log_values.loc[labels.eq(label)]
        mad_rows.append((frame - medians.loc[label]).abs().median().rename(label))
    mad = pd.DataFrame(mad_rows).replace(0, np.nan)
    fallback = log_values.mad(axis=0) if hasattr(log_values, "mad") else None
    if fallback is None:
        global_median = log_values.median()
        fallback = (log_values - global_median).abs().median()
    mad = mad.fillna(fallback).fillna(0.1).clip(lower=0.02)
    return medians, mad


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(VALIDATION_MANIFEST.read_text(encoding="utf-8"))
    features = manifest["features"]
    data = pd.read_parquet(DATA)
    data = data.loc[data["model_ready_broad"]].reset_index(drop=True)
    assert data["record_id"].is_unique
    predictions = pd.read_parquet(PREDICTIONS)
    predictions = predictions[
        predictions["model"].eq("random_forest_balanced")
        & predictions["cv_strategy"].isin(STRATEGIES)
    ].copy()
    metadata_columns = [
        "record_id",
        "SAMPLE_URL",
        "sample_names",
        "citations",
        "citation_set",
        "citation_overlap_component",
        "location_root",
        "geographic_features",
        "latitude",
        "longitude",
        "broad_observed_n",
        "major_total_calc",
        "alteration_severity",
    ]
    merged = predictions.merge(
        data[metadata_columns + features], on="record_id", how="left", validate="many_to_one"
    )
    assert merged["SAMPLE_URL"].notna().all()
    merged["predicted_confidence"] = merged[PROBABILITY_COLUMNS].max(axis=1)
    merged["actual_probability"] = [
        getattr(row, f"prob_{row.actual.lower()}") for row in merged.itertuples(index=False)
    ]
    merged["prediction_margin_over_actual"] = (
        merged["predicted_confidence"] - merged["actual_probability"]
    )
    merged["is_error"] = merged["actual"].ne(merged["predicted"])

    log_values = np.log10(data[features].where(data[features] > 0))
    medians, mad = robust_profiles(log_values, data["label_primary4"].astype(str))
    log_lookup = log_values.set_axis(data["record_id"], axis=0)
    error_rows: list[dict[str, object]] = []
    resemblance_rows: list[dict[str, object]] = []
    for _, row in merged.loc[merged["is_error"]].iterrows():
        values = log_lookup.loc[row["record_id"]]
        observed = values.notna()
        actual_distance = (
            (values - medians.loc[row["actual"]]).abs() / mad.loc[row["actual"]]
        )
        predicted_distance = (
            (values - medians.loc[row["predicted"]]).abs() / mad.loc[row["predicted"]]
        )
        delta = (actual_distance - predicted_distance).where(observed).sort_values(ascending=False)
        top = delta.dropna().head(5)
        base = row.drop(labels=features).to_dict()
        base["observed_common21_n"] = int(observed.sum())
        base["top_predicted_like_features"] = " || ".join(
            f"{feature}:{value:.2f}" for feature, value in top.items()
        )
        error_rows.append(base)
        for feature in features:
            if pd.notna(delta[feature]):
                resemblance_rows.append(
                    {
                        "cv_strategy": row["cv_strategy"],
                        "record_id": row["record_id"],
                        "actual": row["actual"],
                        "predicted": row["predicted"],
                        "feature": feature,
                        "log10_value": values[feature],
                        "distance_from_actual_median_mad": actual_distance[feature],
                        "distance_from_predicted_median_mad": predicted_distance[feature],
                        "predicted_resemblance_delta": delta[feature],
                    }
                )
    errors = pd.DataFrame(error_rows)
    resemblance = pd.DataFrame(resemblance_rows)
    errors = errors.sort_values(
        ["cv_strategy", "prediction_margin_over_actual"], ascending=[True, False]
    )
    errors.to_csv(OUT / "petdb_primary4_error_audit_samples.csv.gz", index=False, compression="gzip")
    errors.groupby("cv_strategy", group_keys=False).head(50).to_csv(
        OUT / "petdb_primary4_error_audit_top50.csv", index=False
    )
    resemblance.to_parquet(
        OUT / "petdb_primary4_error_feature_resemblance.parquet", index=False
    )

    pair_summary = (
        merged.groupby(["cv_strategy", "actual", "predicted"], observed=True)
        .agg(
            n=("record_id", "size"),
            mean_confidence=("predicted_confidence", "mean"),
            median_confidence=("predicted_confidence", "median"),
            mean_margin_over_actual=("prediction_margin_over_actual", "mean"),
        )
        .reset_index()
    )
    actual_totals = merged.groupby(["cv_strategy", "actual"])["record_id"].size().rename("actual_n")
    pair_summary = pair_summary.join(actual_totals, on=["cv_strategy", "actual"])
    pair_summary["share_of_actual"] = pair_summary["n"] / pair_summary["actual_n"]
    pair_summary.to_csv(OUT / "petdb_primary4_error_pair_summary.csv", index=False)

    error_resemblance = resemblance[
        resemblance["actual"].ne(resemblance["predicted"])
    ]
    feature_summary = (
        error_resemblance.groupby(
            ["cv_strategy", "actual", "predicted", "feature"], observed=True
        )
        .agg(
            observed_n=("record_id", "size"),
            median_predicted_resemblance_delta=("predicted_resemblance_delta", "median"),
            mean_predicted_resemblance_delta=("predicted_resemblance_delta", "mean"),
        )
        .reset_index()
    )
    feature_summary["rank_within_pair"] = feature_summary.groupby(
        ["cv_strategy", "actual", "predicted"]
    )["median_predicted_resemblance_delta"].rank(method="first", ascending=False)
    feature_summary.to_csv(
        OUT / "petdb_primary4_error_feature_resemblance_summary.csv", index=False
    )

    group_rows: list[pd.DataFrame] = []
    for strategy, group_column in STRATEGIES.items():
        frame = merged[merged["cv_strategy"].eq(strategy)].copy()
        grouped = (
            frame.groupby(group_column, dropna=False)
            .agg(
                n=("record_id", "size"),
                errors=("is_error", "sum"),
                mean_confidence=("predicted_confidence", "mean"),
                dominant_actual=("actual", lambda values: values.value_counts().index[0]),
            )
            .reset_index()
            .rename(columns={group_column: "group_value"})
        )
        grouped["error_rate"] = grouped["errors"] / grouped["n"]
        grouped["cv_strategy"] = strategy
        grouped["group_field"] = group_column
        group_rows.append(grouped)
    group_risk = pd.concat(group_rows, ignore_index=True)
    group_risk.to_csv(OUT / "petdb_primary4_error_group_risk.csv", index=False)

    profile = {
        "cohort_rows": len(data),
        "features": features,
        "strategies": STRATEGIES,
        "prediction_rows": len(merged),
        "errors_by_strategy": merged.groupby("cv_strategy")["is_error"].sum().astype(int).to_dict(),
        "error_rates_by_strategy": merged.groupby("cv_strategy")["is_error"].mean().to_dict(),
        "resemblance_definition": (
            "For each observed log10 feature, absolute distance to actual- and predicted-class "
            "medians is scaled by the corresponding class MAD. Positive delta means closer to "
            "the predicted-class median than to the actual-class median. This is descriptive, not SHAP."
        ),
        "scope_caveat": (
            "OOF error audit for the primary common-21 random forest. Citation/location metadata "
            "and class-median resemblance are for tracing and geological review, not causal attribution."
        ),
    }
    (OUT / "petdb_primary4_error_audit_manifest.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(profile, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
