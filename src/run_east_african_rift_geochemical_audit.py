from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "petdb_primary4_v0_1.parquet"
PREDICTIONS = ROOT / "reports" / "modeling" / "petdb_primary4_oof_predictions.parquet"
MODEL_MANIFEST = ROOT / "reports" / "modeling" / "petdb_primary4_validation_manifest.json"
OUT = ROOT / "reports" / "modeling"
SEED = 20260715


def extract_country(value: object) -> str:
    matches = re.findall(r"COUNTRY\s*\|\s*([^,\"|]+)", str(value), flags=re.IGNORECASE)
    normalized = sorted({re.sub(r"\s+", " ", item).strip().upper() for item in matches})
    return " | ".join(normalized) if normalized else "<UNRESOLVED>"


def bootstrap_median_difference(
    left: np.ndarray, right: np.ndarray, rng: np.random.Generator, repeats: int = 1000
) -> tuple[float, float, float]:
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]
    if len(left) < 5 or len(right) < 5:
        return np.nan, np.nan, np.nan
    estimate = float(np.median(left) - np.median(right))
    values = np.empty(repeats)
    for index in range(repeats):
        values[index] = np.median(rng.choice(left, len(left), replace=True)) - np.median(
            rng.choice(right, len(right), replace=True)
        )
    low, high = np.quantile(values, [0.025, 0.975])
    return estimate, float(low), float(high)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    features = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))["features"]
    data = pd.read_parquet(DATA)
    data = data.loc[data["model_ready_broad"]].reset_index(drop=True)
    data["country_audit"] = data["geographic_features"].map(extract_country)
    predictions = pd.read_parquet(PREDICTIONS)
    predictions = predictions[
        predictions["cv_strategy"].eq("location_root_grouped")
        & predictions["model"].eq("random_forest_balanced")
    ]
    joined = predictions.merge(
        data[
            [
                "record_id",
                "location_root",
                "country_audit",
                "citation_set",
                "citations",
                "sample_names",
                "geographic_features",
                *features,
            ]
        ],
        on="record_id",
        how="left",
        validate="one_to_one",
    )
    ears = joined[joined["location_root"].eq("EAST AFRICAN RIFT")].copy()
    ears["is_correct"] = ears["actual"].eq(ears["predicted"])
    if len(ears) != 361:
        raise ValueError(f"Unexpected East African Rift cohort: {len(ears)}")

    country = (
        ears.groupby(["country_audit", "actual", "predicted"], observed=True)
        .agg(
            n=("record_id", "size"),
            mean_predicted_confidence=("prob_arc", lambda _: np.nan),
        )
        .reset_index()
        .drop(columns="mean_predicted_confidence")
    )
    country_totals = ears.groupby(["country_audit", "actual"])["record_id"].size().rename("actual_n")
    country = country.join(country_totals, on=["country_audit", "actual"])
    country["share_of_country_actual"] = country["n"] / country["actual_n"]
    country.to_csv(OUT / "east_african_rift_country_confusion.csv", index=False)

    citation = (
        ears.groupby(["citation_set", "actual"], observed=True)
        .agg(
            n=("record_id", "size"),
            correct=("is_correct", "sum"),
            predicted_arc=("predicted", lambda values: int(values.eq("ARC").sum())),
            predicted_cib=("predicted", lambda values: int(values.eq("CIB").sum())),
            predicted_morb=("predicted", lambda values: int(values.eq("MORB").sum())),
            predicted_oib=("predicted", lambda values: int(values.eq("OIB").sum())),
            citation=("citations", "first"),
            sample_examples=("sample_names", lambda values: " || ".join(values.astype(str).head(5))),
            countries=("country_audit", lambda values: " || ".join(sorted(set(values)))),
        )
        .reset_index()
    )
    citation["error_rate"] = 1 - citation["correct"] / citation["n"]
    citation = citation.sort_values(["n", "error_rate"], ascending=[False, False])
    citation.to_csv(OUT / "east_african_rift_citation_audit.csv", index=False)

    log_values = np.log10(data[features].astype(float).mask(data[features].astype(float) <= 0))
    ears_cib_mask = data["location_root"].eq("EAST AFRICAN RIFT") & data["label_primary4"].eq("CIB")
    other_cib_mask = ~data["location_root"].eq("EAST AFRICAN RIFT") & data["label_primary4"].eq("CIB")
    oib_mask = data["label_primary4"].eq("OIB")
    global_median = log_values.median()
    global_mad = (log_values - global_median).abs().median().replace(0, np.nan).fillna(0.1)
    rng = np.random.default_rng(SEED)
    feature_rows: list[dict[str, object]] = []
    for feature in features:
        ears_values = log_values.loc[ears_cib_mask, feature].dropna().to_numpy()
        other_values = log_values.loc[other_cib_mask, feature].dropna().to_numpy()
        oib_values = log_values.loc[oib_mask, feature].dropna().to_numpy()
        ears_median = float(np.median(ears_values)) if len(ears_values) else np.nan
        other_median = float(np.median(other_values)) if len(other_values) else np.nan
        oib_median = float(np.median(oib_values)) if len(oib_values) else np.nan
        difference, low, high = bootstrap_median_difference(ears_values, other_values, rng)
        difference_oib, low_oib, high_oib = bootstrap_median_difference(ears_values, oib_values, rng)
        scale = float(global_mad[feature])
        feature_rows.append(
            {
                "feature": feature,
                "ears_cib_n": len(ears_values),
                "other_cib_n": len(other_values),
                "oib_n": len(oib_values),
                "ears_cib_median_log10": ears_median,
                "other_cib_median_log10": other_median,
                "oib_median_log10": oib_median,
                "ears_minus_other_cib_log10": difference,
                "ears_minus_other_cib_ci_low": low,
                "ears_minus_other_cib_ci_high": high,
                "ears_minus_oib_log10": difference_oib,
                "ears_minus_oib_ci_low": low_oib,
                "ears_minus_oib_ci_high": high_oib,
                "ears_shift_from_other_cib_global_mad": difference / scale,
                "oib_proximity_gain_global_mad": (
                    abs(ears_median - other_median) - abs(ears_median - oib_median)
                )
                / scale,
                "ears_closer_to_oib_than_other_cib": abs(ears_median - oib_median)
                < abs(ears_median - other_median),
            }
        )
    feature_shift = pd.DataFrame(feature_rows).sort_values(
        "oib_proximity_gain_global_mad", ascending=False
    )
    feature_shift.to_csv(OUT / "east_african_rift_feature_shift.csv", index=False)

    confusion = pd.crosstab(ears["actual"], ears["predicted"])
    profile = {
        "cohort": "PetDB primary4 broad model-ready samples with location_root EAST AFRICAN RIFT",
        "rows": len(ears),
        "actual_counts": ears["actual"].value_counts().to_dict(),
        "country_counts": ears["country_audit"].value_counts().to_dict(),
        "confusion": confusion.to_dict(orient="index"),
        "overall_correct": int(ears["is_correct"].sum()),
        "overall_error_rate": float(1 - ears["is_correct"].mean()),
        "cib_recall": float(
            ears.loc[ears["actual"].eq("CIB"), "predicted"].eq("CIB").mean()
        ),
        "cib_to_oib_share": float(
            ears.loc[ears["actual"].eq("CIB"), "predicted"].eq("OIB").mean()
        ),
        "interpretation_caveat": (
            "The audit demonstrates province-level domain shift and a rift-to-plume geochemical continuum. "
            "Class-median shifts mix mantle source, melting, differentiation, analytical coverage, and literature selection; "
            "they do not identify a unique causal mechanism."
        ),
    }
    (OUT / "east_african_rift_audit_manifest.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(profile, ensure_ascii=False, indent=2))
    print("\nTop OIB-proximity shifts")
    print(
        feature_shift[
            [
                "feature",
                "ears_cib_n",
                "ears_shift_from_other_cib_global_mad",
                "oib_proximity_gain_global_mad",
                "ears_closer_to_oib_than_other_cib",
            ]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
