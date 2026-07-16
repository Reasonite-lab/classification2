from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from audit_georoc_cib_provinces import FEATURES, INDEX, MAJORS, normalize_name


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "reports" / "data_quality"
LOCATION_PATTERN = "BIG PINE VOLCANIC FIELD"


def citation_ids(value: object) -> str:
    return "|".join(re.findall(r"\[(\d+)\]", str(value)))


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INDEX, low_memory=False)
    data = data.loc[
        data["TECTONIC SETTING"].eq("RIFT VOLCANICS")
        & data["LOCATION"].astype(str).str.contains(
            LOCATION_PATTERN, case=False, na=False
        )
    ].copy()
    data["FEOT_REPORTED(WT%)"] = data["FEOT(WT%)"]
    data["FEOT_HARMONIZED(WT%)"] = data["FEOT(WT%)"]
    calculated_feot = data["FEO(WT%)"].fillna(0) + 0.8998 * data[
        "FE2O3(WT%)"
    ].fillna(0)
    has_reported_iron = data[["FEO(WT%)", "FE2O3(WT%)"]].notna().any(axis=1)
    data.loc[
        data["FEOT_HARMONIZED(WT%)"].isna() & has_reported_iron,
        "FEOT_HARMONIZED(WT%)",
    ] = calculated_feot
    data["FEOT(WT%)"] = data["FEOT_HARMONIZED(WT%)"]
    model_features = [
        "FEOT(WT%)" if feature == "FEOT_HARMONIZED(WT%)" else feature
        for feature in FEATURES
    ]
    model_majors = model_features[: len(MAJORS)]
    data["major_total"] = data[model_majors].sum(
        axis=1, min_count=len(model_majors)
    )
    data["observed_feature_n"] = data[model_features].notna().sum(axis=1)
    data["model_ready_common18"] = (
        data["flag_strict_candidate"].fillna(False)
        & data["SIO2(WT%)"].between(40, 55)
        & data["major_total"].between(90, 105)
        & data["observed_feature_n"].eq(len(model_features))
        & data[model_features].gt(0).all(axis=1)
    )
    data = data.loc[data["model_ready_common18"]].copy()

    petdb = pd.read_parquet(
        PROCESSED / "petdb_primary4_v0_1.parquet",
        columns=["sample_names", "citations", "location_root"],
    )
    petdb_names = {
        normalize_name(part)
        for value in petdb["sample_names"].dropna()
        for part in str(value).split(" || ")
        if normalize_name(part)
    }
    data["normalized_sample_name"] = data["SAMPLE NAME"].map(normalize_name)
    data["petdb_exact_name_collision"] = data["normalized_sample_name"].isin(
        petdb_names
    )
    data["sample_name"] = data["SAMPLE NAME"].astype(str)
    data["publication_group"] = data["CITATIONS"].map(citation_ids)
    data["record_id"] = "georoc:2jetoa:" + data["UNIQUE_ID"].astype(str)
    data["external_source"] = "GEOROC 2JETOA (2026-06)"
    data["source_dataset_doi"] = "10.25625/2JETOA"
    data["source_license"] = "CC BY-SA 4.0"
    data["source_area"] = "Big Pine volcanic field, California"
    data["province_proxy"] = "Big Pine volcanic field"
    data["mechanism_cohort"] = "CONTINENTAL_RIFT_BASIN_RANGE"
    data["actual_label"] = "CIB"
    data["strict_external_candidate"] = True
    data.to_parquet(
        PROCESSED / "georoc_big_pine_cib_external_v0_1.parquet", index=False
    )

    citation_audit = pd.read_csv(OUT / "georoc_cib_province_citations.csv")
    citation_audit = citation_audit.loc[
        citation_audit["province"].eq("Big Pine volcanic field")
    ]
    audit = {
        "source_dataset_doi": "10.25625/2JETOA",
        "source_version": "2026-06-01",
        "source_license": "CC BY-SA 4.0",
        "source_index": str(INDEX.relative_to(ROOT)),
        "location_pattern": LOCATION_PATTERN,
        "strict_model_ready_common18": len(data),
        "publication_group_counts": data["publication_group"].value_counts().to_dict(),
        "publication_overlap_petdb_n": int(
            citation_audit["publication_overlap_petdb"].fillna(False).sum()
        ),
        "exact_sample_name_collision_n": int(
            data["petdb_exact_name_collision"].sum()
        ),
        "petdb_location_overlap_n": int(
            petdb["location_root"].astype(str).str.contains(
                LOCATION_PATTERN, case=False, na=False
            ).sum()
        ),
        "latitude_range": [
            float(data["LATITUDE MIN"].min()),
            float(data["LATITUDE MIN"].max()),
        ],
        "longitude_range": [
            float(data["LONGITUDE MIN"].min()),
            float(data["LONGITUDE MIN"].max()),
        ],
        "major_total_range": [
            float(data["major_total"].min()),
            float(data["major_total"].max()),
        ],
        "selection_policy": (
            "GEOROC whole-rock volcanic basalt candidates assigned to RIFT VOLCANICS "
            "and Big Pine volcanic field; SiO2 40-55 wt%, positive complete common18, "
            "and harmonized major total 90-105 wt%."
        ),
        "scope_caveat": (
            "This is an independently sourced province-level GEOROC cohort compiled "
            "from six publication groups, not a new single-study analytical campaign."
        ),
    }
    (OUT / "georoc_big_pine_cib_external_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
