from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from audit_georoc_cib_provinces import (
    CITATIONS,
    FEATURES,
    INDEX,
    MAJORS,
    normalize_name,
)


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "reports" / "data_quality"
BASE_PATTERN = (
    "CENTRAL AMERICAN VOLCANIC ARC / CENTRAL AMERICAN VOLCANIC "
    "ARC-SOUTHEASTERN SEGMENT / COSTA RICA /"
)
VOLCANIC_CENTERS = [
    "IRAZU",
    "POAS (POAZ)",
    "MIRAVALLES",
    "TURRIALBA",
    "PLATANAR",
]


def citation_ids(value: object) -> list[int]:
    return [int(item) for item in re.findall(r"\[(\d+)\]", str(value))]


def citation_group(value: object) -> str:
    return "|".join(str(item) for item in citation_ids(value))


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    usecols = [
        "CITATIONS",
        "TECTONIC SETTING",
        "LOCATION",
        "LATITUDE MIN",
        "LONGITUDE MIN",
        "SAMPLE NAME",
        "UNIQUE_ID",
        "source_file",
        "flag_strict_candidate",
        "FEOT(WT%)",
        "FE2O3(WT%)",
        "FEO(WT%)",
    ] + [feature for feature in FEATURES if feature != "FEOT_HARMONIZED(WT%)"]
    data = pd.read_csv(INDEX, usecols=usecols, low_memory=False)
    center_pattern = "(?:" + "|".join(
        re.escape(center) for center in VOLCANIC_CENTERS
    ) + ")$"
    data = data.loc[
        data["TECTONIC SETTING"].eq("CONVERGENT MARGIN")
        & data["LOCATION"].astype(str).str.contains(
            re.escape(BASE_PATTERN), case=False, na=False
        )
        & data["LOCATION"].astype(str).str.contains(
            center_pattern, case=False, na=False, regex=True
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
    raw_named_center_rows = len(data)
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
    petdb_citation_text = "\n".join(
        petdb["citations"].dropna().astype(str)
    ).upper()
    petdb_citation_normalized = normalize_name(petdb_citation_text)
    data["normalized_sample_name"] = data["SAMPLE NAME"].map(normalize_name)
    data["petdb_exact_name_collision"] = data["normalized_sample_name"].isin(
        petdb_names
    )
    data["sample_name"] = data["SAMPLE NAME"].astype(str)
    data["publication_group"] = data["CITATIONS"].map(citation_group)
    data["record_id"] = "georoc:2jetoa:costa-rica:" + data[
        "UNIQUE_ID"
    ].astype(str)
    data["external_source"] = "GEOROC 2JETOA (2026-06)"
    data["source_dataset_doi"] = "10.25625/2JETOA"
    data["source_license"] = "CC BY-SA 4.0"
    data["source_area"] = "Costa Rica active volcanic front"
    data["province_proxy"] = "Costa Rica volcanic front"
    data["mechanism_cohort"] = "ARC_FRONT"
    data["actual_label"] = "ARC"
    data["strict_external_candidate"] = True
    data.to_parquet(
        PROCESSED / "georoc_costa_rica_arc_external_v0_1.parquet", index=False
    )

    catalog = pd.read_csv(CITATIONS, low_memory=False).set_index("CITATIONS")
    selected_ids = sorted(
        {
            item
            for value in data["CITATIONS"].dropna()
            for item in citation_ids(value)
        }
    )
    citation_rows: list[dict[str, object]] = []
    for item in selected_ids:
        mapped = item in catalog.index
        doi = None
        title = None
        year = None
        if mapped:
            citation = catalog.loc[item]
            if isinstance(citation, pd.DataFrame):
                citation = citation.iloc[0]
            doi = citation.get("DOI")
            title = citation.get("TITLE")
            year = citation.get("YEAR")
        doi_overlap = bool(
            pd.notna(doi) and str(doi).upper() in petdb_citation_text
        )
        normalized_title = normalize_name(title) if pd.notna(title) else ""
        title_overlap = bool(
            len(normalized_title) > 24
            and normalized_title in petdb_citation_normalized
        )
        citation_rows.append(
            {
                "georoc_citation_id": item,
                "mapped": mapped,
                "doi": doi,
                "year": year,
                "title": title,
                "doi_overlap_petdb": doi_overlap,
                "title_overlap_petdb": title_overlap,
                "publication_overlap_petdb": doi_overlap or title_overlap,
            }
        )
    citation_audit = pd.DataFrame(citation_rows)
    citation_audit.to_csv(
        OUT / "georoc_costa_rica_arc_citations.csv", index=False
    )
    centers = data["LOCATION"].str.rsplit(" / ", n=1).str[-1]
    petdb_location = petdb["location_root"].fillna("").astype(str)
    audit = {
        "source_dataset_doi": "10.25625/2JETOA",
        "source_version": "2026-06-01",
        "source_license": "CC BY-SA 4.0",
        "source_index": str(INDEX.relative_to(ROOT)),
        "selection_base": BASE_PATTERN,
        "named_volcanic_centers": VOLCANIC_CENTERS,
        "raw_named_center_rows": raw_named_center_rows,
        "strict_model_ready_common18": len(data),
        "center_counts": centers.value_counts().to_dict(),
        "publication_group_counts": data[
            "publication_group"
        ].value_counts().to_dict(),
        "publication_overlap_petdb_n": int(
            citation_audit["publication_overlap_petdb"].sum()
        ),
        "exact_sample_name_collision_n": int(
            data["petdb_exact_name_collision"].sum()
        ),
        "petdb_named_center_location_rows": int(
            petdb_location.str.contains(
                "IRAZU|POAS|MIRAVALLES|TURRIALBA|PLATANAR",
                case=False,
                regex=True,
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
            "GEOROC strict whole-rock volcanic basalt candidates assigned to the "
            "convergent margin and five named Costa Rica volcanic-front centers; SiO2 "
            "40-55 wt%, positive complete common18, major total 90-105 wt%. Broad "
            "country-level rows, old arc terranes, forearc complexes, plateau and "
            "accreted seamount records are excluded."
        ),
        "scope_caveat": (
            "A multi-publication active-arc province proxy. Publication and exact-name "
            "overlap are excluded; geographic overlap by volcano name is retained as a "
            "separate audit because it does not establish sample identity."
        ),
    }
    (OUT / "georoc_costa_rica_arc_external_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
