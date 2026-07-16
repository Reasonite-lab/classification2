from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from audit_georoc_cib_provinces import (
    CITATIONS,
    FEATURES,
    INDEX,
    MAJORS,
    PETDB,
    normalize_name,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "data_quality"
CANDIDATES = {
    "Central Aleutian arc": "ALEUTIAN ARC / CENTRAL ALEUTIAN ARC",
    "Southern Andean Volcanic Zone": (
        "ANDEAN ARC / SOUTHERN ANDEAN VOLCANIC ZONE"
    ),
    "Kamchatka arc front": "KAMCHATKA ARC / KAMCHATKA ARC",
    "High Cascades": "CASCADES / HIGH CASCADES",
    "Central America southeastern segment": (
        "CENTRAL AMERICAN VOLCANIC ARC / CENTRAL AMERICAN VOLCANIC "
        "ARC-SOUTHEASTERN SEGMENT"
    ),
    "Central America northwestern segment": (
        "CENTRAL AMERICAN VOLCANIC ARC / CENTRAL AMERICAN VOLCANIC "
        "ARC-NORTHWESTERN SEGMENT"
    ),
    "South Shetland Islands": "SCOTIA ARC / SOUTH SHETLAND ISLANDS",
    "Tonga islands": "TONGA ARC / TONGA ISLANDS",
}


def citation_ids(value: object) -> list[int]:
    return [int(item) for item in re.findall(r"\[(\d+)\]", str(value))]


def finite(value: object) -> float | None:
    if pd.isna(value) or not np.isfinite(value):
        return None
    return float(value)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    usecols = [
        "CITATIONS",
        "TECTONIC SETTING",
        "LOCATION",
        "LATITUDE MIN",
        "LONGITUDE MIN",
        "SAMPLE NAME",
        "UNIQUE_ID",
        "flag_strict_candidate",
        "FEOT(WT%)",
        "FE2O3(WT%)",
        "FEO(WT%)",
    ] + [feature for feature in FEATURES if feature != "FEOT_HARMONIZED(WT%)"]
    data = pd.read_csv(INDEX, usecols=usecols, low_memory=False)
    data = data.loc[data["TECTONIC SETTING"].eq("CONVERGENT MARGIN")].copy()
    data["FEOT_HARMONIZED(WT%)"] = data["FEOT(WT%)"]
    calculated_feot = data["FEO(WT%)"].fillna(0) + 0.8998 * data[
        "FE2O3(WT%)"
    ].fillna(0)
    has_reported_iron = data[["FEO(WT%)", "FE2O3(WT%)"]].notna().any(axis=1)
    data.loc[
        data["FEOT_HARMONIZED(WT%)"].isna() & has_reported_iron,
        "FEOT_HARMONIZED(WT%)",
    ] = calculated_feot
    data["major_total"] = data[MAJORS].sum(axis=1, min_count=len(MAJORS))
    data["model_ready_common18"] = (
        data["flag_strict_candidate"].fillna(False)
        & data["SIO2(WT%)"].between(40, 55)
        & data["major_total"].between(90, 105)
        & data[FEATURES].notna().all(axis=1)
        & data[FEATURES].gt(0).all(axis=1)
    )

    petdb = pd.read_parquet(PETDB, columns=["sample_names", "citations"])
    petdb_names = {
        normalize_name(part)
        for value in petdb["sample_names"].dropna()
        for part in str(value).split(" || ")
        if normalize_name(part)
    }
    petdb_citations = "\n".join(petdb["citations"].dropna().astype(str)).upper()
    petdb_citations_normalized = normalize_name(petdb_citations)
    catalog = pd.read_csv(CITATIONS, low_memory=False).set_index("CITATIONS")

    summary_rows: list[dict[str, object]] = []
    citation_rows: list[dict[str, object]] = []
    for province, pattern in CANDIDATES.items():
        selected = data.loc[
            data["LOCATION"].astype(str).str.contains(
                re.escape(pattern), case=False, na=False
            )
        ].copy()
        ready = selected.loc[selected["model_ready_common18"]].copy()
        ids = sorted(
            {
                item
                for value in ready["CITATIONS"].dropna()
                for item in citation_ids(value)
            }
        )
        publication_overlap_n = 0
        mapped_n = 0
        for item in ids:
            mapped = item in catalog.index
            doi = None
            title = None
            year = None
            if mapped:
                mapped_n += 1
                citation = catalog.loc[item]
                if isinstance(citation, pd.DataFrame):
                    citation = citation.iloc[0]
                doi = citation.get("DOI")
                title = citation.get("TITLE")
                year = citation.get("YEAR")
            doi_overlap = bool(
                pd.notna(doi) and str(doi).upper() in petdb_citations
            )
            normalized_title = normalize_name(title) if pd.notna(title) else ""
            title_overlap = bool(
                len(normalized_title) > 24
                and normalized_title in petdb_citations_normalized
            )
            overlap = doi_overlap or title_overlap
            publication_overlap_n += int(overlap)
            citation_rows.append(
                {
                    "province": province,
                    "georoc_citation_id": item,
                    "mapped": mapped,
                    "doi": doi,
                    "year": year,
                    "title": title,
                    "doi_overlap_petdb": doi_overlap,
                    "title_overlap_petdb": title_overlap,
                    "publication_overlap_petdb": overlap,
                }
            )
        exact_collisions = ready["SAMPLE NAME"].map(normalize_name).isin(
            petdb_names
        )
        summary_rows.append(
            {
                "province": province,
                "location_pattern": pattern,
                "convergent_margin_rows": len(selected),
                "model_ready_common18": len(ready),
                "location_n": ready["LOCATION"].nunique(),
                "citation_id_n": len(ids),
                "citation_id_mapped_n": mapped_n,
                "publication_overlap_petdb_n": publication_overlap_n,
                "exact_sample_name_collision_n": int(exact_collisions.sum()),
                "latitude_min": finite(ready["LATITUDE MIN"].min()),
                "latitude_max": finite(ready["LATITUDE MIN"].max()),
                "longitude_min": finite(ready["LONGITUDE MIN"].min()),
                "longitude_max": finite(ready["LONGITUDE MIN"].max()),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        [
            "publication_overlap_petdb_n",
            "exact_sample_name_collision_n",
            "model_ready_common18",
        ],
        ascending=[True, True, False],
    )
    summary.to_csv(OUT / "georoc_arc_province_audit.csv", index=False)
    pd.DataFrame(citation_rows).to_csv(
        OUT / "georoc_arc_province_citations.csv", index=False
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
