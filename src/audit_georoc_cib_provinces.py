from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "data" / "interim" / "georoc_basalt_core_index.csv.gz"
CITATIONS = ROOT / "data" / "raw" / "2024-12-4EZ7ID_CITATIONS.tab"
PETDB = ROOT / "data" / "processed" / "petdb_primary4_v0_1.parquet"
OUT = ROOT / "reports" / "data_quality"
FEATURES = [
    "SIO2(WT%)",
    "TIO2(WT%)",
    "AL2O3(WT%)",
    "FEOT_HARMONIZED(WT%)",
    "CAO(WT%)",
    "MGO(WT%)",
    "MNO(WT%)",
    "K2O(WT%)",
    "NA2O(WT%)",
    "P2O5(WT%)",
    "RB(PPM)",
    "SR(PPM)",
    "Y(PPM)",
    "ZR(PPM)",
    "NB(PPM)",
    "LA(PPM)",
    "CE(PPM)",
    "ND(PPM)",
]
MAJORS = FEATURES[:10]
CANDIDATES = {
    "Asal Rift, Djibouti": "RED SEA RIFT / ASAL RIFT / DJIBOUTI",
    "Main Ethiopian Rift": "EAST AFRICAN RIFT / ETHIOPIAN RIFT / MAIN ETHIOPIAN RIFT",
    "Baikal Rift system": "BAIKAL",
    "Oslo Rift": "OSLO-SKAGERRAK RIFT / OSLO RIFT",
    "Big Pine volcanic field": "BIG PINE VOLCANIC FIELD",
    "Cima volcanic field": "CIMA VOLCANIC FIELD",
    "West Antarctic Rift": "ANTARCTICA / WEST ANTARCTIC RIFT SYSTEM",
    "Virunga province": "VIRUNGA PROVINCE",
    "San Rafael subvolcanic field": "SAN RAFAEL SUBVOLCANIC FIELD",
    "Pantelleria": "SICILY CHANNEL RIFT / PANTELLERIA",
}


def normalize_name(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def citation_ids(value: object) -> list[int]:
    return [int(item) for item in re.findall(r"\[(\d+)\]", str(value))]


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
        "source_file",
        "flag_strict_candidate",
        "FEOT(WT%)",
        "FE2O3(WT%)",
        "FEO(WT%)",
    ] + [feature for feature in FEATURES if feature != "FEOT_HARMONIZED(WT%)"]
    data = pd.read_csv(INDEX, usecols=usecols, low_memory=False)
    data = data.loc[data["TECTONIC SETTING"].eq("RIFT VOLCANICS")].copy()
    data["FEOT_HARMONIZED(WT%)"] = data["FEOT(WT%)"]
    calculated_feot = data["FEO(WT%)"].fillna(0) + 0.8998 * data[
        "FE2O3(WT%)"
    ].fillna(0)
    has_reported_iron = data[["FEO(WT%)", "FE2O3(WT%)"]].notna().any(axis=1)
    data.loc[
        data["FEOT_HARMONIZED(WT%)"].isna() & has_reported_iron,
        "FEOT_HARMONIZED(WT%)",
    ] = calculated_feot
    data["observed_common18_n"] = data[FEATURES].notna().sum(axis=1)
    data["major_complete"] = data[MAJORS].notna().all(axis=1)
    data["major_total"] = data[MAJORS].sum(axis=1, min_count=len(MAJORS))
    data["model_ready_common18"] = (
        data["flag_strict_candidate"].fillna(False)
        & data["major_complete"]
        & data["major_total"].between(90, 105)
        & data["SIO2(WT%)"].between(40, 55)
        & data["observed_common18_n"].eq(len(FEATURES))
        & data[FEATURES].gt(0).all(axis=1)
    )

    petdb = pd.read_parquet(PETDB, columns=["sample_names", "citations"])
    petdb_names = {
        normalize_name(part)
        for value in petdb["sample_names"].dropna()
        for part in str(value).split(" || ")
        if normalize_name(part)
    }
    petdb_citation_text = "\n".join(petdb["citations"].dropna().astype(str)).upper()
    catalog = pd.read_csv(CITATIONS, low_memory=False).set_index("CITATIONS")

    summary_rows: list[dict[str, object]] = []
    citation_rows: list[dict[str, object]] = []
    for province, pattern in CANDIDATES.items():
        mask = data["LOCATION"].astype(str).str.contains(
            re.escape(pattern), case=False, na=False
        )
        province_data = data.loc[mask].copy()
        ready = province_data.loc[province_data["model_ready_common18"]].copy()
        ids = sorted(
            {
                item
                for value in ready["CITATIONS"].dropna()
                for item in citation_ids(value)
            }
        )
        publications_in_petdb = 0
        mapped_ids = 0
        for item in ids:
            mapped = item in catalog.index
            doi = None
            title = None
            year = None
            if mapped:
                mapped_ids += 1
                citation = catalog.loc[item]
                if isinstance(citation, pd.DataFrame):
                    citation = citation.iloc[0]
                doi = citation.get("DOI")
                title = citation.get("TITLE")
                year = citation.get("YEAR")
            doi_overlap = bool(
                pd.notna(doi) and str(doi).upper() in petdb_citation_text
            )
            title_overlap = bool(
                pd.notna(title)
                and len(normalize_name(title)) > 24
                and normalize_name(title) in normalize_name(petdb_citation_text)
            )
            overlap = doi_overlap or title_overlap
            publications_in_petdb += int(overlap)
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

        exact_names = ready["SAMPLE NAME"].map(normalize_name).isin(petdb_names)
        summary_rows.append(
            {
                "province": province,
                "location_pattern": pattern,
                "rift_rows": len(province_data),
                "strict_index_rows": int(
                    province_data["flag_strict_candidate"].fillna(False).sum()
                ),
                "model_ready_common18": len(ready),
                "citation_id_n": len(ids),
                "citation_id_mapped_n": mapped_ids,
                "publication_overlap_petdb_n": publications_in_petdb,
                "exact_sample_name_collision_n": int(exact_names.sum()),
                "latitude_min": finite(ready["LATITUDE MIN"].min()),
                "latitude_max": finite(ready["LATITUDE MIN"].max()),
                "longitude_min": finite(ready["LONGITUDE MIN"].min()),
                "longitude_max": finite(ready["LONGITUDE MIN"].max()),
                "median_nb_y": finite(
                    (ready["NB(PPM)"] / ready["Y(PPM)"]).median()
                ),
                "median_zr_y": finite(
                    (ready["ZR(PPM)"] / ready["Y(PPM)"]).median()
                ),
                "median_thickness_proxy_k2o": finite(ready["K2O(WT%)"].median()),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["publication_overlap_petdb_n", "model_ready_common18"],
        ascending=[True, False],
    )
    summary.to_csv(OUT / "georoc_cib_province_audit.csv", index=False)
    pd.DataFrame(citation_rows).to_csv(
        OUT / "georoc_cib_province_citations.csv", index=False
    )
    print(summary.to_string(index=False))


def finite(value: object) -> float | None:
    if pd.isna(value) or not np.isfinite(value):
        return None
    return float(value)


if __name__ == "__main__":
    main()
