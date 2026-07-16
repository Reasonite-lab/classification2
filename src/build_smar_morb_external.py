from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT / ".deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import pandas as pd


RAW = ROOT / "data" / "raw" / "external_candidates"
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "reports" / "data_quality"
SOURCE_ZIP = RAW / "minerals-09-00659-s001.zip"
SOURCE_XLS = RAW / "smar_mdpi_supplement" / "Table S1.xls"
FEATURES = [
    "SIO2(WT%)",
    "TIO2(WT%)",
    "AL2O3(WT%)",
    "FEOT(WT%)",
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
ROW_MAP = {
    "SIO2(WT%)": "SiO2",
    "TIO2(WT%)": "TiO2",
    "AL2O3(WT%)": "Al2O3",
    "CAO(WT%)": "CaO",
    "MGO(WT%)": "MgO",
    "MNO(WT%)": "MnO",
    "K2O(WT%)": "K2O",
    "NA2O(WT%)": "Na2O",
    "P2O5(WT%)": "P2O5",
    "RB(PPM)": "Rb",
    "SR(PPM)": "Sr",
    "Y(PPM)": "Y",
    "ZR(PPM)": "Zr",
    "NB(PPM)": "Nb",
    "LA(PPM)": "La",
    "CE(PPM)": "Ce",
    "ND(PPM)": "Nd",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def normalize_name(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_excel(SOURCE_XLS, sheet_name="Table S1", header=None)
    row_by_label = {
        str(raw.iloc[row, 0]).strip(): row
        for row in range(len(raw))
        if pd.notna(raw.iloc[row, 0])
    }

    rows: list[dict[str, object]] = []
    for column in range(1, 13):
        sample = str(raw.iloc[1, column]).strip()
        record: dict[str, object] = {
            "sample_name": sample,
            "record_id": f"mdpi:10.3390/min9110659:{sample}",
            "material": "whole-rock basalt",
            "source_area": "South Mid-Atlantic Ridge, 18.0-20.6 S",
            "province_proxy": "South Mid-Atlantic Ridge 18-20.6 S",
            "mechanism_cohort": "SPREADING_CENTER",
            "actual_label": "MORB",
            "external_source": "Zhong et al. 2019 Table S1",
            "publication_group": "10.3390/min9110659",
            "publication_doi": "10.3390/min9110659",
            "source_dataset_doi": "10.3390/min9110659",
            "source_license": "CC BY 4.0",
            "provenance_tier": "publisher_primary_publication_linked",
        }
        for feature, label in ROW_MAP.items():
            record[feature] = pd.to_numeric(
                raw.iloc[row_by_label[label], column], errors="coerce"
            )
        fe2o3t = pd.to_numeric(
            raw.iloc[row_by_label["Fe2O3T"], column], errors="coerce"
        )
        source_feot = pd.to_numeric(
            raw.iloc[row_by_label["FeOT"], column], errors="coerce"
        )
        record["FE2O3T_REPORTED(WT%)"] = fe2o3t
        record["FEOT_SOURCE_TABLE(WT%)"] = source_feot
        record["FEOT(WT%)"] = 0.8998 * fe2o3t
        rows.append(record)

    data = pd.DataFrame(rows)
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
    data["normalized_sample_name"] = data["sample_name"].map(normalize_name)
    data["petdb_exact_name_collision"] = data["normalized_sample_name"].isin(
        petdb_names
    )
    data["major_complete"] = data[MAJORS].notna().all(axis=1)
    data["major_total_calc_feot"] = data[MAJORS].sum(
        axis=1, min_count=len(MAJORS)
    )
    data["observed_feature_n"] = data[FEATURES].notna().sum(axis=1)
    data["model_ready_common18"] = (
        data["major_complete"]
        & data["SIO2(WT%)"].between(40, 55)
        & data["major_total_calc_feot"].between(90, 105)
        & data["observed_feature_n"].eq(len(FEATURES))
        & data[FEATURES].gt(0).all(axis=1)
    )
    data["strict_external_candidate"] = data["model_ready_common18"]
    data.to_parquet(
        PROCESSED / "smar_2019_morb_external_v0_1.parquet", index=False
    )

    citations = "\n".join(petdb["citations"].dropna().astype(str)).upper()
    location_text = petdb["location_root"].fillna("").astype(str)
    audit = {
        "source_article": "https://www.mdpi.com/2075-163X/9/11/659",
        "source_supplement": (
            "https://mdpi-res.com/d_attachment/minerals/minerals-09-00659/"
            "article_deploy/minerals-09-00659-s001.zip"
        ),
        "source_zip": str(SOURCE_ZIP.relative_to(ROOT)),
        "source_xls": str(SOURCE_XLS.relative_to(ROOT)),
        "zip_bytes": SOURCE_ZIP.stat().st_size,
        "zip_sha256": digest(SOURCE_ZIP),
        "xls_bytes": SOURCE_XLS.stat().st_size,
        "xls_sha256": digest(SOURCE_XLS),
        "publication_doi": "10.3390/min9110659",
        "license": "CC BY 4.0",
        "raw_whole_rock_samples": len(data),
        "strict_model_ready_common18": int(data["model_ready_common18"].sum()),
        "feature_nonmissing": data[FEATURES].notna().sum().to_dict(),
        "publication_detected_in_petdb": any(
            key in citations
            for key in [
                "10.3390/MIN9110659",
                "GEOCHEMISTRY AND MINERALOGY OF BASALTS FROM THE SOUTH MID-ATLANTIC RIDGE",
            ]
        ),
        "exact_sample_name_collision_n": int(
            data["petdb_exact_name_collision"].sum()
        ),
        "petdb_south_mid_atlantic_location_rows": int(
            location_text.str.contains(
                r"SOUTH.*MID.?ATLANTIC|MID.?ATLANTIC.*SOUTH",
                case=False,
                regex=True,
            ).sum()
        ),
        "major_total_range": [
            float(data["major_total_calc_feot"].min()),
            float(data["major_total_calc_feot"].max()),
        ],
        "source_vs_recalculated_feot_max_abs_difference": float(
            (
                data["FEOT_SOURCE_TABLE(WT%)"] - data["FEOT(WT%)"]
            ).abs().max()
        ),
        "selection_policy": (
            "Retain all 12 author-reported whole-rock basalt samples; exclude reference "
            "materials and blanks. Require SiO2 40-55 wt%, positive complete common18, "
            "and harmonized major total 90-105 wt%."
        ),
        "fe_harmonization": (
            "Use FEOT = 0.8998 * author-reported Fe2O3T. The supplementary table also "
            "provides the same derived FeOT values, enabling a direct conversion check."
        ),
        "independence_scope": (
            "One publication-linked province proxy comprising 12 samples dredged at four "
            "locations. Publication and exact sample-name overlap are audited against the "
            "PetDB training table; broad geographic overlap alone is not sample leakage."
        ),
    }
    (OUT / "smar_morb_external_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "candidate": "South Mid-Atlantic Ridge 18-20.6 S Table S1",
                "target": "MORB",
                "raw_whole_rock_rows": len(data),
                "strict_model_ready_rows": int(
                    data["model_ready_common18"].sum()
                ),
                "publication_overlap_petdb": audit[
                    "publication_detected_in_petdb"
                ],
                "exact_name_collision_n": audit[
                    "exact_sample_name_collision_n"
                ],
                "status": "strict_external_candidate",
            }
        ]
    ).to_csv(OUT / "smar_morb_external_audit.csv", index=False)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
