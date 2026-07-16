from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "figshare_25295671"
INTERIM = ROOT / "data" / "interim" / "figshare_25295671_table_s1.csv"
PROCESSED = ROOT / "data" / "processed" / "petdb_morb_v0_1.parquet"
OUT = ROOT / "reports" / "data_quality"

FEATURE_MAP = {
    "SIO2(WT%)": "SiO2",
    "TIO2(WT%)": "TiO2",
    "AL2O3(WT%)": "Al2O3",
    "FEOT(WT%)": "FeOt",
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
MAJOR10 = list(FEATURE_MAP)[:10]


def normalize_name(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def digest(path: Path, algorithm: str = "sha256") -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((RAW / "metadata.json").read_text(encoding="utf-8"))
    raw = pd.read_csv(INTERIM)
    candidate = raw.rename(columns={source: target for target, source in FEATURE_MAP.items()})
    candidate["normalized_sample_name"] = candidate["Sample"].map(normalize_name)

    petdb = pd.read_parquet(PROCESSED)
    petdb_names = {
        normalize_name(part)
        for value in petdb["sample_names"].dropna()
        for part in str(value).split(" || ")
        if normalize_name(part)
    }
    petdb_citations = "\n".join(petdb["citations"].dropna().astype(str)).upper()
    candidate["petdb_exact_normalized_name_overlap"] = candidate[
        "normalized_sample_name"
    ].isin(petdb_names)
    candidate["observed_feature_n"] = candidate[list(FEATURE_MAP)].notna().sum(axis=1)
    candidate["major_total_calc"] = candidate[MAJOR10].sum(
        axis=1, min_count=len(MAJOR10)
    )
    candidate["flag_complete_major10"] = candidate[MAJOR10].notna().all(axis=1)
    candidate["flag_silica_40_55"] = candidate["SIO2(WT%)"].between(40, 55)
    candidate["flag_total_85_105"] = candidate["major_total_calc"].between(85, 105)
    candidate["model_ready_common18"] = (
        candidate["flag_complete_major10"]
        & candidate["flag_silica_40_55"]
        & candidate["flag_total_85_105"]
        & candidate["observed_feature_n"].ge(17)
    )
    candidate["provenance_tier"] = candidate["source_region"].map(
        {
            "East Pacific Rise": "secondary_literature_exclude_strict",
            "Southwest Indian Ridge": "repository_original_provisional",
        }
    )
    candidate["strict_external_candidate"] = (
        candidate["source_region"].eq("Southwest Indian Ridge")
        & candidate["model_ready_common18"]
    )

    keep = [
        "source_region",
        "source_excel_row",
        "Sample",
        "normalized_sample_name",
        *FEATURE_MAP,
        "LOI",
        "Total",
        "major_total_calc",
        "observed_feature_n",
        "flag_complete_major10",
        "flag_silica_40_55",
        "flag_total_85_105",
        "model_ready_common18",
        "petdb_exact_normalized_name_overlap",
        "provenance_tier",
        "strict_external_candidate",
    ]
    candidate[keep].to_parquet(
        ROOT / "data" / "processed" / "figshare_morb_external_v0_1.parquet",
        index=False,
    )

    summary_rows = []
    for region, frame in candidate.groupby("source_region", sort=True):
        summary_rows.append(
            {
                "source_region": region,
                "n": len(frame),
                "model_ready_common18": int(frame["model_ready_common18"].sum()),
                "strict_external_candidate": int(frame["strict_external_candidate"].sum()),
                "petdb_exact_normalized_name_overlap": int(
                    frame["petdb_exact_normalized_name_overlap"].sum()
                ),
                "complete_major10": int(frame["flag_complete_major10"].sum()),
                "median_observed_feature_n": float(frame["observed_feature_n"].median()),
                "major_total_min": float(frame["major_total_calc"].min()),
                "major_total_max": float(frame["major_total_calc"].max()),
                "loi_abs_max": float(frame["LOI"].abs().max()),
                "provenance_tier": frame["provenance_tier"].iloc[0],
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "figshare_25295671_morb_candidate_audit.csv", index=False)

    detail = {
        "dataset_doi": metadata["doi"],
        "dataset_url": metadata["figshare_url"],
        "license": metadata["license"]["name"],
        "raw_sha256": digest(RAW / "Supplementary_Tables_S1_to_S6.xlsx"),
        "later_mendeley_dataset": {
            "doi": "10.17632/fntfnf92tg.1",
            "title": (
                "Cadmium isotope compositions of basalts from the East Pacific Rise "
                "and Southwest Indian Ridge: Implications for magma differentiation "
                "and mantle heterogeneity"
            ),
            "relationship": (
                "A later repository record supplies the study title, but no journal "
                "article DOI was located as of the audit date."
            ),
        },
        "feature_map": FEATURE_MAP,
        "petdb_publication_title_detected": (
            "CADMIUM ISOTOPE COMPOSITIONS OF BASALTS FROM THE EAST PACIFIC RISE"
            in petdb_citations
        ),
        "epr_policy": (
            "Exclude from strict external evidence because workbook Table S1 states "
            "that EPR data are from Li et al. (2019); retain as sensitivity only."
        ),
        "swir_policy": (
            "Treat as repository-original provisional external data because no secondary "
            "source note is attached and no exact normalized sample-name overlap was found. "
            "Do not describe it as publication-independent until a journal DOI is linked."
        ),
        "name_overlap_scope": (
            "Exact normalized sample-name comparison only; it cannot exclude renamed or "
            "aliased samples."
        ),
        "summary": summary.to_dict(orient="records"),
    }
    (OUT / "figshare_25295671_morb_candidate_audit.json").write_text(
        json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
