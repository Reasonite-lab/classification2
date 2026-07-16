from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from profile_petdb_morb import normalize_sample_name, normalize_string_series


ROOT = Path(__file__).resolve().parents[1]
EXPORT_ID = "earthchem_export_jul14_26_14620"
RAW_DIR = ROOT / "data" / "raw" / "petdb_morb" / EXPORT_ID
MAIN_CSV = RAW_DIR / f"{EXPORT_ID}.csv"
SAMPLE_CSV = RAW_DIR / f"{EXPORT_ID}_sample_metadata.csv"
STUDY_CONFIG = ROOT / "config" / "study_config.yaml"
OUT_DATA = ROOT / "data" / "processed" / "petdb_morb_v0_1.parquet"
OUT_DATA_CSV = ROOT / "data" / "processed" / "petdb_morb_v0_1.csv.gz"
OUT_QC = ROOT / "reports" / "data_quality"
OUT_REFS = ROOT / "references" / "petdb_morb_bibliography.csv"
DUPLICATE_INDEX_CACHE = OUT_QC / "petdb_morb_exact_duplicate_indices.csv"


DIRECT_FEATURE_MAP = {
    "SiO2 (wt%)": "SIO2(WT%)",
    "TiO2 (wt%)": "TIO2(WT%)",
    "Al2O3 (wt%)": "AL2O3(WT%)",
    "CaO (wt%)": "CAO(WT%)",
    "MgO (wt%)": "MGO(WT%)",
    "MnO (wt%)": "MNO(WT%)",
    "K2O (wt%)": "K2O(WT%)",
    "Na2O (wt%)": "NA2O(WT%)",
    "P2O5 (wt%)": "P2O5(WT%)",
    "V (ppm)": "V(PPM)",
    "Cr (ppm)": "CR(PPM)",
    "Ni (ppm)": "NI(PPM)",
    "Rb (ppm)": "RB(PPM)",
    "Sr (ppm)": "SR(PPM)",
    "Y (ppm)": "Y(PPM)",
    "Zr (ppm)": "ZR(PPM)",
    "Nb (ppm)": "NB(PPM)",
    "La (ppm)": "LA(PPM)",
    "Ce (ppm)": "CE(PPM)",
    "Nd (ppm)": "ND(PPM)",
    "Sm (ppm)": "SM(PPM)",
    "Eu (ppm)": "EU(PPM)",
    "Yb (ppm)": "YB(PPM)",
    "Lu (ppm)": "LU(PPM)",
    "Hf (ppm)": "HF(PPM)",
    "Ta (ppm)": "TA(PPM)",
    "Th (ppm)": "TH(PPM)",
    "U (ppm)": "U(PPM)",
}

IRON_RAW_COLUMNS = [
    "FeOT (wt%)",
    "Fe2O3T (wt%)",
    "FeO (wt%)",
    "Fe2O3 (wt%)",
]

MAIN_METADATA_COLUMNS = [
    "Sample Name",
    "Sample URL",
    "Citation",
    "Citation URL",
    "Data Compilation",
    "Analysis Type",
    "Analyzed Material",
    "Latitude",
    "Longitude",
    "Sample Type",
    "Sample SubType",
    "Analytical batch variables; Method Code; Method ID",
    "Analysis Comment",
    "Calc Avg",
    "Number Of Replicates",
]

SAMPLE_METADATA_COLUMNS = [
    "Analyzed Sample Name",
    "Latitude",
    "Longitude",
    "Geographic Features",
    "Tectonic Setting",
    "Rock Classification",
    "Rock Subtype/Descriptor",
    "Rock Type",
    "Rock Texture",
    "Station/Site",
    "Expedition",
    "Sample Alteration",
    "Sample Alteration Type",
    "Geologic Age",
    "Geologic Age Max (Ma)",
    "Geologic Age Min (Ma)",
]

MAJOR_FEATURES = [
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
]

SEVERE_ALTERATION_TERMS = {
    "EXTENSIVELY ALTERED",
    "ALMOST COMPLETELY ALTERED",
    "TOTALLY ALTERED",
}


def joined_unique(values: pd.Series) -> str:
    clean = normalize_string_series(values).dropna().astype(str).unique().tolist()
    return " || ".join(sorted(clean))


def sanitize_feature_name(feature: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", feature.upper()).strip("_")


def parse_coordinate(series: pd.Series, positive: str, negative: str) -> pd.Series:
    extracted = series.astype("string").str.extract(
        rf"([+-]?[0-9]+(?:\.[0-9]+)?).*?([{positive}{negative}])",
        expand=True,
    )
    magnitude = pd.to_numeric(extracted[0], errors="coerce").abs()
    direction = extracted[1]
    signed = magnitude.where(direction.eq(positive), -magnitude)
    fallback = pd.to_numeric(normalize_string_series(series), errors="coerce")
    return signed.fillna(fallback)


def duplicate_raw_row_indices(path: Path) -> list[int]:
    seen: set[int] = set()
    duplicate_indices: list[int] = []
    for chunk in pd.read_csv(path, dtype=str, chunksize=1000, low_memory=False):
        normalized = chunk.fillna("").astype(str)
        row_hashes = pd.util.hash_pandas_object(normalized, index=False).astype("uint64")
        for index, value in zip(chunk.index, row_hashes.tolist(), strict=True):
            value_int = int(value)
            if value_int in seen:
                duplicate_indices.append(int(index))
            else:
                seen.add(value_int)
    return duplicate_indices


def extract_citation_id(url: str) -> str:
    return str(url).rstrip("/").rsplit("/", 1)[-1]


def build_overlap_components(citation_sets: pd.Series) -> pd.Series:
    parent: dict[str, str] = {}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            low, high = sorted((left_root, right_root))
            parent[high] = low

    parsed = [
        [] if value == "<MISSING>" else [part for part in str(value).split("|") if part]
        for value in citation_sets
    ]
    for citation_ids in parsed:
        for citation_id in citation_ids:
            find(citation_id)
        for citation_id in citation_ids[1:]:
            union(citation_ids[0], citation_id)
    return pd.Series(
        [find(ids[0]) if ids else "<MISSING>" for ids in parsed],
        index=citation_sets.index,
        dtype="string",
    )


def extract_location_group(geographic_features: str) -> str:
    if not geographic_features:
        return "<MISSING>"
    ridge_names = re.findall(
        r"SPREADING_CENTER\s*\|\s*([^,\"|]+)",
        geographic_features,
        flags=re.IGNORECASE,
    )
    ridge_names = sorted(
        {re.sub(r"\s+", " ", name).strip().upper() for name in ridge_names if name.strip()}
    )
    if ridge_names:
        return " | ".join(ridge_names)
    # EarthChem may serialize the same set of geographic descriptors in a
    # different order across records.  Sort comma/record-separated descriptors
    # so semantically identical locations cannot leak across grouped folds.
    descriptors = {
        re.sub(r"\s+", " ", part).strip().strip('"').upper()
        for part in re.split(r"\s*(?:\|\||,)\s*", geographic_features)
        if part.strip().strip('"')
    }
    return ", ".join(sorted(descriptors)) or "<MISSING>"


def alteration_severity(value: str) -> str:
    tags = {part.strip().upper() for part in value.split("||") if part.strip()}
    if not tags:
        return "UNKNOWN"
    if tags & SEVERE_ALTERATION_TERMS:
        return "SEVERE_PRESENT"
    if "MODERATELY ALTERED" in tags or "M" in tags:
        return "MODERATE_PRESENT"
    if "SLIGHTLY ALTERED" in tags:
        return "SLIGHT_PRESENT"
    if tags <= {"FRESH", "F"}:
        return "FRESH"
    return "OTHER_OR_MIXED"


def main() -> None:
    OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    OUT_QC.mkdir(parents=True, exist_ok=True)
    OUT_REFS.parent.mkdir(parents=True, exist_ok=True)

    study_cfg = yaml.safe_load(STUDY_CONFIG.read_text(encoding="utf-8"))
    broad_features = study_cfg["feature_sets"]["broad_v1"]
    immobile_features = study_cfg["feature_sets"]["immobile_v1"]
    model_features = list(dict.fromkeys(broad_features + immobile_features))
    expected_model_features = set(DIRECT_FEATURE_MAP.values()) | {"FEOT(WT%)"}
    if not set(model_features).issubset(expected_model_features):
        missing = sorted(set(model_features) - expected_model_features)
        raise ValueError(f"PetDB feature map is missing study features: {missing}")

    if DUPLICATE_INDEX_CACHE.exists():
        duplicate_indices = (
            pd.read_csv(DUPLICATE_INDEX_CACHE)["row_index"].astype(int).tolist()
        )
    else:
        duplicate_indices = duplicate_raw_row_indices(MAIN_CSV)
        pd.DataFrame({"row_index": duplicate_indices}).to_csv(
            DUPLICATE_INDEX_CACHE, index=False
        )
    if len(duplicate_indices) != 32:
        raise ValueError(f"Expected 32 exact duplicate rows, found {len(duplicate_indices)}")

    main_usecols = MAIN_METADATA_COLUMNS + list(DIRECT_FEATURE_MAP) + IRON_RAW_COLUMNS
    analyses = pd.read_csv(MAIN_CSV, usecols=main_usecols, dtype=str, low_memory=False)
    sample_metadata = pd.read_csv(
        SAMPLE_CSV, usecols=SAMPLE_METADATA_COLUMNS, dtype=str, low_memory=False
    )
    if len(analyses) != len(sample_metadata) or len(analyses) != 22680:
        raise ValueError("PetDB main and sample metadata tables do not have expected row counts")

    main_names = analyses["Sample Name"].map(normalize_sample_name)
    metadata_names = sample_metadata["Analyzed Sample Name"].map(normalize_sample_name)
    name_match = (main_names == metadata_names) | (
        main_names.isna() & metadata_names.isna()
    )
    coordinate_match = (
        normalize_string_series(analyses["Latitude"])
        == normalize_string_series(sample_metadata["Latitude"])
    ) & (
        normalize_string_series(analyses["Longitude"])
        == normalize_string_series(sample_metadata["Longitude"])
    )
    if not name_match.all() or not coordinate_match.all():
        raise ValueError("Row linkage failed normalized-name or coordinate verification")

    analyses["normalized_sample_name"] = main_names
    analyses["sample_url"] = normalize_string_series(analyses["Sample URL"])
    if analyses["sample_url"].isna().any():
        raise ValueError("Every PetDB analysis row must have a Sample URL")
    analyses["latitude_decimal"] = parse_coordinate(analyses["Latitude"], "N", "S")
    analyses["longitude_decimal"] = parse_coordinate(analyses["Longitude"], "E", "W")
    for column in SAMPLE_METADATA_COLUMNS:
        analyses[f"meta::{column}"] = sample_metadata[column].to_numpy()

    analyses = analyses.drop(index=duplicate_indices).reset_index(drop=True)

    numeric_columns: dict[str, pd.Series] = {}
    nonpositive_replacements: dict[str, int] = {}
    for raw_column, standard_column in DIRECT_FEATURE_MAP.items():
        values = pd.to_numeric(normalize_string_series(analyses[raw_column]), errors="coerce")
        if standard_column.endswith("(PPM)"):
            invalid = values.notna() & (values <= 0)
            nonpositive_replacements[standard_column] = int(invalid.sum())
            values = values.mask(invalid)
        else:
            invalid = values.notna() & (values < 0)
            nonpositive_replacements[standard_column] = int(invalid.sum())
            values = values.mask(invalid)
        numeric_columns[standard_column] = values
        analyses[standard_column] = values

    feot_reported = pd.to_numeric(
        normalize_string_series(analyses["FeOT (wt%)"]), errors="coerce"
    ).mask(lambda x: x < 0)
    fe2o3t = pd.to_numeric(
        normalize_string_series(analyses["Fe2O3T (wt%)"]), errors="coerce"
    ).mask(lambda x: x < 0)
    feo = pd.to_numeric(
        normalize_string_series(analyses["FeO (wt%)"]), errors="coerce"
    ).mask(lambda x: x < 0)
    fe2o3 = pd.to_numeric(
        normalize_string_series(analyses["Fe2O3 (wt%)"]), errors="coerce"
    ).mask(lambda x: x < 0)
    component_available = feo.notna() | fe2o3.notna()
    component_feo_eq = (feo.fillna(0) + 0.8998 * fe2o3.fillna(0)).where(
        component_available
    )
    analyses["FEOT_REPORTED(WT%)"] = feot_reported
    analyses["FE2O3T_REPORTED(WT%)"] = fe2o3t
    analyses["FEO_REPORTED(WT%)"] = feo
    analyses["FE2O3_REPORTED(WT%)"] = fe2o3
    analyses["FEOT(WT%)"] = (
        feot_reported.combine_first(0.8998 * fe2o3t).combine_first(component_feo_eq)
    )
    analyses["iron_value_source"] = np.select(
        [feot_reported.notna(), fe2o3t.notna(), component_available],
        ["FEOT_REPORTED", "FE2O3T_CONVERTED", "FEO_PLUS_FE2O3_CONVERTED"],
        default="MISSING",
    )

    group = analyses.groupby("sample_url", sort=True, dropna=False)
    samples = group.size().rename("analysis_row_n").to_frame()
    for feature in model_features:
        samples[feature] = group[feature].median()
        count_column = f"MEASUREMENT_N__{sanitize_feature_name(feature)}"
        samples[count_column] = group[feature].count().astype("int16")

    metadata_aggregations = {
        "sample_names": "normalized_sample_name",
        "citations": "Citation",
        "citation_urls": "Citation URL",
        "method_ids": "Analytical batch variables; Method Code; Method ID",
        "analysis_comments": "Analysis Comment",
        "calc_avg_values": "Calc Avg",
        "tectonic_settings": "meta::Tectonic Setting",
        "geographic_features": "meta::Geographic Features",
        "rock_classifications": "meta::Rock Classification",
        "rock_subtypes": "meta::Rock Subtype/Descriptor",
        "rock_types": "meta::Rock Type",
        "rock_textures": "meta::Rock Texture",
        "station_sites": "meta::Station/Site",
        "expeditions": "meta::Expedition",
        "alteration_tags": "meta::Sample Alteration",
        "alteration_types": "meta::Sample Alteration Type",
        "geologic_ages": "meta::Geologic Age",
        "iron_value_sources": "iron_value_source",
    }
    for output_column, source_column in metadata_aggregations.items():
        samples[output_column] = group[source_column].apply(joined_unique)

    samples["latitude"] = group["latitude_decimal"].median()
    samples["longitude"] = group["longitude_decimal"].median()
    samples["reported_replicates_max"] = group["Number Of Replicates"].apply(
        lambda values: pd.to_numeric(values, errors="coerce").max()
    )

    samples["citation_ids"] = samples["citation_urls"].map(
        lambda value: "|".join(
            sorted(
                {
                    extract_citation_id(url.strip())
                    for url in value.split("||")
                    if url.strip()
                }
            )
        )
    )
    samples["citation_set"] = samples["citation_ids"].replace("", "<MISSING>")
    samples["citation_overlap_component"] = build_overlap_components(
        samples["citation_set"]
    )
    samples["location_root"] = samples["geographic_features"].map(
        extract_location_group
    )
    samples["alteration_severity"] = samples["alteration_tags"].map(
        alteration_severity
    )
    samples["flag_no_severe_alteration"] = ~samples[
        "alteration_severity"
    ].eq("SEVERE_PRESENT")

    samples["flag_exclusive_spreading_center"] = samples[
        "tectonic_settings"
    ].eq("SPREADING CENTER")
    samples["flag_silica_40_55"] = samples["SIO2(WT%)"].between(
        40.0, 55.0, inclusive="both"
    )
    samples["major_complete"] = samples[MAJOR_FEATURES].notna().all(axis=1)
    samples["major_total_calc"] = samples[MAJOR_FEATURES].sum(axis=1, min_count=10)
    samples["major_total_plausible"] = samples["major_complete"] & samples[
        "major_total_calc"
    ].between(85.0, 105.0, inclusive="both")
    broad_major_ranges = {
        "TIO2(WT%)": (0.0, 8.0),
        "AL2O3(WT%)": (5.0, 30.0),
        "FEOT(WT%)": (2.0, 25.0),
        "CAO(WT%)": (2.0, 20.0),
        "MGO(WT%)": (0.5, 40.0),
        "MNO(WT%)": (0.0, 1.0),
        "K2O(WT%)": (0.0, 10.0),
        "NA2O(WT%)": (0.0, 10.0),
        "P2O5(WT%)": (0.0, 5.0),
    }
    range_flags = pd.DataFrame(
        {
            feature: samples[feature].between(low, high, inclusive="both")
            for feature, (low, high) in broad_major_ranges.items()
        },
        index=samples.index,
    )
    samples["major_component_range_plausible"] = range_flags.all(axis=1)
    samples["broad_observed_n"] = samples[broad_features].notna().sum(axis=1)
    samples["immobile_observed_n"] = samples[immobile_features].notna().sum(axis=1)

    samples["strict_candidate"] = (
        samples["flag_exclusive_spreading_center"]
        & samples["flag_silica_40_55"]
        & samples["major_total_plausible"]
        & samples["citation_set"].ne("<MISSING>")
    )
    samples["model_ready_broad"] = samples["strict_candidate"] & samples[
        "broad_observed_n"
    ].ge(12)
    samples["model_ready_immobile"] = samples["strict_candidate"] & samples[
        "immobile_observed_n"
    ].ge(10)
    samples["model_ready_broad_no_severe_alteration"] = samples[
        "model_ready_broad"
    ] & samples["flag_no_severe_alteration"]
    samples["model_ready_immobile_no_severe_alteration"] = samples[
        "model_ready_immobile"
    ] & samples["flag_no_severe_alteration"]
    samples["model_ready_broad_major_range_sensitivity"] = samples[
        "model_ready_broad"
    ] & samples["major_component_range_plausible"]
    samples["model_ready_immobile_major_range_sensitivity"] = samples[
        "model_ready_immobile"
    ] & samples["major_component_range_plausible"]

    name_to_settings: dict[str, set[str]] = defaultdict(set)
    all_meta_names = sample_metadata["Analyzed Sample Name"].map(normalize_sample_name)
    all_meta_settings = normalize_string_series(sample_metadata["Tectonic Setting"])
    for name, setting in zip(all_meta_names, all_meta_settings, strict=True):
        if name is not None and not pd.isna(name) and not pd.isna(setting):
            name_to_settings[str(name)].add(str(setting))
    url_name_sets = group["normalized_sample_name"].apply(
        lambda values: {
            str(value) for value in values if value is not None and not pd.isna(value)
        }
    )

    def name_linkage_status(names: set[str]) -> str:
        settings: set[str] = set()
        for name in names:
            settings.update(name_to_settings.get(name, set()))
        if settings == {"SPREADING CENTER"}:
            return "EXCLUSIVE"
        if settings:
            return "MIXED"
        return "UNKNOWN"

    samples["name_only_linkage_status"] = url_name_sets.map(name_linkage_status)
    samples["flag_name_only_exclusive_sensitivity"] = samples[
        "name_only_linkage_status"
    ].eq("EXCLUSIVE")

    samples["multi_measure_feature_n"] = 0
    samples["discordant_feature_n"] = 0
    for feature in model_features:
        counts = group[feature].count()
        minimum = group[feature].min()
        maximum = group[feature].max()
        median = samples[feature].abs().replace(0, np.nan)
        spread = maximum - minimum
        relative_spread = spread / median
        samples["multi_measure_feature_n"] += counts.ge(2).astype("int16")
        if feature.endswith("(PPM)"):
            discordant = counts.ge(2) & spread.gt(1.0) & relative_spread.gt(0.20)
        else:
            discordant = counts.ge(2) & spread.gt(0.1) & relative_spread.gt(0.10)
        samples["discordant_feature_n"] += discordant.astype("int16")

    samples["label_primary4"] = "MORB"
    samples["label_fine"] = "SPREADING_CENTER"
    samples["label_confidence"] = np.where(
        samples["flag_exclusive_spreading_center"], "HIGH", "EXCLUDE_MIXED"
    )
    samples["data_source"] = "PetDB2"
    samples["source_export_id"] = EXPORT_ID
    samples["record_id"] = "PETDB:" + samples.index.str.rstrip("/").str.rsplit("/").str[-1]
    samples = samples.reset_index().rename(columns={"sample_url": "SAMPLE_URL"})
    if samples["record_id"].duplicated().any():
        raise ValueError("PetDB record_id must be unique after sample aggregation")

    samples.to_parquet(OUT_DATA, index=False)
    samples.to_csv(OUT_DATA_CSV, index=False, compression="gzip")

    cohort_masks = {
        "all_sample_urls": pd.Series(True, index=samples.index),
        "exclusive_spreading_center": samples["flag_exclusive_spreading_center"],
        "strict_candidate": samples["strict_candidate"],
        "model_ready_broad": samples["model_ready_broad"],
        "model_ready_immobile": samples["model_ready_immobile"],
    }
    coverage_rows: list[dict[str, Any]] = []
    for cohort, mask in cohort_masks.items():
        cohort_data = samples.loc[mask]
        for feature in model_features:
            values = cohort_data[feature]
            coverage_rows.append(
                {
                    "cohort": cohort,
                    "feature": feature,
                    "samples": int(len(cohort_data)),
                    "non_null": int(values.notna().sum()),
                    "non_null_pct": round(100 * values.notna().mean(), 4)
                    if len(cohort_data)
                    else None,
                    "min": float(values.min()) if values.notna().any() else None,
                    "median": float(values.median()) if values.notna().any() else None,
                    "max": float(values.max()) if values.notna().any() else None,
                }
            )
    pd.DataFrame(coverage_rows).to_csv(
        OUT_QC / "petdb_morb_sample_feature_coverage.csv", index=False
    )

    qc_flow = pd.DataFrame(
        [
            {"stage": "raw_analysis_rows", "n": 22680},
            {"stage": "after_exact_duplicate_removal", "n": len(analyses)},
            {"stage": "unique_sample_urls", "n": len(samples)},
            {
                "stage": "exclusive_spreading_center_urls",
                "n": int(samples["flag_exclusive_spreading_center"].sum()),
            },
            {
                "stage": "exclusive_with_silica_40_55",
                "n": int(
                    (
                        samples["flag_exclusive_spreading_center"]
                        & samples["flag_silica_40_55"]
                    ).sum()
                ),
            },
            {"stage": "strict_candidate", "n": int(samples["strict_candidate"].sum())},
            {"stage": "model_ready_broad", "n": int(samples["model_ready_broad"].sum())},
            {
                "stage": "model_ready_immobile",
                "n": int(samples["model_ready_immobile"].sum()),
            },
        ]
    )
    qc_flow.to_csv(OUT_QC / "petdb_morb_qc_flow.csv", index=False)

    bibliography = (
        analyses[["Citation URL", "Citation"]]
        .dropna(how="all")
        .drop_duplicates()
        .sort_values(["Citation URL", "Citation"], na_position="last")
        .reset_index(drop=True)
    )
    bibliography["citation_id"] = normalize_string_series(
        bibliography["Citation URL"]
    ).map(lambda value: extract_citation_id(value) if not pd.isna(value) else None)
    bibliography.to_csv(OUT_REFS, index=False)

    profile = {
        "dataset": "PetDB 2.0 MORB sample-level cohort",
        "version": "0.1",
        "source_export_id": EXPORT_ID,
        "raw_analysis_rows": 22680,
        "exact_duplicate_rows_removed": len(duplicate_indices),
        "analysis_rows_after_exact_deduplication": len(analyses),
        "sample_urls": len(samples),
        "row_linkage": {
            "normalized_name_matches": int(name_match.sum()),
            "coordinate_matches": int(coordinate_match.sum()),
            "method": "row alignment accepted only after 100% normalized-name and coordinate agreement",
        },
        "tectonic_linkage": {
            "exclusive_spreading_center": int(
                samples["flag_exclusive_spreading_center"].sum()
            ),
            "mixed_or_secondary": int(
                (~samples["flag_exclusive_spreading_center"]).sum()
            ),
            "name_only_sensitivity_exclusive": int(
                samples["flag_name_only_exclusive_sensitivity"].sum()
            ),
            "primary_method": "verified row linkage",
            "sensitivity_method": "normalized-name-only mapping",
        },
        "cohorts": {
            "strict_candidate": int(samples["strict_candidate"].sum()),
            "model_ready_broad": int(samples["model_ready_broad"].sum()),
            "model_ready_immobile": int(samples["model_ready_immobile"].sum()),
            "model_ready_broad_no_severe_alteration": int(
                samples["model_ready_broad_no_severe_alteration"].sum()
            ),
            "model_ready_immobile_no_severe_alteration": int(
                samples["model_ready_immobile_no_severe_alteration"].sum()
            ),
            "model_ready_broad_major_range_sensitivity": int(
                samples["model_ready_broad_major_range_sensitivity"].sum()
            ),
            "model_ready_immobile_major_range_sensitivity": int(
                samples["model_ready_immobile_major_range_sensitivity"].sum()
            ),
        },
        "iron_harmonization": {
            "target": "total iron as FeO equivalent",
            "row_precedence": [
                "reported FeOT",
                "0.8998 * reported Fe2O3T",
                "reported FeO + 0.8998 * reported Fe2O3",
            ],
            "sample_aggregation": "median across non-duplicate analysis rows",
        },
        "replicate_policy": {
            "aggregation": "median by stable Sample URL",
            "discordance_major": "at least two values, >0.1 wt% absolute and >10% relative range",
            "discordance_trace": "at least two values, >1 ppm absolute and >20% relative range",
            "discordance_is_exclusion": False,
        },
        "major_component_range_sensitivity": {
            "ranges": {
                feature: [low, high]
                for feature, (low, high) in broad_major_ranges.items()
            },
            "is_primary_exclusion": False,
            "purpose": "Flag gross component outliers without changing the preregistered primary total-sum contract",
        },
        "nonpositive_replacements": nonpositive_replacements,
        "bibliography_rows": len(bibliography),
        "outputs": {
            "sample_data": str(OUT_DATA.relative_to(ROOT)),
            "sample_data_portable_csv": str(OUT_DATA_CSV.relative_to(ROOT)),
            "coverage": "reports/data_quality/petdb_morb_sample_feature_coverage.csv",
            "qc_flow": "reports/data_quality/petdb_morb_qc_flow.csv",
            "bibliography": str(OUT_REFS.relative_to(ROOT)),
        },
        "modeling_gate": {
            "status": "blocked_for_final_four_class_claims_but_not_for_processing",
            "reason": (
                "PetDB currently contributes only MORB while the processed GEOROC cohort "
                "contributes ARC/CIB/OIB; in a naive pooled four-class dataset, class and "
                "database source are perfectly confounded. Collect non-MORB PetDB classes "
                "or an additional independent MORB source before claiming source-robust "
                "four-class performance."
            ),
        },
    }
    (OUT_QC / "petdb_morb_processing_profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(profile, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
