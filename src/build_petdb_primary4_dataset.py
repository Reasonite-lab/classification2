from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from build_petdb_morb_dataset import (
    DIRECT_FEATURE_MAP,
    IRON_RAW_COLUMNS,
    MAIN_METADATA_COLUMNS,
    MAJOR_FEATURES,
    SAMPLE_METADATA_COLUMNS,
    alteration_severity,
    build_overlap_components,
    duplicate_raw_row_indices,
    extract_citation_id,
    extract_location_group,
    joined_unique,
    parse_coordinate,
    sanitize_feature_name,
)
from profile_petdb_morb import normalize_sample_name, normalize_string_series


ROOT = Path(__file__).resolve().parents[1]
EXPORT_CONFIG = ROOT / "config" / "petdb_primary4_exports.yaml"
STUDY_CONFIG = ROOT / "config" / "study_config.yaml"
OUT_PARQUET = ROOT / "data" / "processed" / "petdb_primary4_v0_1.parquet"
OUT_CSV = ROOT / "data" / "processed" / "petdb_primary4_v0_1.csv.gz"
OUT_QC = ROOT / "reports" / "data_quality"
OUT_REFS = ROOT / "references" / "petdb_primary4_bibliography.csv"


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def equal_with_missing(left: pd.Series, right: pd.Series) -> pd.Series:
    return (left == right) | (left.isna() & right.isna())


def load_export(
    spec: dict[str, Any],
    model_features: list[str],
    broad_features: list[str],
    immobile_features: list[str],
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    label = str(spec["label"])
    slug = str(spec["slug"])
    target_setting = str(spec["target_setting"])
    export_id = str(spec["export_id"])
    raw_root = ROOT / "data" / "raw" / slug
    archive = raw_root / f"{export_id}.zip"
    export_dir = raw_root / export_id
    main_csv = export_dir / f"{export_id}.csv"
    sample_csv = export_dir / f"{export_id}_sample_metadata.csv"

    actual_sha256 = file_hash(archive)
    if actual_sha256 != str(spec["sha256"]):
        raise ValueError(f"{label}: archive SHA256 mismatch")

    header = pd.read_csv(main_csv, nrows=0).columns.tolist()
    required_metadata = set(MAIN_METADATA_COLUMNS)
    missing_metadata = sorted(required_metadata - set(header))
    if missing_metadata:
        raise ValueError(f"{label}: missing required metadata columns {missing_metadata}")
    feature_sources = list(DIRECT_FEATURE_MAP) + IRON_RAW_COLUMNS
    available_features = [column for column in feature_sources if column in header]
    missing_feature_sources = sorted(set(feature_sources) - set(available_features))

    duplicate_cache = OUT_QC / f"{slug}_exact_duplicate_indices.csv"
    if duplicate_cache.exists():
        duplicate_indices = (
            pd.read_csv(duplicate_cache)["row_index"].astype(int).tolist()
        )
    else:
        duplicate_indices = duplicate_raw_row_indices(main_csv)
        pd.DataFrame({"row_index": duplicate_indices}).to_csv(
            duplicate_cache, index=False
        )

    analyses = pd.read_csv(
        main_csv,
        usecols=MAIN_METADATA_COLUMNS + available_features,
        dtype=str,
        low_memory=False,
    )
    for column in missing_feature_sources:
        analyses[column] = pd.NA
    sample_metadata = pd.read_csv(
        sample_csv, usecols=SAMPLE_METADATA_COLUMNS, dtype=str, low_memory=False
    )
    expected_rows = int(spec["delivered_rows"])
    if len(analyses) != expected_rows or len(sample_metadata) != expected_rows:
        raise ValueError(
            f"{label}: delivered row contract failed: "
            f"{len(analyses)} main / {len(sample_metadata)} metadata / {expected_rows} expected"
        )

    main_names = analyses["Sample Name"].map(normalize_sample_name)
    meta_names = sample_metadata["Analyzed Sample Name"].map(normalize_sample_name)
    name_match = equal_with_missing(main_names, meta_names)
    latitude_match = equal_with_missing(
        normalize_string_series(analyses["Latitude"]),
        normalize_string_series(sample_metadata["Latitude"]),
    )
    longitude_match = equal_with_missing(
        normalize_string_series(analyses["Longitude"]),
        normalize_string_series(sample_metadata["Longitude"]),
    )
    coordinate_match = latitude_match & longitude_match
    if not name_match.all() or not coordinate_match.all():
        raise ValueError(f"{label}: row linkage failed name or coordinate verification")

    analyses["normalized_sample_name"] = main_names
    analyses["sample_url"] = normalize_string_series(analyses["Sample URL"])
    if analyses["sample_url"].isna().any():
        raise ValueError(f"{label}: every row must have a Sample URL")
    analyses["latitude_decimal"] = parse_coordinate(analyses["Latitude"], "N", "S")
    analyses["longitude_decimal"] = parse_coordinate(analyses["Longitude"], "E", "W")
    for column in SAMPLE_METADATA_COLUMNS:
        analyses[f"meta::{column}"] = sample_metadata[column].to_numpy()
    analyses = analyses.drop(index=duplicate_indices).reset_index(drop=True)

    nonpositive_replacements: dict[str, int] = {}
    for raw_column, standard_column in DIRECT_FEATURE_MAP.items():
        values = pd.to_numeric(
            normalize_string_series(analyses[raw_column]), errors="coerce"
        )
        invalid = values.notna() & (
            values.le(0) if standard_column.endswith("(PPM)") else values.lt(0)
        )
        nonpositive_replacements[standard_column] = int(invalid.sum())
        analyses[standard_column] = values.mask(invalid)

    feot = pd.to_numeric(
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
    analyses["FEOT(WT%)"] = feot.combine_first(0.8998 * fe2o3t).combine_first(
        component_feo_eq
    )
    analyses["iron_value_source"] = np.select(
        [feot.notna(), fe2o3t.notna(), component_available],
        ["FEOT_REPORTED", "FE2O3T_CONVERTED", "FEO_PLUS_FE2O3_CONVERTED"],
        default="MISSING",
    )

    group = analyses.groupby("sample_url", sort=True, dropna=False)
    samples = group.size().rename("analysis_row_n").to_frame()
    for feature in model_features:
        samples[feature] = group[feature].median()
        samples[f"MEASUREMENT_N__{sanitize_feature_name(feature)}"] = (
            group[feature].count().astype("int16")
        )

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
    samples["location_root"] = samples["geographic_features"].map(
        extract_location_group
    )
    samples["alteration_severity"] = samples["alteration_tags"].map(
        alteration_severity
    )
    samples["flag_no_severe_alteration"] = ~samples["alteration_severity"].eq(
        "SEVERE_PRESENT"
    )

    samples["flag_exclusive_target_setting"] = samples["tectonic_settings"].eq(
        target_setting
    )
    samples["flag_silica_40_55"] = samples["SIO2(WT%)"].between(40, 55)
    samples["major_complete"] = samples[MAJOR_FEATURES].notna().all(axis=1)
    samples["major_total_calc"] = samples[MAJOR_FEATURES].sum(axis=1, min_count=10)
    samples["major_total_plausible"] = samples["major_complete"] & samples[
        "major_total_calc"
    ].between(85, 105)
    samples["broad_observed_n"] = samples[broad_features].notna().sum(axis=1)
    samples["immobile_observed_n"] = samples[immobile_features].notna().sum(axis=1)
    samples["strict_candidate"] = (
        samples["flag_exclusive_target_setting"]
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

    samples["multi_measure_feature_n"] = 0
    samples["discordant_feature_n"] = 0
    for feature in model_features:
        counts = group[feature].count()
        spread = group[feature].max() - group[feature].min()
        relative_spread = spread / samples[feature].abs().replace(0, np.nan)
        samples["multi_measure_feature_n"] += counts.ge(2).astype("int16")
        if feature.endswith("(PPM)"):
            discordant = counts.ge(2) & spread.gt(1) & relative_spread.gt(0.20)
        else:
            discordant = counts.ge(2) & spread.gt(0.1) & relative_spread.gt(0.10)
        samples["discordant_feature_n"] += discordant.astype("int16")

    samples["label_primary4"] = label
    samples["label_fine"] = target_setting.replace(" ", "_")
    samples["label_confidence"] = np.where(
        samples["flag_exclusive_target_setting"], "HIGH", "EXCLUDE_MIXED"
    )
    samples["data_source"] = "PetDB2"
    samples["source_export_id"] = export_id
    samples["source_export_slug"] = slug
    sample_token = samples.index.str.rstrip("/").str.rsplit("/").str[-1]
    samples["record_id"] = "PETDB:" + label + ":" + sample_token
    samples = samples.reset_index().rename(columns={"sample_url": "SAMPLE_URL"})

    bibliography = (
        analyses[["Citation URL", "Citation"]]
        .dropna(how="all")
        .drop_duplicates()
        .rename(columns={"Citation URL": "citation_url", "Citation": "citation"})
    )
    bibliography["citation_id"] = normalize_string_series(
        bibliography["citation_url"]
    ).map(lambda value: extract_citation_id(value) if not pd.isna(value) else None)
    bibliography["source_export_id"] = export_id
    bibliography["export_label"] = label

    profile = {
        "label": label,
        "target_setting": target_setting,
        "export_id": export_id,
        "api_count_at_submission": int(spec["api_count_at_submission"]),
        "raw_analysis_rows": expected_rows,
        "raw_columns": len(header),
        "missing_feature_source_columns": missing_feature_sources,
        "exact_duplicate_rows_removed": len(duplicate_indices),
        "analysis_rows_after_exact_deduplication": len(analyses),
        "sample_urls": len(samples),
        "row_linkage": {
            "normalized_name_matches": int(name_match.sum()),
            "coordinate_matches": int(coordinate_match.sum()),
        },
        "cohorts": {
            "exclusive_target_setting": int(
                samples["flag_exclusive_target_setting"].sum()
            ),
            "mixed_or_secondary": int(
                (~samples["flag_exclusive_target_setting"]).sum()
            ),
            "strict_candidate": int(samples["strict_candidate"].sum()),
            "model_ready_broad": int(samples["model_ready_broad"].sum()),
            "model_ready_immobile": int(samples["model_ready_immobile"].sum()),
            "model_ready_broad_no_severe_alteration": int(
                samples["model_ready_broad_no_severe_alteration"].sum()
            ),
            "model_ready_immobile_no_severe_alteration": int(
                samples["model_ready_immobile_no_severe_alteration"].sum()
            ),
        },
        "nonpositive_replacements": nonpositive_replacements,
        "bibliography_rows": len(bibliography),
    }
    return samples, profile, bibliography


def main() -> None:
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    OUT_QC.mkdir(parents=True, exist_ok=True)
    OUT_REFS.parent.mkdir(parents=True, exist_ok=True)
    exports = yaml.safe_load(EXPORT_CONFIG.read_text(encoding="utf-8"))["exports"]
    study = yaml.safe_load(STUDY_CONFIG.read_text(encoding="utf-8"))
    broad_features = study["feature_sets"]["broad_v1"]
    immobile_features = study["feature_sets"]["immobile_v1"]
    model_features = list(dict.fromkeys(broad_features + immobile_features))

    frames: list[pd.DataFrame] = []
    profiles: list[dict[str, Any]] = []
    bibliographies: list[pd.DataFrame] = []
    for spec in exports:
        frame, profile, bibliography = load_export(
            spec, model_features, broad_features, immobile_features
        )
        frames.append(frame)
        profiles.append(profile)
        bibliographies.append(bibliography)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    if combined["record_id"].duplicated().any():
        raise ValueError("PetDB primary4 record_id must be unique")
    combined["citation_overlap_component"] = build_overlap_components(
        combined["citation_set"]
    )
    combined.to_parquet(OUT_PARQUET, index=False)
    combined.to_csv(OUT_CSV, index=False, compression="gzip")

    bibliography = (
        pd.concat(bibliographies, ignore_index=True)
        .sort_values(["citation_url", "citation", "export_label"], na_position="last")
        .reset_index(drop=True)
    )
    bibliography.to_csv(OUT_REFS, index=False)

    flow_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    for profile in profiles:
        label = profile["label"]
        flow_rows.extend(
            {"label": label, "stage": stage, "n": n}
            for stage, n in {
                "raw_analysis_rows": profile["raw_analysis_rows"],
                "after_exact_duplicate_removal": profile[
                    "analysis_rows_after_exact_deduplication"
                ],
                "unique_sample_urls": profile["sample_urls"],
                **profile["cohorts"],
            }.items()
        )
        class_data = combined[combined["label_primary4"].eq(label)]
        for cohort in ["all", "model_ready_broad", "model_ready_immobile"]:
            cohort_data = (
                class_data
                if cohort == "all"
                else class_data[class_data[cohort]]
            )
            for feature in model_features:
                values = cohort_data[feature]
                coverage_rows.append(
                    {
                        "label": label,
                        "cohort": cohort,
                        "feature": feature,
                        "samples": len(cohort_data),
                        "non_null": int(values.notna().sum()),
                        "non_null_pct": round(100 * values.notna().mean(), 4)
                        if len(cohort_data)
                        else None,
                    }
                )
    pd.DataFrame(flow_rows).to_csv(
        OUT_QC / "petdb_primary4_qc_flow.csv", index=False
    )
    pd.DataFrame(coverage_rows).to_csv(
        OUT_QC / "petdb_primary4_sample_feature_coverage.csv", index=False
    )

    summary = {
        "dataset": "PetDB 2.0 primary four-class sample-level cohort",
        "version": "0.1",
        "feature_contract": {
            "broad": broad_features,
            "immobile": immobile_features,
            "sample_aggregation": "median by stable Sample URL",
            "iron_harmonization": [
                "reported FeOT",
                "0.8998 * reported Fe2O3T",
                "reported FeO + 0.8998 * reported Fe2O3",
            ],
        },
        "exports": profiles,
        "combined": {
            "sample_rows": len(combined),
            "exclusive_high_confidence": int(
                combined["flag_exclusive_target_setting"].sum()
            ),
            "model_ready_broad": int(combined["model_ready_broad"].sum()),
            "model_ready_immobile": int(combined["model_ready_immobile"].sum()),
            "class_counts_all": combined["label_primary4"].value_counts().to_dict(),
            "class_counts_broad": combined.loc[
                combined["model_ready_broad"], "label_primary4"
            ].value_counts().to_dict(),
            "class_counts_immobile": combined.loc[
                combined["model_ready_immobile"], "label_primary4"
            ].value_counts().to_dict(),
        },
        "service_anomaly_policy": (
            "The collided/headerless 5353 bundles are retained as evidence but excluded; "
            "only hash-verified clean exports listed in config/petdb_primary4_exports.yaml are used."
        ),
        "outputs": {
            "parquet": str(OUT_PARQUET.relative_to(ROOT)),
            "portable_csv": str(OUT_CSV.relative_to(ROOT)),
            "bibliography": str(OUT_REFS.relative_to(ROOT)),
            "qc_flow": "reports/data_quality/petdb_primary4_qc_flow.csv",
            "coverage": "reports/data_quality/petdb_primary4_sample_feature_coverage.csv",
        },
    }
    (OUT_QC / "petdb_primary4_processing_profile.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
