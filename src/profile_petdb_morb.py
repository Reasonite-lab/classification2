from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "petdb_morb"
EXPORT_ID = "earthchem_export_jul14_26_14620"
ZIP_PATH = RAW_DIR / f"{EXPORT_ID}.zip"
EXPORT_DIR = RAW_DIR / EXPORT_ID
MAIN_CSV = EXPORT_DIR / f"{EXPORT_ID}.csv"
SAMPLE_CSV = EXPORT_DIR / f"{EXPORT_ID}_sample_metadata.csv"
TERMS_PATH = EXPORT_DIR / f"{EXPORT_ID}.txt"
QUALITY_DIR = ROOT / "reports" / "data_quality"

SOURCE_URL = (
    "https://earthchem-search-export-production.s3.us-east-2.amazonaws.com/"
    f"{EXPORT_ID}.zip"
)
QUERY_URL = (
    "https://www.earthchem.org/petdb?"
    "analysisTypes=rock%3A%3A%5BWHOLE+ROCK%5D&"
    "tectonicSettings=SPREADING+CENTER&"
    "taxons=igneous%3Avolcanic%3Amafic%3A%3A%5BBASALT%5D&"
    "outputMetadata=all&timestamp=1784037977190"
)
EXPECTED_SHA256 = "24cbc5c90410248790c3560a82aa599a39a4032a24ab762b72db198a6363eb05"
EXPECTED_MD5 = "210133d92712637ccd017fed1a567b93"

TARGET_FEATURES = [
    "SiO2 (wt%)",
    "TiO2 (wt%)",
    "Al2O3 (wt%)",
    "FeOT (wt%)",
    "CaO (wt%)",
    "MgO (wt%)",
    "MnO (wt%)",
    "K2O (wt%)",
    "Na2O (wt%)",
    "P2O5 (wt%)",
    "V (ppm)",
    "Cr (ppm)",
    "Ni (ppm)",
    "Rb (ppm)",
    "Sr (ppm)",
    "Y (ppm)",
    "Zr (ppm)",
    "Nb (ppm)",
    "La (ppm)",
    "Ce (ppm)",
    "Nd (ppm)",
    "Sm (ppm)",
    "Eu (ppm)",
    "Yb (ppm)",
    "Lu (ppm)",
    "Hf (ppm)",
    "Ta (ppm)",
    "Th (ppm)",
    "U (ppm)",
]

KEY_METADATA = [
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


def file_hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_string_series(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    return text.replace({"": pd.NA, "N/A": pd.NA, "NA": pd.NA, "NULL": pd.NA})


def normalize_sample_name(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if text.startswith('="') and text.endswith('"'):
        text = text[2:-1].replace('""', '"').strip()
    return text or None


def valid_sample_name(value: Any) -> bool:
    return value is not None and not pd.isna(value) and bool(str(value).strip())


def update_counter(counter: Counter[str], series: pd.Series) -> None:
    values = normalize_string_series(series).dropna().astype(str)
    counter.update(values.tolist())


def header_profile(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle))
    counts = Counter(header)
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    return {
        "columns": len(header),
        "unique_columns": len(counts),
        "duplicate_column_names": duplicates,
        "all_target_features_present": all(feature in counts for feature in TARGET_FEATURES),
        "missing_target_features": [feature for feature in TARGET_FEATURES if feature not in counts],
        "missing_key_metadata": [column for column in KEY_METADATA if column not in counts],
    }


def profile_main_csv(path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    row_count = 0
    exact_duplicate_rows = 0
    composite_duplicate_rows = 0
    seen_row_hashes: set[int] = set()
    seen_composite_hashes: set[int] = set()
    sample_urls: set[str] = set()
    sample_names: set[str] = set()
    citation_urls: set[str] = set()
    category_counts: dict[str, Counter[str]] = {
        column: Counter()
        for column in [
            "Data Compilation",
            "Analysis Type",
            "Analyzed Material",
            "Sample Type",
            "Sample SubType",
        ]
    }
    feature_non_null = Counter()
    feature_numeric_valid = Counter()
    feature_numeric_invalid = Counter()
    feature_nonpositive = Counter()
    feature_min: dict[str, float] = {}
    feature_max: dict[str, float] = {}
    sample_feature_presence: dict[str, set[str]] = defaultdict(set)
    silica_in_strict_range = 0
    silica_numeric_rows = 0
    composite_columns = [
        "Sample URL",
        "Citation URL",
        "Analytical batch variables; Method Code; Method ID",
        "Analysis Comment",
        "Calc Avg",
        "Number Of Replicates",
    ]

    for chunk in pd.read_csv(path, dtype=str, chunksize=1500, low_memory=False):
        row_count += len(chunk)
        normalized = chunk.fillna("").astype(str)
        row_hashes = pd.util.hash_pandas_object(normalized, index=False).astype("uint64")
        for value in row_hashes.tolist():
            value_int = int(value)
            if value_int in seen_row_hashes:
                exact_duplicate_rows += 1
            else:
                seen_row_hashes.add(value_int)

        available_composite = [column for column in composite_columns if column in chunk.columns]
        composite_hashes = pd.util.hash_pandas_object(
            normalized[available_composite], index=False
        ).astype("uint64")
        for value in composite_hashes.tolist():
            value_int = int(value)
            if value_int in seen_composite_hashes:
                composite_duplicate_rows += 1
            else:
                seen_composite_hashes.add(value_int)

        url_series = normalize_string_series(chunk["Sample URL"])
        sample_urls.update(url_series.dropna().astype(str).tolist())
        sample_names.update(
            name
            for name in chunk["Sample Name"].map(normalize_sample_name).tolist()
            if valid_sample_name(name)
        )
        citation_urls.update(
            normalize_string_series(chunk["Citation URL"]).dropna().astype(str).tolist()
        )
        for column, counter in category_counts.items():
            update_counter(counter, chunk[column])

        for feature in TARGET_FEATURES:
            source = normalize_string_series(chunk[feature])
            present = source.notna()
            numeric = pd.to_numeric(source, errors="coerce")
            valid = numeric.notna()
            feature_non_null[feature] += int(present.sum())
            feature_numeric_valid[feature] += int(valid.sum())
            feature_numeric_invalid[feature] += int((present & ~valid).sum())
            feature_nonpositive[feature] += int((numeric <= 0).sum())
            if valid.any():
                current_min = float(numeric.min())
                current_max = float(numeric.max())
                feature_min[feature] = min(feature_min.get(feature, current_min), current_min)
                feature_max[feature] = max(feature_max.get(feature, current_max), current_max)
            for sample_url in url_series[present].dropna().astype(str).unique().tolist():
                sample_feature_presence[sample_url].add(feature)

        silica = pd.to_numeric(normalize_string_series(chunk["SiO2 (wt%)"]), errors="coerce")
        silica_numeric_rows += int(silica.notna().sum())
        silica_in_strict_range += int(silica.between(40.0, 55.0, inclusive="both").sum())

    sample_count = len(sample_urls)
    sample_feature_counts = Counter()
    for present_features in sample_feature_presence.values():
        for feature in present_features:
            sample_feature_counts[feature] += 1

    coverage_rows = []
    for feature in TARGET_FEATURES:
        coverage_rows.append(
            {
                "feature": feature,
                "row_non_null": feature_non_null[feature],
                "row_non_null_pct": round(100 * feature_non_null[feature] / row_count, 4),
                "sample_non_null": sample_feature_counts[feature],
                "sample_non_null_pct": round(
                    100 * sample_feature_counts[feature] / sample_count, 4
                ),
                "numeric_valid": feature_numeric_valid[feature],
                "numeric_invalid": feature_numeric_invalid[feature],
                "nonpositive": feature_nonpositive[feature],
                "min": feature_min.get(feature),
                "max": feature_max.get(feature),
            }
        )
    coverage = pd.DataFrame(coverage_rows)

    profile = {
        "rows": row_count,
        "unique_sample_urls": sample_count,
        "unique_normalized_sample_names": len(sample_names),
        "unique_citation_urls": len(citation_urls),
        "exact_duplicate_rows": exact_duplicate_rows,
        "exact_duplicate_row_pct": round(100 * exact_duplicate_rows / row_count, 6),
        "duplicate_composite_analysis_keys": composite_duplicate_rows,
        "duplicate_composite_analysis_key_pct": round(
            100 * composite_duplicate_rows / row_count, 6
        ),
        "silica_numeric_rows": silica_numeric_rows,
        "silica_40_55_rows": silica_in_strict_range,
        "silica_40_55_pct_of_numeric": round(
            100 * silica_in_strict_range / silica_numeric_rows, 4
        )
        if silica_numeric_rows
        else None,
        "category_counts": {
            column: dict(counter.most_common()) for column, counter in category_counts.items()
        },
    }
    return profile, coverage


def profile_sample_metadata(
    path: Path,
    main_sample_names: set[str] | None = None,
    main_name_sequence: list[str | None] | None = None,
    main_url_sequence: list[str | None] | None = None,
    main_latitude_sequence: list[str | None] | None = None,
    main_longitude_sequence: list[str | None] | None = None,
    main_url_to_names: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    frame = pd.read_csv(path, dtype=str, low_memory=False)
    normalized_names = {
        name
        for name in frame["Analyzed Sample Name"].map(normalize_sample_name).tolist()
        if valid_sample_name(name)
    }
    required_columns = [
        "Analyzed Sample Name",
        "Latitude",
        "Longitude",
        "Tectonic Setting",
        "Rock Classification",
        "Rock Subtype/Descriptor",
        "Data Compilation",
    ]
    completeness = {
        column: {
            "non_null": int(normalize_string_series(frame[column]).notna().sum()),
            "non_null_pct": round(
                100 * normalize_string_series(frame[column]).notna().mean(), 4
            ),
        }
        for column in required_columns
    }
    tectonic_counts = Counter(
        normalize_string_series(frame["Tectonic Setting"]).dropna().astype(str).tolist()
    )
    rock_counts = Counter(
        normalize_string_series(frame["Rock Classification"]).dropna().astype(str).tolist()
    )
    subtype_counts = Counter(
        normalize_string_series(frame["Rock Subtype/Descriptor"])
        .dropna()
        .astype(str)
        .tolist()
    )
    alteration_counts = Counter(
        normalize_string_series(frame["Sample Alteration"]).dropna().astype(str).tolist()
    )
    metadata_name_sequence = frame["Analyzed Sample Name"].map(normalize_sample_name).tolist()
    row_name_match_count = None
    row_name_match_pct = None
    row_coordinate_match_count = None
    row_coordinate_match_pct = None
    exclusive_spreading_sample_urls = None
    mixed_setting_sample_urls = None
    unknown_setting_sample_urls = None
    name_linkage_sensitivity = None
    if main_name_sequence is not None and len(main_name_sequence) == len(frame):
        row_name_match_count = sum(
            (not valid_sample_name(left) and not valid_sample_name(right))
            or left == right
            for left, right in zip(main_name_sequence, metadata_name_sequence, strict=True)
        )
        row_name_match_pct = round(100 * row_name_match_count / len(frame), 4)
    metadata_latitudes = normalize_string_series(frame["Latitude"]).tolist()
    metadata_longitudes = normalize_string_series(frame["Longitude"]).tolist()
    if (
        main_latitude_sequence is not None
        and main_longitude_sequence is not None
        and len(main_latitude_sequence) == len(frame)
        and len(main_longitude_sequence) == len(frame)
    ):
        row_coordinate_match_count = sum(
            ((pd.isna(main_lat) and pd.isna(meta_lat)) or main_lat == meta_lat)
            and ((pd.isna(main_lon) and pd.isna(meta_lon)) or main_lon == meta_lon)
            for main_lat, meta_lat, main_lon, meta_lon in zip(
                main_latitude_sequence,
                metadata_latitudes,
                main_longitude_sequence,
                metadata_longitudes,
                strict=True,
            )
        )
        row_coordinate_match_pct = round(
            100 * row_coordinate_match_count / len(frame), 4
        )

    metadata_settings = normalize_string_series(frame["Tectonic Setting"])
    row_linkage_verified = (
        main_url_sequence is not None
        and len(main_url_sequence) == len(frame)
        and row_name_match_count == len(frame)
        and row_coordinate_match_count == len(frame)
    )
    if row_linkage_verified:
        url_to_settings: dict[str, set[str]] = defaultdict(set)
        for sample_url, setting in zip(
            main_url_sequence, metadata_settings, strict=True
        ):
            if not pd.isna(sample_url):
                url_key = str(sample_url)
                url_to_settings.setdefault(url_key, set())
                if not pd.isna(setting):
                    url_to_settings[url_key].add(str(setting))
        exclusive_spreading_sample_urls = sum(
            settings == {"SPREADING CENTER"}
            for settings in url_to_settings.values()
        )
        mixed_setting_sample_urls = sum(
            bool(settings) and settings != {"SPREADING CENTER"}
            for settings in url_to_settings.values()
        )
        unknown_setting_sample_urls = sum(
            not settings for settings in url_to_settings.values()
        )
    if main_url_to_names is not None:
        metadata_names = frame["Analyzed Sample Name"].map(normalize_sample_name)
        name_to_settings: dict[str, set[str]] = defaultdict(set)
        for sample_name, setting in zip(metadata_names, metadata_settings, strict=True):
            if valid_sample_name(sample_name) and not pd.isna(setting):
                name_to_settings[sample_name].add(str(setting))
        name_exclusive = 0
        name_mixed = 0
        name_unknown = 0
        for sample_names_for_url in main_url_to_names.values():
            settings_for_url: set[str] = set()
            for sample_name in sample_names_for_url:
                settings_for_url.update(name_to_settings.get(sample_name, set()))
            if settings_for_url == {"SPREADING CENTER"}:
                name_exclusive += 1
            elif settings_for_url:
                name_mixed += 1
            else:
                name_unknown += 1
        name_to_urls: dict[str, set[str]] = defaultdict(set)
        for sample_url, names in main_url_to_names.items():
            for name in names:
                name_to_urls[name].add(sample_url)
        ambiguous_names = {
            name for name, urls in name_to_urls.items() if len(urls) > 1
        }
        name_linkage_sensitivity = {
            "exclusive_spreading_center_sample_urls": name_exclusive,
            "mixed_or_secondary_setting_sample_urls": name_mixed,
            "unknown_setting_sample_urls": name_unknown,
            "normalized_names_linked_to_multiple_sample_urls": len(ambiguous_names),
            "sample_urls_affected_by_ambiguous_names": len(
                {
                    sample_url
                    for name in ambiguous_names
                    for sample_url in name_to_urls[name]
                }
            ),
            "method": "normalized sample-name mapping without row alignment",
        }
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "unique_normalized_sample_names": len(normalized_names),
        "duplicate_normalized_sample_name_rows": int(len(frame) - len(normalized_names)),
        "completeness": completeness,
        "tectonic_setting_counts": dict(tectonic_counts.most_common()),
        "rock_classification_counts": dict(rock_counts.most_common()),
        "rock_subtype_counts": dict(subtype_counts.most_common()),
        "sample_alteration_counts": dict(alteration_counts.most_common()),
        "main_name_overlap": len(normalized_names & main_sample_names)
        if main_sample_names is not None
        else None,
        "row_aligned_name_match_count": row_name_match_count,
        "row_aligned_name_match_pct": row_name_match_pct,
        "row_aligned_coordinate_match_count": row_coordinate_match_count,
        "row_aligned_coordinate_match_pct": row_coordinate_match_pct,
        "exclusive_spreading_center_sample_urls": exclusive_spreading_sample_urls,
        "mixed_or_secondary_setting_sample_urls": mixed_setting_sample_urls,
        "unknown_setting_sample_urls": unknown_setting_sample_urls,
        "tectonic_linkage_method": (
            "row-aligned metadata after 100% normalized-name and coordinate verification"
            if row_linkage_verified
            else "unverified"
        ),
        "name_linkage_sensitivity": name_linkage_sensitivity,
    }


def main() -> None:
    QUALITY_DIR.mkdir(parents=True, exist_ok=True)
    sha256 = file_hash(ZIP_PATH, "sha256")
    md5 = file_hash(ZIP_PATH, "md5")
    if sha256 != EXPECTED_SHA256 or md5 != EXPECTED_MD5:
        raise ValueError("PetDB export archive hash mismatch")

    header = header_profile(MAIN_CSV)
    main_profile, coverage = profile_main_csv(MAIN_CSV)
    main_names = set()
    main_name_sequence: list[str | None] = []
    main_url_sequence: list[str | None] = []
    main_latitude_sequence: list[str | None] = []
    main_longitude_sequence: list[str | None] = []
    main_url_to_names: dict[str, set[str]] = defaultdict(set)
    for chunk in pd.read_csv(
        MAIN_CSV,
        usecols=["Sample Name", "Sample URL", "Latitude", "Longitude"],
        dtype=str,
        chunksize=5000,
    ):
        normalized_chunk_names = chunk["Sample Name"].map(normalize_sample_name).tolist()
        main_name_sequence.extend(normalized_chunk_names)
        normalized_chunk_urls = normalize_string_series(chunk["Sample URL"]).tolist()
        main_url_sequence.extend(normalized_chunk_urls)
        main_latitude_sequence.extend(
            normalize_string_series(chunk["Latitude"]).tolist()
        )
        main_longitude_sequence.extend(
            normalize_string_series(chunk["Longitude"]).tolist()
        )
        for sample_url, sample_name in zip(
            normalized_chunk_urls, normalized_chunk_names, strict=True
        ):
            if not pd.isna(sample_url):
                url_key = str(sample_url)
                # Retain URLs whose exported sample name is blank. These records
                # cannot be linked to tectonic metadata by name and must be
                # counted explicitly as unknown rather than silently omitted.
                main_url_to_names.setdefault(url_key, set())
                if valid_sample_name(sample_name):
                    main_url_to_names[url_key].add(sample_name)
        main_names.update(
            name
            for name in normalized_chunk_names
            if valid_sample_name(name)
        )
    sample_profile = profile_sample_metadata(
        SAMPLE_CSV,
        main_names,
        main_name_sequence,
        main_url_sequence,
        main_latitude_sequence,
        main_longitude_sequence,
        main_url_to_names,
    )
    terms_text = TERMS_PATH.read_text(encoding="utf-8", errors="replace")

    primary_filter_values_consistent = (
        set(main_profile["category_counts"]["Analyzed Material"]) == {"WHOLE ROCK"}
        and set(main_profile["category_counts"]["Sample Type"])
        == {"igneous:volcanic:mafic"}
        and set(main_profile["category_counts"]["Sample SubType"]) == {"BASALT"}
    )

    profile = {
        "dataset": "PetDB 2.0 MORB export",
        "export_id": EXPORT_ID,
        "export_datetime_utc": "2026-07-14T14:07:56.434Z",
        "source_url": SOURCE_URL,
        "query_url": QUERY_URL,
        "archive": {
            "path": str(ZIP_PATH.relative_to(ROOT)),
            "bytes": ZIP_PATH.stat().st_size,
            "sha256": sha256,
            "md5": md5,
            "hashes_verified": True,
        },
        "files": {
            MAIN_CSV.name: MAIN_CSV.stat().st_size,
            SAMPLE_CSV.name: SAMPLE_CSV.stat().st_size,
            TERMS_PATH.name: TERMS_PATH.stat().st_size,
        },
        "header": header,
        "main_analysis_table": main_profile,
        "sample_metadata_table": sample_profile,
        "ui_export_summary": {
            "reported_rows": 22786,
            "reported_samples": 7721,
            "exported_rows": main_profile["rows"],
            "unique_sample_urls": main_profile["unique_sample_urls"],
            "row_difference": 22786 - main_profile["rows"],
            "sample_difference": 7721 - main_profile["unique_sample_urls"],
        },
        "primary_filter_values_consistent_with_query": primary_filter_values_consistent,
        "tectonic_filter_is_inclusive_not_exclusive": (
            set(sample_profile["tectonic_setting_counts"]) != {"SPREADING CENTER"}
        ),
        "intended_model_grain": "one normalized sample after within-sample aggregation",
        "raw_grain": "one exported analysis/batch row; multiple rows may map to one sample",
        "license_audit": {
            "browser_terms": "CC BY-SA 4.0",
            "bundle_terms": "CC BY 4.0",
            "bundle_mentions_astromaterials_not_petdb": "Astromaterials Data Synthesis" in terms_text,
            "conservative_project_policy": (
                "Treat the export as CC BY-SA 4.0, cite PetDB/EarthChem and every "
                "contributing publication, and preserve the downloaded terms file."
            ),
        },
        "quality_status": "share_with_caveats",
        "quality_caveats": [
            "Raw rows are repeated analyses rather than independent rock samples.",
            "Target-element coverage is sparse and must be profiled at sample level.",
            "The downloaded terms text and browser license label are inconsistent; use the stricter license.",
            "PetDB disclaims analytical accuracy; method, citation, alteration, and replicate metadata must be retained.",
        ],
    }

    profile_path = QUALITY_DIR / "petdb_morb_raw_profile.json"
    coverage_path = QUALITY_DIR / "petdb_morb_feature_coverage.csv"
    metadata_path = RAW_DIR / "export_metadata.json"
    profile_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    coverage.to_csv(coverage_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                key: profile[key]
                for key in [
                    "dataset",
                    "export_id",
                    "export_datetime_utc",
                    "source_url",
                    "query_url",
                    "archive",
                    "files",
                    "license_audit",
                ]
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps(profile, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
