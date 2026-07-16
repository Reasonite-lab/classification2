from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "georoc_2jetoa"
OUT_DIR = ROOT / "reports" / "data_quality"
INTERIM_DIR = ROOT / "data" / "interim"

MAJOR = [
    "SIO2(WT%)", "TIO2(WT%)", "AL2O3(WT%)", "FE2O3(WT%)", "FEO(WT%)",
    "FEOT(WT%)", "CAO(WT%)", "MGO(WT%)", "MNO(WT%)", "K2O(WT%)",
    "NA2O(WT%)", "P2O5(WT%)",
]
TRACE = [
    "SC(PPM)", "TI(PPM)", "V(PPM)", "CR(PPM)", "CO(PPM)", "NI(PPM)",
    "RB(PPM)", "SR(PPM)", "Y(PPM)", "ZR(PPM)", "NB(PPM)", "CS(PPM)",
    "BA(PPM)", "LA(PPM)", "CE(PPM)", "PR(PPM)", "ND(PPM)", "SM(PPM)",
    "EU(PPM)", "GD(PPM)", "TB(PPM)", "DY(PPM)", "HO(PPM)", "ER(PPM)",
    "TM(PPM)", "YB(PPM)", "LU(PPM)", "HF(PPM)", "TA(PPM)", "PB(PPM)",
    "TH(PPM)", "U(PPM)",
]
ISOTOPES = [
    "ND143_ND144", "SR87_SR86", "PB206_PB204", "PB207_PB204", "PB208_PB204",
]
META = [
    "CITATIONS", "TECTONIC SETTING", "LOCATION", "LOCATION COMMENT",
    "LATITUDE MIN", "LATITUDE MAX", "LONGITUDE MIN", "LONGITUDE MAX",
    "SAMPLE NAME", "ROCK NAME", "MIN. AGE (YRS.)", "MAX. AGE (YRS.)",
    "AGE", "ROCK TEXTURE", "ROCK TYPE", "ALTERATION", "MATERIAL", "UNIQUE_ID",
]


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_parts() -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = sorted(RAW_DIR.glob("2026-06-2JETOA_BASALT_part*.csv"))
    if len(parts) != 9:
        raise RuntimeError(f"Expected 9 GEOROC BASALT parts, found {len(parts)}")

    frames: list[pd.DataFrame] = []
    manifest: list[dict[str, object]] = []
    for path in parts:
        # GEOROC exports contain legacy extended characters in citation/location
        # fields, so latin-1 is the lossless single-byte decoding for ingestion.
        frame = pd.read_csv(path, dtype=str, low_memory=False, encoding="latin-1")
        frame = frame.loc[:, ~frame.columns.str.startswith("Unnamed:")]
        frame["source_file"] = path.name
        frames.append(frame)
        manifest.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "rows": len(frame),
                "columns": len(frame.columns) - 1,
                "md5": md5(path),
            }
        )
    return pd.concat(frames, ignore_index=True), pd.DataFrame(manifest)


def normalize_text(series: pd.Series) -> pd.Series:
    return series.fillna("").str.strip().str.replace(r"\s+", " ", regex=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    raw, manifest = read_parts()
    target_numeric = [c for c in MAJOR + TRACE + ISOTOPES if c in raw.columns]
    for column in target_numeric + ["LATITUDE MIN", "LATITUDE MAX", "LONGITUDE MIN", "LONGITUDE MAX"]:
        if column in raw.columns:
            raw[column] = pd.to_numeric(raw[column], errors="coerce")

    tectonic = normalize_text(raw["TECTONIC SETTING"])
    material = normalize_text(raw["MATERIAL"])
    rock_type = normalize_text(raw["ROCK TYPE"])
    alteration = normalize_text(raw["ALTERATION"])
    rock_name = normalize_text(raw["ROCK NAME"])

    raw["flag_whole_rock"] = material.str.contains(r"(?:^|\s|/)WR(?:\s|\[|/|$)", regex=True)
    raw["flag_volcanic"] = rock_type.str.contains(r"(?:^|\s|/)VOL(?:\s|\[|/|$)", regex=True)
    raw["flag_basalt_name"] = rock_name.str.contains("BASALT", case=False, regex=False)
    raw["flag_silica_40_55"] = raw["SIO2(WT%)"].between(40.0, 55.0, inclusive="both")
    raw["flag_strict_candidate"] = (
        raw["flag_whole_rock"]
        & raw["flag_volcanic"]
        & raw["flag_basalt_name"]
        & raw["flag_silica_40_55"]
    )
    raw["core_numeric_count"] = raw[target_numeric].notna().sum(axis=1)

    field_coverage = pd.DataFrame(
        {
            "field": raw.columns,
            "non_null_n": raw.notna().sum().to_numpy(),
            "non_null_pct": (raw.notna().mean().to_numpy() * 100).round(3),
            "unique_n": raw.nunique(dropna=True).to_numpy(),
        }
    ).sort_values(["non_null_pct", "field"], ascending=[False, True])

    class_counts = (
        tectonic.replace("", "<MISSING>")
        .value_counts(dropna=False)
        .rename_axis("tectonic_setting_raw")
        .reset_index(name="n")
    )
    class_counts["pct"] = (class_counts["n"] / len(raw) * 100).round(3)

    strict_counts = (
        raw.loc[raw["flag_strict_candidate"], "TECTONIC SETTING"]
        .fillna("<MISSING>")
        .value_counts(dropna=False)
        .rename_axis("tectonic_setting_raw")
        .reset_index(name="n")
    )

    profile = {
        "source": "GEOROC Compilation: Rock Types",
        "persistent_id": "doi:10.25625/2JETOA",
        "source_version": "files generated 2026-06-01",
        "files": int(len(manifest)),
        "rows": int(len(raw)),
        "columns_raw": int(len(raw.columns) - 7),
        "unique_id_missing_n": int(raw["UNIQUE_ID"].isna().sum()),
        "unique_id_duplicate_nonmissing_rows_n": int(
            raw.loc[raw["UNIQUE_ID"].notna(), "UNIQUE_ID"].duplicated(keep=False).sum()
        ),
        "exact_duplicate_rows_n": int(raw.drop(columns=["source_file"]).duplicated(keep=False).sum()),
        "tectonic_setting_missing_n": int(tectonic.eq("").sum()),
        "whole_rock_n": int(raw["flag_whole_rock"].sum()),
        "volcanic_n": int(raw["flag_volcanic"].sum()),
        "basalt_name_n": int(raw["flag_basalt_name"].sum()),
        "silica_40_55_n": int(raw["flag_silica_40_55"].sum()),
        "strict_candidate_n": int(raw["flag_strict_candidate"].sum()),
        "alteration_nonempty_n": int(alteration.ne("").sum()),
        "core_numeric_median_available": float(raw["core_numeric_count"].median()),
        "core_numeric_p10_available": float(raw["core_numeric_count"].quantile(0.10)),
        "core_numeric_p90_available": float(raw["core_numeric_count"].quantile(0.90)),
    }

    manifest.to_csv(OUT_DIR / "raw_file_manifest.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    field_coverage.to_csv(OUT_DIR / "field_coverage.csv", index=False)
    class_counts.to_csv(OUT_DIR / "tectonic_class_counts_raw.csv", index=False)
    strict_counts.to_csv(OUT_DIR / "tectonic_class_counts_strict_candidate.csv", index=False)
    with (OUT_DIR / "initial_profile.json").open("w", encoding="utf-8") as handle:
        json.dump(profile, handle, ensure_ascii=False, indent=2)

    index_columns = [c for c in META + MAJOR + TRACE + ISOTOPES if c in raw.columns]
    index_columns += [
        "source_file", "flag_whole_rock", "flag_volcanic", "flag_basalt_name",
        "flag_silica_40_55", "flag_strict_candidate", "core_numeric_count",
    ]
    raw[index_columns].to_csv(
        INTERIM_DIR / "georoc_basalt_core_index.csv.gz",
        index=False,
        compression="gzip",
    )

    print(json.dumps(profile, ensure_ascii=False, indent=2))
    print("\nTop tectonic settings:\n", class_counts.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
