from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "reports" / "data_quality"

CANDIDATES = {
    "PANGAEA.956077": {
        "file": "pangaea_956077.tsv",
        "doi": "10.1594/PANGAEA.956077",
        "license": "CC BY 4.0",
        "setting": "Reykjanes Ridge",
        "publication_doi": "10.1029/2022GC010788",
        "citation_key": "VARIATIONS IN VOLCANISM AND TECTONICS ALONG THE HOTSPOT",
    },
    "PANGAEA.727391": {
        "file": "pangaea_727391.tsv",
        "doi": "10.1594/PANGAEA.727391",
        "license": "CC BY 3.0",
        "setting": "Mid-Atlantic Ridge near Ascension Island",
        "publication_doi": "10.1093/petrology/egm068",
        "citation_key": "DEPTH OF PARTIAL CRYSTALLIZATION OF H2O-BEARING MORB",
    },
    "PANGAEA.707361": {
        "file": "pangaea_707361.tsv",
        "doi": "10.1594/PANGAEA.707361",
        "license": "CC BY 3.0",
        "setting": "DSDP Hole 24-238, Indian Ocean",
        "publication_doi": "10.1016/0016-7037(80)90209-4",
        "citation_key": "TRACE ELEMENTS IN OCEAN RIDGE BASALT GLASSES",
    },
}

FEATURE_ALIASES = {
    "SIO2(WT%)": ["SiO2 [%]"],
    "TIO2(WT%)": ["TiO2 [%]"],
    "AL2O3(WT%)": ["Al2O3 [%]"],
    "FEOT(WT%)": ["FeO [%]", "Fe2O3 [%]"],
    "CAO(WT%)": ["CaO [%]"],
    "MGO(WT%)": ["MgO [%]"],
    "MNO(WT%)": ["MnO [%]"],
    "K2O(WT%)": ["K2O [%]"],
    "NA2O(WT%)": ["Na2O [%]"],
    "P2O5(WT%)": ["P2O5 [%]"],
    "V(PPM)": ["V [mg/kg]"],
    "CR(PPM)": ["Cr [mg/kg]"],
    "NI(PPM)": ["Ni [mg/kg]"],
    "RB(PPM)": ["Rb [mg/kg]"],
    "SR(PPM)": ["Sr [mg/kg]"],
    "Y(PPM)": ["Y [mg/kg]"],
    "ZR(PPM)": ["Zr [mg/kg]"],
    "NB(PPM)": ["Nb [mg/kg]"],
    "LA(PPM)": ["La [mg/kg]"],
    "CE(PPM)": ["Ce [mg/kg]"],
    "ND(PPM)": ["Nd [mg/kg]"],
}


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_pangaea(path: Path) -> tuple[pd.DataFrame, str]:
    text = path.read_text(encoding="utf-8")
    end = text.index("*/")
    header = text[: end + 2]
    data = pd.read_csv(path, sep="\t", skiprows=header.count("\n") + 1, low_memory=False)
    return data, header


def normalize_name(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def resolve_column(columns: pd.Index, aliases: list[str]) -> str | None:
    for alias in aliases:
        for column in columns:
            if column == alias or column.startswith(f"{alias} ("):
                return column
    return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    petdb = pd.read_parquet(ROOT / "data" / "processed" / "petdb_morb_v0_1.parquet")
    petdb_names = {
        normalize_name(part)
        for value in petdb["sample_names"].dropna()
        for part in str(value).split(" || ")
        if normalize_name(part)
    }
    petdb_citations = "\n".join(petdb["citations"].dropna().astype(str)).upper()

    rows: list[dict[str, object]] = []
    detail: dict[str, object] = {}
    for dataset_id, metadata in CANDIDATES.items():
        path = RAW / metadata["file"]
        frame, header = read_pangaea(path)
        name_column = (
            "Event"
            if "Event" in frame and frame["Event"].notna().sum() > 0
            else "Sample label" if "Sample label" in frame else frame.columns[0]
        )
        valid = frame[name_column].notna()
        names = frame.loc[valid, name_column].astype(str)
        normalized = {normalize_name(value) for value in names if normalize_name(value)}
        overlap = sorted(normalized & petdb_names)
        feature_coverage: dict[str, int] = {}
        feature_frame = pd.DataFrame(index=frame.index)
        for feature, aliases in FEATURE_ALIASES.items():
            column = resolve_column(frame.columns, aliases)
            feature_frame[feature] = (
                pd.to_numeric(frame[column], errors="coerce")
                if column is not None
                else pd.Series(index=frame.index, dtype=float)
            )
            feature_coverage[feature] = int(feature_frame.loc[valid, feature].notna().sum())
        observed_features = [feature for feature, count in feature_coverage.items() if count > 0]
        sample_features = feature_frame.loc[valid].groupby(frame.loc[valid, name_column]).median()
        rows.append(
            {
                "dataset_id": dataset_id,
                "doi": metadata["doi"],
                "publication_doi": metadata["publication_doi"],
                "setting": metadata["setting"],
                "license": metadata["license"],
                "analysis_rows": int(valid.sum()),
                "unique_sample_names": len(normalized),
                "petdb_exact_normalized_name_overlap": len(overlap),
                "shared_primary_features": len(observed_features),
                "samples_with_at_least_10_shared_features": int(
                    sample_features.notna().sum(axis=1).ge(10).sum()
                ),
                "publication_citation_detected_in_petdb": metadata["citation_key"] in petdb_citations,
                "bytes": path.stat().st_size,
                "sha256": digest(path, "sha256"),
            }
        )
        detail[dataset_id] = {
            **metadata,
            "columns": list(frame.columns),
            "feature_nonmissing_rows": feature_coverage,
            "observed_primary_features": observed_features,
            "normalized_name_overlap_examples": overlap[:20],
            "header_contains_license": "License:" in header,
            "header_contains_citation": "Citation:" in header,
        }

    summary = pd.DataFrame(rows).sort_values(
        ["samples_with_at_least_10_shared_features", "unique_sample_names"],
        ascending=False,
    )
    summary.to_csv(OUT / "pangaea_morb_candidate_audit.csv", index=False)
    (OUT / "pangaea_morb_candidate_audit.json").write_text(
        json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
