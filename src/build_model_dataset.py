from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "interim" / "georoc_basalt_core_index.csv.gz"
CITATIONS = ROOT / "data" / "raw" / "2024-12-4EZ7ID_CITATIONS.tab"
LABEL_CONFIG = ROOT / "config" / "label_map.yaml"
STUDY_CONFIG = ROOT / "config" / "study_config.yaml"
OUT_DATA = ROOT / "data" / "processed" / "georoc_primary3_v0_1.parquet"
OUT_QC = ROOT / "reports" / "data_quality"
OUT_REFS = ROOT / "references" / "georoc_primary3_bibliography.csv"


def citation_ids(value: object) -> list[str]:
    ids = set(re.findall(r"[0-9]+", "" if pd.isna(value) else str(value)))
    return sorted(ids, key=int)


def normalized_citation_set(value: object) -> str:
    return "|".join(citation_ids(value))


def normalized_location_root(value: object) -> str:
    if pd.isna(value):
        return "<MISSING>"
    root = str(value).split("/")[0]
    return re.sub(r"\s+", " ", root).strip().upper() or "<MISSING>"


def build_overlap_components(citation_sets: pd.Series) -> pd.Series:
    """Group rows when their citation ID sets overlap, for conservative sensitivity tests."""

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
            low, high = sorted((left_root, right_root), key=int)
            parent[high] = low

    parsed = [item.split("|") if item else [] for item in citation_sets]
    for ids in parsed:
        for item in ids:
            find(item)
        for item in ids[1:]:
            union(ids[0], item)

    component = []
    for ids in parsed:
        component.append(find(ids[0]) if ids else "<MISSING>")
    return pd.Series(component, index=citation_sets.index, dtype="string")


def main() -> None:
    OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    OUT_QC.mkdir(parents=True, exist_ok=True)
    OUT_REFS.parent.mkdir(parents=True, exist_ok=True)

    label_cfg = yaml.safe_load(LABEL_CONFIG.read_text(encoding="utf-8"))
    study_cfg = yaml.safe_load(STUDY_CONFIG.read_text(encoding="utf-8"))
    raw_map = label_cfg["raw_settings"]
    primary_map = {
        raw_label: spec["primary3"]
        for raw_label, spec in raw_map.items()
        if spec["include_primary3"]
    }
    fine_map = {raw_label: spec["fine"] for raw_label, spec in raw_map.items()}
    confidence_map = {raw_label: spec["confidence"] for raw_label, spec in raw_map.items()}

    data = pd.read_csv(INPUT, low_memory=False)
    starting_rows = len(data)
    data["label_fine"] = data["TECTONIC SETTING"].map(fine_map)
    data["label_primary3"] = data["TECTONIC SETTING"].map(primary_map)
    data["label_confidence"] = data["TECTONIC SETTING"].map(confidence_map)

    candidate = data.loc[
        data["flag_strict_candidate"] & data["label_primary3"].notna()
    ].copy()
    after_scope_rows = len(candidate)

    derived_columns = {
        "source_file", "flag_whole_rock", "flag_volcanic", "flag_basalt_name",
        "flag_silica_40_55", "flag_strict_candidate", "core_numeric_count",
        "label_fine", "label_primary3", "label_confidence",
    }
    dedupe_subset = [column for column in candidate.columns if column not in derived_columns]
    duplicate_mask = candidate.duplicated(subset=dedupe_subset, keep="first")
    exact_duplicates_removed = int(duplicate_mask.sum())
    candidate = candidate.loc[~duplicate_mask].copy()

    major_base = [
        "SIO2(WT%)", "TIO2(WT%)", "AL2O3(WT%)", "CAO(WT%)", "MGO(WT%)",
        "MNO(WT%)", "K2O(WT%)", "NA2O(WT%)", "P2O5(WT%)",
    ]
    candidate["FEO_EQ(WT%)"] = candidate["FEOT(WT%)"].where(
        candidate["FEOT(WT%)"].notna(),
        candidate["FEO(WT%)"].fillna(0) + 0.8998 * candidate["FE2O3(WT%)"].fillna(0),
    )
    candidate["major_complete"] = (
        candidate[major_base].notna().all(axis=1) & candidate["FEO_EQ(WT%)"].notna()
    )
    candidate["major_total_calc"] = candidate[major_base].sum(axis=1) + candidate["FEO_EQ(WT%)"]
    candidate["major_total_plausible"] = (
        candidate["major_complete"] & candidate["major_total_calc"].between(85.0, 105.0)
    )

    invalid_major_rows = int((~candidate["major_total_plausible"]).sum())
    candidate = candidate.loc[candidate["major_total_plausible"]].copy()

    broad_features = study_cfg["feature_sets"]["broad_v1"]
    immobile_features = study_cfg["feature_sets"]["immobile_v1"]
    trace_features = [column for column in set(broad_features + immobile_features) if "(PPM)" in column]
    zero_replacements: list[dict[str, object]] = []
    for column in trace_features:
        nonpositive = candidate[column].notna() & (candidate[column] <= 0)
        zero_replacements.append(
            {"field": column, "nonpositive_replaced_n": int(nonpositive.sum())}
        )
        candidate.loc[nonpositive, column] = np.nan

    candidate["citation_set"] = candidate["CITATIONS"].map(normalized_citation_set)
    candidate["citation_overlap_component"] = build_overlap_components(candidate["citation_set"])
    candidate["location_root"] = candidate["LOCATION"].map(normalized_location_root)

    citation_group_classes = candidate.groupby("citation_set")["label_primary3"].nunique()
    ambiguous_citation_sets = set(citation_group_classes[citation_group_classes > 1].index)
    candidate["citation_set_unambiguous"] = ~candidate["citation_set"].isin(ambiguous_citation_sets)

    candidate["broad_observed_n"] = candidate[broad_features].notna().sum(axis=1)
    candidate["immobile_observed_n"] = candidate[immobile_features].notna().sum(axis=1)
    candidate["model_ready_broad"] = (
        candidate["citation_set_unambiguous"] & (candidate["broad_observed_n"] >= 12)
    )
    candidate["model_ready_immobile"] = (
        candidate["citation_set_unambiguous"] & (candidate["immobile_observed_n"] >= 10)
    )

    candidate["record_id"] = np.where(
        candidate["UNIQUE_ID"].notna(),
        "GEOROC:" + candidate["UNIQUE_ID"].astype("Int64").astype(str),
        "GEOROC-NOID:" + pd.util.hash_pandas_object(
            candidate[dedupe_subset].astype(str), index=False
        ).astype(str),
    )

    if candidate["record_id"].duplicated().any():
        raise RuntimeError("record_id must be unique after exact deduplication")

    candidate.to_parquet(OUT_DATA, index=False)

    coverage = (
        candidate.groupby("label_primary3")[broad_features + immobile_features]
        .agg(lambda series: series.notna().mean())
        .T.reset_index(names="field")
    )
    coverage["overall"] = candidate[broad_features + immobile_features].notna().mean().reindex(coverage["field"]).to_numpy()
    coverage = coverage.drop_duplicates("field").sort_values("overall", ascending=False)
    coverage.to_csv(OUT_QC / "primary3_feature_coverage.csv", index=False)
    pd.DataFrame(zero_replacements).sort_values("nonpositive_replaced_n", ascending=False).to_csv(
        OUT_QC / "trace_nonpositive_replacements.csv", index=False
    )

    class_counts = (
        candidate.groupby(["label_primary3", "TECTONIC SETTING"], dropna=False)
        .agg(
            rows=("record_id", "size"),
            model_ready_broad=("model_ready_broad", "sum"),
            citation_sets=("citation_set", "nunique"),
            location_roots=("location_root", "nunique"),
        )
        .reset_index()
    )
    class_counts.to_csv(OUT_QC / "primary3_class_counts.csv", index=False)

    citation_table = pd.read_csv(CITATIONS, encoding="latin-1", low_memory=False)
    citation_table["CITATIONS"] = citation_table["CITATIONS"].astype(str)
    used_ids = sorted(
        {item for value in candidate.loc[candidate["model_ready_broad"], "CITATIONS"] for item in citation_ids(value)},
        key=int,
    )
    used_refs = citation_table[citation_table["CITATIONS"].isin(used_ids)].copy()
    used_refs.to_csv(OUT_REFS, index=False)
    missing_ids = sorted(set(used_ids) - set(citation_table["CITATIONS"]), key=int)
    pd.DataFrame({"missing_citation_id": missing_ids}).to_csv(
        OUT_QC / "primary3_missing_citation_ids.csv", index=False
    )

    overlap_stats = (
        candidate.groupby("citation_overlap_component")
        .agg(rows=("record_id", "size"), classes=("label_primary3", "nunique"))
    )
    profile = {
        "dataset_version": "georoc_primary3_v0_1",
        "input_rows": starting_rows,
        "strict_in_scope_rows": after_scope_rows,
        "exact_duplicates_removed": exact_duplicates_removed,
        "major_incomplete_or_implausible_removed": invalid_major_rows,
        "processed_rows": int(len(candidate)),
        "ambiguous_citation_sets_n": int(len(ambiguous_citation_sets)),
        "rows_in_ambiguous_citation_sets": int((~candidate["citation_set_unambiguous"]).sum()),
        "model_ready_broad_rows": int(candidate["model_ready_broad"].sum()),
        "model_ready_immobile_rows": int(candidate["model_ready_immobile"].sum()),
        "citation_sets_n": int(candidate["citation_set"].nunique()),
        "location_roots_n": int(candidate["location_root"].nunique()),
        "citation_overlap_components_n": int(candidate["citation_overlap_component"].nunique()),
        "citation_overlap_ambiguous_components_n": int((overlap_stats["classes"] > 1).sum()),
        "largest_citation_overlap_component_rows": int(overlap_stats["rows"].max()),
        "used_citation_ids_n": int(len(used_ids)),
        "bibliography_resolved_ids_n": int(len(used_refs)),
        "bibliography_missing_ids_n": int(len(missing_ids)),
    }
    (OUT_QC / "primary3_processing_profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(profile, ensure_ascii=False, indent=2))
    print("\nClass counts:\n", class_counts.to_string(index=False))


if __name__ == "__main__":
    main()

