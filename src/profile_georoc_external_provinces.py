from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from audit_georoc_cib_provinces import FEATURES, INDEX, MAJORS


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "data_quality"


def province_root(value: object) -> str:
    parts = [part.strip() for part in str(value).split(" / ") if part.strip()]
    return " / ".join(parts[:2]) if parts else "UNKNOWN"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    usecols = [
        "CITATIONS",
        "TECTONIC SETTING",
        "LOCATION",
        "SAMPLE NAME",
        "flag_strict_candidate",
        "FEOT(WT%)",
        "FE2O3(WT%)",
        "FEO(WT%)",
    ] + [feature for feature in FEATURES if feature != "FEOT_HARMONIZED(WT%)"]
    data = pd.read_csv(INDEX, usecols=usecols, low_memory=False)
    data = data.loc[
        data["TECTONIC SETTING"].isin(["CONVERGENT MARGIN", "SUBMARINE RIDGE"])
    ].copy()
    data["FEOT_HARMONIZED(WT%)"] = data["FEOT(WT%)"]
    calculated = data["FEO(WT%)"].fillna(0) + 0.8998 * data[
        "FE2O3(WT%)"
    ].fillna(0)
    iron_reported = data[["FEO(WT%)", "FE2O3(WT%)"]].notna().any(axis=1)
    data.loc[
        data["FEOT_HARMONIZED(WT%)"].isna() & iron_reported,
        "FEOT_HARMONIZED(WT%)",
    ] = calculated
    data["model_ready_common18"] = (
        data["flag_strict_candidate"].fillna(False)
        & data["SIO2(WT%)"].between(40, 55)
        & data[FEATURES].notna().all(axis=1)
        & data[FEATURES].gt(0).all(axis=1)
        & data[MAJORS].sum(axis=1, min_count=len(MAJORS)).between(90, 105)
    )
    data = data.loc[data["model_ready_common18"]].copy()
    data["province_root"] = data["LOCATION"].map(province_root)
    data["citation_ids"] = data["CITATIONS"].map(
        lambda value: tuple(re.findall(r"\[(\d+)\]", str(value)))
    )
    profile = (
        data.groupby(["TECTONIC SETTING", "province_root"], dropna=False)
        .agg(
            model_ready_common18=("SAMPLE NAME", "size"),
            location_n=("LOCATION", "nunique"),
            citation_bundle_n=("citation_ids", "nunique"),
            sample_name_n=("SAMPLE NAME", "nunique"),
        )
        .reset_index()
        .sort_values(
            ["TECTONIC SETTING", "model_ready_common18"],
            ascending=[True, False],
        )
    )
    profile.to_csv(OUT / "georoc_external_province_profile.csv", index=False)
    print(
        profile.groupby("TECTONIC SETTING", group_keys=False)
        .head(30)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
