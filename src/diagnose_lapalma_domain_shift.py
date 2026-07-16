from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_lapalma_oib_external import FEATURES as FEATURES_22
from run_petdb_primary4_validation import models


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "reports" / "modeling"


def finite_float(value: object) -> float | None:
    return None if pd.isna(value) else float(value)


def log_features(frame: pd.DataFrame) -> pd.DataFrame:
    return np.log10(frame[FEATURES_22].where(frame[FEATURES_22] > 0))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(PROCESSED / "petdb_primary4_v0_1.parquet")
    train = train.loc[train["model_ready_broad"]].reset_index(drop=True)
    palma = pd.read_parquet(PROCESSED / "la_palma_2021_oib_external_v0_1.parquet")
    palma = palma.loc[palma["strict_external_candidate"]].reset_index(drop=True)

    comparison_rows: list[dict[str, object]] = []
    for feature in FEATURES_22:
        external_values = palma[feature].dropna()
        row: dict[str, object] = {
            "feature": feature,
            "la_palma_median": float(external_values.median()),
            "la_palma_min": float(external_values.min()),
            "la_palma_max": float(external_values.max()),
        }
        for label, group in train.groupby("label_primary4"):
            values = group[feature].dropna()
            row[f"petdb_{label.lower()}_median"] = finite_float(values.median())
            row[f"la_palma_median_percentile_in_{label.lower()}"] = (
                float(values.le(external_values.median()).mean())
                if len(values)
                else None
            )
        comparison_rows.append(row)
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(OUT / "la_palma_feature_domain_comparison.csv", index=False)

    y = train["label_primary4"].astype(str)
    X_train = log_features(train)
    X_external = log_features(palma)
    model = models()["logistic_balanced"]
    model.fit(X_train, y)
    imputer = model.named_steps["imputer"]
    scaler = model.named_steps["scale"]
    classifier = model.named_steps["model"]
    median_external = pd.DataFrame([X_external.median()], columns=FEATURES_22)
    transformed = scaler.transform(imputer.transform(median_external))[0]
    transformed_names = imputer.get_feature_names_out(FEATURES_22)
    class_index = {label: i for i, label in enumerate(classifier.classes_)}
    coefficient_difference = (
        classifier.coef_[class_index["CIB"]] - classifier.coef_[class_index["OIB"]]
    )
    contributions = pd.DataFrame(
        {
            "transformed_feature": transformed_names,
            "scaled_external_value": transformed,
            "coefficient_cib_minus_oib": coefficient_difference,
            "contribution_to_cib_minus_oib_logit": (
                transformed * coefficient_difference
            ),
        }
    ).sort_values("contribution_to_cib_minus_oib_logit", ascending=False)
    contributions.to_csv(
        OUT / "la_palma_logistic_cib_vs_oib_contributions.csv", index=False
    )

    ratio_rows: list[dict[str, object]] = []
    for cohort, frame in [("La Palma 2021", palma)] + [
        (f"PetDB {label}", train.loc[train["label_primary4"].eq(label)])
        for label in ["ARC", "CIB", "MORB", "OIB"]
    ]:
        ratio_rows.append(
            {
                "cohort": cohort,
                "nb_y_median": finite_float(
                    (frame["NB(PPM)"] / frame["Y(PPM)"]).median()
                ),
                "zr_y_median": finite_float(
                    (frame["ZR(PPM)"] / frame["Y(PPM)"]).median()
                ),
                "th_nb_median": finite_float(
                    (frame["TH(PPM)"] / frame["NB(PPM)"]).median()
                ),
            }
        )
    ratios = pd.DataFrame(ratio_rows)
    ratios.to_csv(OUT / "la_palma_process_ratio_comparison.csv", index=False)

    intercept_difference = float(
        classifier.intercept_[class_index["CIB"]]
        - classifier.intercept_[class_index["OIB"]]
    )
    top_positive = contributions.head(8)[
        ["transformed_feature", "contribution_to_cib_minus_oib_logit"]
    ].to_dict("records")
    top_negative = contributions.tail(8).sort_values(
        "contribution_to_cib_minus_oib_logit"
    )[["transformed_feature", "contribution_to_cib_minus_oib_logit"]].to_dict(
        "records"
    )
    audit = {
        "external_n": len(palma),
        "logistic_intercept_cib_minus_oib": intercept_difference,
        "logistic_median_sample_total_cib_minus_oib_logit": float(
            intercept_difference
            + contributions["contribution_to_cib_minus_oib_logit"].sum()
        ),
        "top_contributions_toward_cib": top_positive,
        "top_contributions_toward_oib": top_negative,
        "interpretation_policy": (
            "Contributions diagnose the fitted PetDB classifier, not causal geological "
            "effects. Geological interpretation requires concordance with process ratios "
            "and eruption-stage behavior."
        ),
        "structural_missingness_warning": (
            "This diagnostic intentionally fits the 22-variable contract to reveal "
            "database-specific missingness leakage. Scientific external scoring uses "
            "the common18 contract, which excludes V, Cr, Ni and Th."
        ),
    }
    (OUT / "la_palma_domain_shift_manifest.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(ratios.to_string(index=False))
    print(contributions.head(10).to_string(index=False))
    print(contributions.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
