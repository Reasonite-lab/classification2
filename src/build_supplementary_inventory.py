"""Build a curated, checksum-fixed inventory for the manuscript supplement.

The inventory points to frozen project artifacts. It does not copy or relicense
third-party raw files; public deposition must preserve source-specific terms.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "reports" / "manuscript" / "supplementary_inventory.csv"
OUT_MD = ROOT / "paper" / "SUPPLEMENTARY_DATA_INDEX.md"


ARTIFACTS = [
    ("contracts", "config/study_config.yaml", "Study inclusion, feature and validation contract"),
    ("contracts", "config/label_map.yaml", "Fine-to-primary tectonic label ontology"),
    ("contracts", "config/petdb_primary4_exports.yaml", "Accepted and excluded PetDB export bundles"),
    ("provenance", "references/data_sources.csv", "Unified source registry, identifiers, licences and audit status"),
    ("provenance", "reports/data_quality/raw_file_manifest.csv", "Raw-file sizes and checksums"),
    ("provenance", "references/georoc_primary3_bibliography.csv", "GEOROC sample-publication bibliography"),
    ("provenance", "references/petdb_primary4_bibliography.csv", "PetDB sample-publication bibliography by class export"),
    ("processed_data", "data/processed/georoc_primary3_v0_1.parquet", "Filtered GEOROC primary three-class table"),
    ("processed_data", "data/processed/petdb_primary4_v0_1.parquet", "Sample-median PetDB four-class table"),
    ("processed_data", "data/processed/pangaea_922011_arc_backarc_external_v0_1.parquet", "New Hebrides arc and Vate Trough cohort"),
    ("processed_data", "data/processed/rio_grande_rift_cib_external_v0_1.parquet", "Rio Grande Rift external CIB cohort"),
    ("processed_data", "data/processed/georoc_big_pine_cib_external_v0_1.parquet", "Big Pine external CIB cohort"),
    ("processed_data", "data/processed/georoc_costa_rica_arc_external_v0_1.parquet", "Costa Rica external ARC cohort"),
    ("processed_data", "data/processed/figshare_morb_external_v0_1.parquet", "SWIR provisional external MORB cohort"),
    ("processed_data", "data/processed/smar_2019_morb_external_v0_1.parquet", "South Mid-Atlantic Ridge external MORB cohort"),
    ("processed_data", "data/processed/mauna_loa_2022_oib_external_v0_1.parquet", "Mauna Loa external OIB cohort"),
    ("processed_data", "data/processed/la_palma_2021_oib_external_v0_1.parquet", "La Palma external OIB cohort"),
    ("internal_validation", "reports/modeling/petdb_primary4_overall_metrics.csv", "PetDB random and grouped validation metrics"),
    ("internal_validation", "reports/modeling/petdb_primary4_calibration_summary.csv", "Conservative probability calibration summary"),
    ("internal_validation", "reports/modeling/feature_bias_sensitivity_summary.csv", "Missingness and feature-panel sensitivity"),
    ("internal_validation", "reports/modeling/cross_database_overall_metrics.csv", "Bidirectional GEOROC-PetDB transfer metrics"),
    ("interpretability", "reports/modeling/petdb_primary4_permutation_importance_summary.csv", "Conservative grouped permutation importance"),
    ("interpretability", "reports/modeling/petdb_primary4_shap_global_overall.csv", "Descriptive SHAP importance"),
    ("external_validation", "reports/modeling/multiclass_external_manifest.json", "External-test feature and provenance contract"),
    ("external_validation", "reports/modeling/multiclass_external_metrics.csv", "External overall metrics"),
    ("external_validation", "reports/modeling/multiclass_external_per_class.csv", "External class-wise metrics"),
    ("external_validation", "reports/modeling/multiclass_external_by_province.csv", "Province-proxy outcomes"),
    ("external_validation", "reports/modeling/multiclass_external_predictions.csv", "Sample-level external probabilities and predictions"),
    ("transition_tests", "reports/modeling/east_african_rift_stress_summary.csv", "Whole-province East African Rift affinity summary"),
    ("transition_tests", "reports/modeling/east_african_rift_stress_predictions.parquet", "East African Rift sample-level probabilities"),
    ("transition_tests", "reports/modeling/multiclass_external_backarc_summary.csv", "Unscored Vate Trough mechanism test"),
    ("transition_tests", "reports/modeling/la_palma_oib_external_summary.csv", "La Palma eruption-state transfer summary"),
    ("audit", "reports/data_quality/georoc_big_pine_cib_external_audit.json", "Big Pine independence and chemistry audit"),
    ("audit", "reports/data_quality/georoc_costa_rica_arc_external_audit.json", "Costa Rica independence and chemistry audit"),
    ("audit", "reports/data_quality/smar_morb_external_audit.json", "South Mid-Atlantic Ridge source and chemistry audit"),
    ("audit", "reports/data_quality/la_palma_oib_external_audit.json", "La Palma source and chemistry audit"),
    ("audit", "reports/data_quality/mauna_loa_oib_external_audit.json", "Mauna Loa source and chemistry audit"),
    ("audit", "reports/data_quality/swir_publication_link_audit_2026-07-15.md", "SWIR journal-link search and provisional-status decision"),
    ("manuscript", "reports/manuscript/nsr_compliance_v1.json", "NSR length, reference and display-item checks"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    rows = []
    missing = []
    for section, relative, description in ARTIFACTS:
        path = ROOT / relative
        if not path.is_file():
            missing.append(relative)
            continue
        rows.append(
            {
                "section": section,
                "path": relative,
                "description": description,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    if missing:
        raise FileNotFoundError("Required supplementary artifacts missing: " + ", ".join(missing))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["section", "path", "description", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["section"]), []).append(row)

    lines = [
        "# Supplementary Data Index",
        "",
        "This index defines the frozen evidence package for the NSR manuscript. "
        "The machine-readable inventory is `reports/manuscript/supplementary_inventory.csv`; "
        "each entry is fixed by byte size and SHA-256.",
        "",
        "## Release policy",
        "",
        "The inventory distinguishes project-derived artifacts from third-party source files. "
        "A public archive must cite every source in `references/data_sources.csv`, preserve "
        "GEOROC/PetDB CC BY-SA terms, and must not relicense publisher supplements. The SWIR "
        "cohort remains provisional until its linked journal article is established or the cohort is replaced.",
        "",
    ]
    titles = {
        "contracts": "S1. Contracts and ontology",
        "provenance": "S2. Provenance and bibliographies",
        "processed_data": "S3. Processed analytical tables",
        "internal_validation": "S4. Internal and cross-database validation",
        "interpretability": "S5. Interpretability outputs",
        "external_validation": "S6. Strict external validation",
        "transition_tests": "S7. Geological transition tests",
        "audit": "S8. External-source audits",
        "manuscript": "S9. Manuscript compliance",
    }
    for section, entries in grouped.items():
        lines.extend([f"## {titles[section]}", ""])
        for row in entries:
            lines.append(f"- `{row['path']}` — {row['description']}")
        lines.append("")

    lines.extend(
        [
            "## Reproduction gates",
            "",
            "1. Run `python src/test_data_contract.py` and require the final contract-passed message.",
            "2. Run the external validation only if its inputs or model code changed.",
            "3. Run `src/validate_nsr_manuscript.py` after any numerical or structural manuscript edit.",
            "4. Rebuild this inventory after any frozen artifact changes so checksums remain authoritative.",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_CSV.relative_to(ROOT)} with {len(rows)} artifacts")
    print(f"Wrote {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
