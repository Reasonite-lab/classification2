# Supplementary Data Index

This index defines the frozen evidence package for the NSR manuscript. The machine-readable inventory is `reports/manuscript/supplementary_inventory.csv`; each entry is fixed by byte size and SHA-256.

## Release policy

The inventory distinguishes project-derived artifacts from third-party source files. A public archive must cite every source in `references/data_sources.csv`, preserve GEOROC/PetDB CC BY-SA terms, and must not relicense publisher supplements. The SWIR cohort remains provisional until its linked journal article is established or the cohort is replaced.

## S1. Contracts and ontology

- `config/study_config.yaml` — Study inclusion, feature and validation contract
- `config/label_map.yaml` — Fine-to-primary tectonic label ontology
- `config/petdb_primary4_exports.yaml` — Accepted and excluded PetDB export bundles

## S2. Provenance and bibliographies

- `references/data_sources.csv` — Unified source registry, identifiers, licences and audit status
- `reports/data_quality/raw_file_manifest.csv` — Raw-file sizes and checksums
- `references/georoc_primary3_bibliography.csv` — GEOROC sample-publication bibliography
- `references/petdb_primary4_bibliography.csv` — PetDB sample-publication bibliography by class export

## S3. Processed analytical tables

- `data/processed/georoc_primary3_v0_1.parquet` — Filtered GEOROC primary three-class table
- `data/processed/petdb_primary4_v0_1.parquet` — Sample-median PetDB four-class table
- `data/processed/pangaea_922011_arc_backarc_external_v0_1.parquet` — New Hebrides arc and Vate Trough cohort
- `data/processed/rio_grande_rift_cib_external_v0_1.parquet` — Rio Grande Rift external CIB cohort
- `data/processed/georoc_big_pine_cib_external_v0_1.parquet` — Big Pine external CIB cohort
- `data/processed/georoc_costa_rica_arc_external_v0_1.parquet` — Costa Rica external ARC cohort
- `data/processed/figshare_morb_external_v0_1.parquet` — SWIR provisional external MORB cohort
- `data/processed/smar_2019_morb_external_v0_1.parquet` — South Mid-Atlantic Ridge external MORB cohort
- `data/processed/mauna_loa_2022_oib_external_v0_1.parquet` — Mauna Loa external OIB cohort
- `data/processed/la_palma_2021_oib_external_v0_1.parquet` — La Palma external OIB cohort

## S4. Internal and cross-database validation

- `reports/modeling/petdb_primary4_overall_metrics.csv` — PetDB random and grouped validation metrics
- `reports/modeling/petdb_primary4_calibration_summary.csv` — Conservative probability calibration summary
- `reports/modeling/feature_bias_sensitivity_summary.csv` — Missingness and feature-panel sensitivity
- `reports/modeling/cross_database_overall_metrics.csv` — Bidirectional GEOROC-PetDB transfer metrics

## S5. Interpretability outputs

- `reports/modeling/petdb_primary4_permutation_importance_summary.csv` — Conservative grouped permutation importance
- `reports/modeling/petdb_primary4_shap_global_overall.csv` — Descriptive SHAP importance

## S6. Strict external validation

- `reports/modeling/multiclass_external_manifest.json` — External-test feature and provenance contract
- `reports/modeling/multiclass_external_metrics.csv` — External overall metrics
- `reports/modeling/multiclass_external_per_class.csv` — External class-wise metrics
- `reports/modeling/multiclass_external_by_province.csv` — Province-proxy outcomes
- `reports/modeling/multiclass_external_predictions.csv` — Sample-level external probabilities and predictions

## S7. Geological transition tests

- `reports/modeling/east_african_rift_stress_summary.csv` — Whole-province East African Rift affinity summary
- `reports/modeling/east_african_rift_stress_predictions.parquet` — East African Rift sample-level probabilities
- `reports/modeling/multiclass_external_backarc_summary.csv` — Unscored Vate Trough mechanism test
- `reports/modeling/la_palma_oib_external_summary.csv` — La Palma eruption-state transfer summary

## S8. External-source audits

- `reports/data_quality/georoc_big_pine_cib_external_audit.json` — Big Pine independence and chemistry audit
- `reports/data_quality/georoc_costa_rica_arc_external_audit.json` — Costa Rica independence and chemistry audit
- `reports/data_quality/smar_morb_external_audit.json` — South Mid-Atlantic Ridge source and chemistry audit
- `reports/data_quality/la_palma_oib_external_audit.json` — La Palma source and chemistry audit
- `reports/data_quality/mauna_loa_oib_external_audit.json` — Mauna Loa source and chemistry audit
- `reports/data_quality/swir_publication_link_audit_2026-07-15.md` — SWIR journal-link search and provisional-status decision

## S9. Manuscript compliance

- `reports/manuscript/nsr_compliance_v1.json` — NSR length, reference and display-item checks

## Reproduction gates

1. Run `python src/test_data_contract.py` and require the final contract-passed message.
2. Run the external validation only if its inputs or model code changed.
3. Run `src/validate_nsr_manuscript.py` after any numerical or structural manuscript edit.
4. Rebuild this inventory after any frozen artifact changes so checksums remain authoritative.
