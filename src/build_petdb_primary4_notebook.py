from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "06_petdb_primary4_validation.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"]["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb["metadata"]["language_info"] = {"name": "python", "version": "3.12"}
nb["cells"] = [
    nbf.v4.new_markdown_cell(
        """# PetDB Primary Four-Class Validation

## tl;dr

Four hash-verified PetDB exports now provide MORB, ARC, OIB, and CIB under the same Whole Rock + Basalt contract. Sample-URL aggregation yields **13,718** rows, including **6,148** broad-ready samples. The OIB export lacks the entire Th column, so the primary validation excludes Th and uses 21 features observed in every class.

The balanced random forest reaches macro-F1 **0.758** under conservative citation-overlap grouping and **0.656** under location grouping, compared with **0.913** under random stratification. CIB location-held-out recall is only **0.136**. Missingness alone remains predictive, so the result is an internal PetDB four-class estimate, not yet a cross-database external validation."""
    ),
    nbf.v4.new_markdown_cell(
        """## Audit assumptions

- Only clean exports listed in `config/petdb_primary4_exports.yaml` are used.
- The collided/headerless S3 bundle is retained as evidence and excluded.
- Main and metadata rows are linked only after 100% normalized-name and coordinate agreement.
- Exact duplicate analyses are removed before median aggregation by stable Sample URL.
- Citation-overlap grouping is the primary literature-leakage control; location grouping is the geographic stress test.
- A feature absent from an entire class export cannot enter the primary model."""
    ),
    nbf.v4.new_code_cell(
        """from pathlib import Path
import json
import pandas as pd
import yaml

cwd = Path.cwd().resolve()
ROOT = cwd if (cwd / 'config' / 'petdb_primary4_exports.yaml').exists() else cwd.parent
assert (ROOT / 'config' / 'petdb_primary4_exports.yaml').exists(), ROOT
ROOT"""
    ),
    nbf.v4.new_code_cell(
        """exports = yaml.safe_load((ROOT / 'config' / 'petdb_primary4_exports.yaml').read_text(encoding='utf-8'))
profile = json.loads((ROOT / 'reports' / 'data_quality' / 'petdb_primary4_processing_profile.json').read_text(encoding='utf-8'))
flow = pd.read_csv(ROOT / 'reports' / 'data_quality' / 'petdb_primary4_qc_flow.csv')
overall = pd.read_csv(ROOT / 'reports' / 'modeling' / 'petdb_primary4_overall_metrics.csv')
per_class = pd.read_csv(ROOT / 'reports' / 'modeling' / 'petdb_primary4_per_class_metrics.csv')
missingness = pd.read_csv(ROOT / 'reports' / 'modeling' / 'petdb_primary4_missingness_summary.csv')
manifest = json.loads((ROOT / 'reports' / 'modeling' / 'petdb_primary4_validation_manifest.json').read_text(encoding='utf-8'))"""
    ),
    nbf.v4.new_markdown_cell("## Export and cohort audit"),
    nbf.v4.new_code_cell(
        """export_table = pd.DataFrame(exports['exports'])[
    ['label', 'export_id', 'api_count_at_submission', 'delivered_rows', 'archive_bytes', 'sha256']
]
assert profile['combined']['sample_rows'] == 13718
assert profile['combined']['model_ready_broad'] == 6148
assert profile['combined']['model_ready_immobile'] == 4005
export_table"""
    ),
    nbf.v4.new_code_cell(
        """flow.pivot(index='label', columns='stage', values='n').loc[
    ['ARC', 'CIB', 'MORB', 'OIB'],
    ['unique_sample_urls', 'exclusive_target_setting', 'strict_candidate', 'model_ready_broad', 'model_ready_immobile'],
]"""
    ),
    nbf.v4.new_markdown_cell("## Grouped four-class validation"),
    nbf.v4.new_code_cell(
        """assert manifest['excluded_structurally_absent_features'] == ['TH(PPM)']
primary = overall[overall['model'].eq('random_forest_balanced')][
    ['cv_strategy', 'accuracy', 'balanced_accuracy', 'macro_f1', 'log_loss']
].sort_values('macro_f1', ascending=False)
primary"""
    ),
    nbf.v4.new_code_cell(
        """per_class[
    per_class['model'].eq('random_forest_balanced')
    & per_class['cv_strategy'].isin(['citation_overlap_conservative', 'location_root_grouped'])
][['cv_strategy', 'label', 'precision', 'recall', 'f1-score', 'support']]"""
    ),
    nbf.v4.new_markdown_cell("## Missingness bias gate"),
    nbf.v4.new_code_cell(
        """missingness[
    missingness['cv_strategy'].eq('citation_overlap_conservative')
][['variant', 'representation', 'balanced_accuracy_mean', 'macro_f1_mean', 'log_loss_mean']]"""
    ),
    nbf.v4.new_markdown_cell(
        """## Takeaways

1. The initial OIB/CIB S3 collision was caught by package-internal query and header checks; the affected bundle is not analyzed.
2. PetDB now supplies all four labels, removing the earlier deterministic mapping between database source and MORB.
3. Random stratification remains optimistic. Citation-overlap and location grouping are the decision-relevant estimates.
4. Th is excluded from the primary model because it is structurally absent from OIB. The archived 22-feature score is a bias sensitivity result.
5. MORB is the most stable class; CIB geographic transfer is currently inadequate.
6. Cross-database GEOROC–PetDB validation and four-class interpretability remain required before final submission."""
    ),
]

OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(OUT)
