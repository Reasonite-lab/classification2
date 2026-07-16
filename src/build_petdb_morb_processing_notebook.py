from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "05_petdb_morb_processing_and_source_gate.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"]["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb["metadata"]["language_info"] = {"name": "python", "version": "3.12"}

nb["cells"] = [
    nbf.v4.new_markdown_cell(
        """# PetDB MORB Sample Processing and Source-Confounding Gate

## tl;dr

The fixed PetDB export was reduced from **22,680 analysis rows** to **7,668 stable Sample URLs** after removing 32 exact duplicate rows and aggregating repeated measurements by median. Verified row linkage identifies **7,557** exclusive `SPREADING CENTER` URLs. Applying the same basalt scope used for GEOROC (SiO2 40–55 wt%, complete 85–105 wt% major-element total, trace provenance) yields **4,249 strict candidates**; **3,650** meet the 22-feature broad contract and **2,302** meet the 19-feature immobile contract.

Twenty broad-ready samples carry severe alteration tags, and 20 fail deliberately broad component-range checks. These are sensitivity cohorts, not silent primary exclusions. A citation-component-held-out source diagnostic remains concerning even after harmonizing total iron as FeO equivalent: missingness alone distinguishes GEOROC from PetDB with mean balanced accuracy **0.658** (ROC-AUC **0.668**), while geochemical values reach **0.906** balanced accuracy. The value result cannot separate true MORB geology from database effects because PetDB currently supplies only MORB and GEOROC supplies only ARC/CIB/OIB.

**Decision:** the sample-level MORB cohort is suitable for continued processing and sensitivity work, but a naive pooled four-class score is not suitable as a source-robust paper result. Overlapping tectonic classes across databases are required."""
    ),
    nbf.v4.new_markdown_cell(
        """## Context & Methods

This executable audit answers whether the PetDB export can enter the final ARC–CIB–OIB–MORB study without confusing database provenance with tectonic setting.

### Key Assumptions

- The statistical entity is a stable PetDB `Sample URL`, not an analysis row.
- Main and metadata rows are linked only after all 22,680 normalized names and coordinate pairs agree.
- Exact duplicate rows are removed using all 1,176 raw columns; other repeated measurements are retained and aggregated by sample median.
- Total iron is harmonized to FeO equivalent with row precedence: reported FeOT; 0.8998 × Fe2O3T; FeO + 0.8998 × Fe2O3.
- The primary cohort requires exclusive `SPREADING CENTER`, SiO2 40–55 wt%, complete major elements, an 85–105 wt% calculated total, and a citation.
- Severe alteration and broad component-range checks define sensitivity cohorts only.
- Source classification uses source-prefixed citation-overlap components in five-fold grouped validation."""
    ),
    nbf.v4.new_code_cell(
        """from pathlib import Path
import json
import sys
import pandas as pd

cwd = Path.cwd().resolve()
ROOT = cwd if (cwd / 'src' / 'build_petdb_morb_dataset.py').exists() else cwd.parent
assert (ROOT / 'src' / 'build_petdb_morb_dataset.py').exists(), ROOT
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))
ROOT"""
    ),
    nbf.v4.new_markdown_cell(
        """## Data

- Raw source: `data/raw/petdb_morb/earthchem_export_jul14_26_14620.zip`
- Sample outputs: `data/processed/petdb_morb_v0_1.parquet` and portable `petdb_morb_v0_1.csv.gz`
- GEOROC comparison: `data/processed/georoc_primary3_v0_1.parquet`
- Export date: 2026-07-14; processing audit date: 2026-07-15."""
    ),
    nbf.v4.new_code_cell(
        """# Rebuild commands (run from the project root in the configured project Python):
# python src/build_petdb_morb_dataset.py
# python src/run_petdb_source_confounding_diagnostic.py

quality_dir = ROOT / 'reports' / 'data_quality'
modeling_dir = ROOT / 'reports' / 'modeling'
profile = json.loads((quality_dir / 'petdb_morb_processing_profile.json').read_text(encoding='utf-8'))
samples = pd.read_csv(ROOT / 'data' / 'processed' / 'petdb_morb_v0_1.csv.gz', low_memory=False)
qc_flow = pd.read_csv(quality_dir / 'petdb_morb_qc_flow.csv')
coverage = pd.read_csv(quality_dir / 'petdb_morb_sample_feature_coverage.csv')
source_summary = pd.read_csv(modeling_dir / 'petdb_source_confounding_summary.csv')
missingness = pd.read_csv(modeling_dir / 'petdb_source_missingness_prevalence.csv')"""
    ),
    nbf.v4.new_markdown_cell("## Results"),
    nbf.v4.new_code_cell(
        """assert profile['exact_duplicate_rows_removed'] == 32
assert profile['sample_urls'] == 7668
assert profile['row_linkage']['normalized_name_matches'] == 22680
assert profile['row_linkage']['coordinate_matches'] == 22680
assert profile['tectonic_linkage']['exclusive_spreading_center'] == 7557
assert profile['cohorts']['strict_candidate'] == 4249
assert profile['cohorts']['model_ready_broad'] == 3650
assert profile['cohorts']['model_ready_immobile'] == 2302
assert samples['record_id'].is_unique and samples['SAMPLE_URL'].is_unique
qc_flow"""
    ),
    nbf.v4.new_code_cell(
        """coverage.loc[
    coverage['cohort'].eq('model_ready_broad'),
    ['feature', 'non_null', 'non_null_pct', 'median', 'min', 'max'],
].sort_values('non_null_pct', ascending=False).reset_index(drop=True)"""
    ),
    nbf.v4.new_code_cell(
        """sample_diagnostics = pd.DataFrame(
    {
        'metric': [
            'Broad-ready samples',
            'Broad-ready without severe alteration',
            'Broad-ready passing component-range sensitivity',
            'Median analysis rows per broad-ready sample',
            'Broad-ready samples with any discordant feature',
            'Broad-ready citation sets',
            'Broad-ready location groups',
        ],
        'value': [
            int(samples['model_ready_broad'].sum()),
            int(samples['model_ready_broad_no_severe_alteration'].sum()),
            int(samples['model_ready_broad_major_range_sensitivity'].sum()),
            float(samples.loc[samples['model_ready_broad'], 'analysis_row_n'].median()),
            int((samples.loc[samples['model_ready_broad'], 'discordant_feature_n'] > 0).sum()),
            int(samples.loc[samples['model_ready_broad'], 'citation_set'].nunique()),
            int(samples.loc[samples['model_ready_broad'], 'location_root'].nunique()),
        ],
    }
)
sample_diagnostics"""
    ),
    nbf.v4.new_code_cell(
        """source_view = source_summary[
    ['input_spec', 'balanced_accuracy_mean', 'balanced_accuracy_sd', 'roc_auc_mean',
     'georoc_recall_mean', 'petdb_recall_mean']
].sort_values('balanced_accuracy_mean')
source_view"""
    ),
    nbf.v4.new_code_cell(
        """missingness[
    ['feature', 'GEOROC', 'PETDB', 'petdb_minus_georoc_missing_pct']
].assign(abs_difference=lambda frame: frame['petdb_minus_georoc_missing_pct'].abs()) \
 .sort_values('abs_difference', ascending=False).head(12)"""
    ),
    nbf.v4.new_markdown_cell(
        """## Takeaways

1. The sample aggregation is reproducible and preserves provenance: 22,648 non-duplicate analyses map to 7,668 unique Sample URLs and 431 citation records.
2. Verified row linkage is preferable to name-only linkage because 160 normalized names are reused across 335 URLs. The name-only 7,511-sample result remains a conservative sensitivity check.
3. The strict cohort contains 4,249 samples; 3,650 broad-ready and 2,302 immobile-ready samples provide enough MORB observations for substantive modeling.
4. Alteration and gross major-component ranges affect only small sensitivity subsets (20 broad-ready samples each), while replicate discordance is retained as metadata rather than hidden by aggregation.
5. Database provenance remains predictable from coverage patterns even after iron harmonization. Value-based source separation is stronger but is inseparable from real MORB-vs-non-MORB geology in the current design.
6. The next data action is to obtain PetDB ARC/OIB/CIB counterparts with the same export and processing contract, or an independent non-PetDB MORB cohort. Until then, any pooled four-class result is diagnostic only, not the paper's source-robust endpoint."""
    ),
]

OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(OUT)
