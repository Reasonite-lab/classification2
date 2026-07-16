from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "04_petdb_morb_raw_audit.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"]["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb["metadata"]["language_info"] = {"name": "python", "version": "3.12"}

nb["cells"] = [
    nbf.v4.new_markdown_cell(
        """# PetDB MORB Raw Export Audit

## tl;dr

The fixed PetDB 2.0 export contains **22,680** analysis rows, **1,176** columns, **7,668** unique Sample URLs, and **431** source citations. SHA-256 and MD5 are verified. There are **32** exact duplicate rows. The web preview overstated the delivered file by 106 rows and 53 samples.

`Spreading Center` is an inclusive, not exclusive, filter. The main and metadata tables have **100% row agreement in normalized sample name and coordinates**. This verified row linkage identifies **7,557** Sample URLs with only `SPREADING CENTER` metadata and **111** with mixed or secondary tectonic tags. The latter group is not eligible for the high-purity MORB cohort.

A row-order-free name-linkage sensitivity check yields 7,511 exclusive, 106 mixed, and 51 unknown URLs, but 160 normalized names are reused across 335 Sample URLs. It is retained as a conservative sensitivity result rather than the primary linkage because it conflates unrelated samples with duplicate names.

Sample-level coverage of the 29 target geochemical variables is about **16% to 65%**. Modeling therefore requires sample-level aggregation, explicit missingness handling, and feature-set sensitivity checks. The bundle text says CC BY 4.0 while the browser terms say CC BY-SA 4.0; the project follows the stricter **CC BY-SA 4.0** policy and preserves source-publication citations."""
    ),
    nbf.v4.new_markdown_cell(
        """## Context & Methods

This notebook is the executable acceptance record for the PetDB MORB export. It recomputes file hashes, table grain, duplicate rates, filter consistency, target-feature coverage, and numeric validity.

### Key Assumptions

- One main-table row is an analysis or analytical batch, not an independent rock sample.
- `Sample URL` is the most stable available entity identifier; names have many-to-many mappings.
- Tectonic metadata use row alignment only after all 22,680 rows match on normalized sample name and both coordinates.
- A normalized-name-only linkage is retained as a conservative sensitivity check.
- The high-purity MORB cohort requires a tectonic-tag set exactly equal to `SPREADING CENTER`.
- Raw sparsity is retained and documented rather than hidden by imputation."""
    ),
    nbf.v4.new_code_cell(
        """from pathlib import Path
import json
import runpy
import pandas as pd

cwd = Path.cwd().resolve()
ROOT = cwd if (cwd / 'src' / 'profile_petdb_morb.py').exists() else cwd.parent
assert (ROOT / 'src' / 'profile_petdb_morb.py').exists(), ROOT
ROOT"""
    ),
    nbf.v4.new_markdown_cell(
        """## Data

Source: fixed EarthChem/PetDB 2.0 export `earthchem_export_jul14_26_14620.zip`, downloaded 2026-07-14 with Whole Rock, Basalt, and Spreading Center filters."""
    ),
    nbf.v4.new_code_cell(
        """runpy.run_path(str(ROOT / 'src' / 'profile_petdb_morb.py'), run_name='__main__')

quality_dir = ROOT / 'reports' / 'data_quality'
profile = json.loads((quality_dir / 'petdb_morb_raw_profile.json').read_text(encoding='utf-8'))
coverage = pd.read_csv(quality_dir / 'petdb_morb_feature_coverage.csv')"""
    ),
    nbf.v4.new_markdown_cell("## Results"),
    nbf.v4.new_code_cell(
        """summary = pd.DataFrame(
    [
        ('Archive bytes', profile['archive']['bytes']),
        ('Analysis rows', profile['main_analysis_table']['rows']),
        ('Unique Sample URLs', profile['main_analysis_table']['unique_sample_urls']),
        ('Exclusive Spreading Center Sample URLs', profile['sample_metadata_table']['exclusive_spreading_center_sample_urls']),
        ('Mixed/secondary-setting Sample URLs', profile['sample_metadata_table']['mixed_or_secondary_setting_sample_urls']),
        ('Unknown-setting Sample URLs', profile['sample_metadata_table']['unknown_setting_sample_urls']),
        ('Unique citation URLs', profile['main_analysis_table']['unique_citation_urls']),
        ('Columns', profile['header']['columns']),
        ('Exact duplicate rows', profile['main_analysis_table']['exact_duplicate_rows']),
    ],
    columns=['check', 'value'],
)
summary"""
    ),
    nbf.v4.new_code_cell(
        """assert profile['archive']['hashes_verified'] is True
assert profile['header']['all_target_features_present'] is True
assert profile['main_analysis_table']['rows'] == 22680
assert profile['main_analysis_table']['unique_sample_urls'] == 7668
assert profile['sample_metadata_table']['row_aligned_name_match_count'] == 22680
assert profile['sample_metadata_table']['row_aligned_coordinate_match_count'] == 22680
assert profile['sample_metadata_table']['exclusive_spreading_center_sample_urls'] == 7557
assert profile['sample_metadata_table']['mixed_or_secondary_setting_sample_urls'] == 111
assert profile['sample_metadata_table']['unknown_setting_sample_urls'] == 0
assert profile['sample_metadata_table']['name_linkage_sensitivity']['exclusive_spreading_center_sample_urls'] == 7511
assert profile['sample_metadata_table']['name_linkage_sensitivity']['sample_urls_affected_by_ambiguous_names'] == 335
summary"""
    ),
    nbf.v4.new_code_cell(
        """tectonic_counts = pd.Series(
    profile['sample_metadata_table']['tectonic_setting_counts'],
    name='analysis_rows',
).rename_axis('tectonic_setting').reset_index()
tectonic_counts"""
    ),
    nbf.v4.new_code_cell(
        """coverage_view = coverage[
    ['feature', 'row_non_null_pct', 'sample_non_null_pct', 'numeric_invalid', 'min', 'max']
].sort_values('sample_non_null_pct', ascending=False)
coverage_view"""
    ),
    nbf.v4.new_code_cell(
        """silica_check = pd.Series(
    {
        'numeric_SiO2_rows': profile['main_analysis_table']['silica_numeric_rows'],
        'SiO2_40_55_rows': profile['main_analysis_table']['silica_40_55_rows'],
        'pct_of_numeric_in_range': profile['main_analysis_table']['silica_40_55_pct_of_numeric'],
        'ui_minus_exported_rows': profile['ui_export_summary']['row_difference'],
        'ui_minus_unique_sample_urls': profile['ui_export_summary']['sample_difference'],
    },
    name='value',
).to_frame()
silica_check"""
    ),
    nbf.v4.new_markdown_cell(
        """## Takeaways

1. The archive is intact, and all 29 target geochemical variables plus required provenance fields are present.
2. The web preview is not an authoritative volume count; manuscript claims must use the delivered CSV.
3. PetDB tectonic settings are multi-valued. The 111 mixed-tag Sample URLs require exclusion or separate sensitivity analysis.
4. Name-only linkage is over-conservative because duplicate sample names occur across 335 URLs; its 7,511-sample result is retained as a linkage sensitivity check.
5. Major-element sample coverage peaks near 65%; HFSE and REE coverage is roughly 16% to 52%. A wide export is not a complete matrix.
6. The 32 exact duplicates and 354 duplicate composite analysis keys require review before sample aggregation.
7. The project applies the stricter CC BY-SA 4.0 interpretation and will generate a secondary bibliography for all 431 contributing citations."""
    ),
]

OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(OUT)
