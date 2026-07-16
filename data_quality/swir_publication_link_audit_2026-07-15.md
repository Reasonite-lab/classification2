# SWIR publication-link audit

**Audit date:** 2026-07-15  
**Repository cohort:** Figshare `10.6084/m9.figshare.25295671.v1`  
**Later supporting-data record:** Mendeley Data `10.17632/fntfnf92tg.1`

## Question

Does the provisional Southwest Indian Ridge cohort used in the strict external test have a verifiable linked journal article DOI?

## Evidence checked

1. The Figshare workbook and its repository metadata already frozen in the project.
2. The Mendeley Data record at `https://data.mendeley.com/datasets/fntfnf92tg`, published 2025-12-15 by Chuanwei Zhu. It describes itself as supporting data for an article entitled *Cadmium isotope compositions of basalts from the East Pacific Rise and Southwest Indian Ridge: Implications for magma differentiation and mantle heterogeneity*, but the indexed record exposes only the dataset DOI `10.17632/fntfnf92tg.1` and no journal DOI.
3. Exact-title and author/title searches for the stated article title, the Figshare identifier, the Mendeley identifier and the title phrase “magma differentiation and mantle heterogeneity”. Searches returned the Mendeley Data record but no matching journal article.
4. Other published SWIR articles returned by the searches were rejected as links because their titles, authors or sample scopes did not establish identity with the Zhu repository workbook.

## Decision

No journal article DOI can be verified for this repository cohort as of the audit date. The cohort therefore remains **provisional_external_morb**. Its 36 SWIR samples may remain in the explicitly qualified sensitivity/external design because the repository object is persistent, CC BY 4.0 and independently audited, but the manuscript must not describe the cohort as publication-linked.

The symmetric external test retains a second, publication-primary MORB proxy: Zhong et al. (2019), South Mid-Atlantic Ridge 18.0–20.6°S, DOI `10.3390/min9110659`.

## Submission gate

Before formal submission, repeat the exact-title lookup. If no article DOI is found, either:

1. retain the SWIR cohort with the present explicit provisional caveat and provide a reviewer-facing sensitivity analysis excluding it; or
2. replace it with another publisher-primary MORB province cohort and rerun the frozen external-validation pipeline.

Do not infer a journal link from topical similarity alone.
