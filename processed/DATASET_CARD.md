# GEOROC Primary-3 v0.1 Dataset Card

## Purpose

This is a provisional, auditable three-class dataset for basaltic volcanic-rock tectonic discrimination. It is an intermediate research artifact, not the final four-class dataset. MORB is intentionally absent until an explicit mid-ocean-ridge source is integrated.

## Source

- GEOROC Compilation: Rock Types, DOI `10.25625/2JETOA`, files generated 2026-06-01.
- GEOROC Compilation: Sample Metadata citation table, DOI `10.25625/4EZ7ID`, generated 2024-12-01.
- Both sources are distributed under CC BY-SA 4.0. Derived redistributable datasets must retain attribution and share-alike terms.

## Row selection

Records must be whole-rock analyses of volcanic material, have a rock name containing basalt, and contain SiO2 between 40 and 55 wt%. The primary-3 cohort includes only GEOROC raw settings mapped to ARC, OIB, or broad continental intraplate basalt (CIB). Records require a complete calculated major-element total between 85 and 105 wt%.

## Labels

- `ARC`: raw setting `CONVERGENT MARGIN`
- `OIB`: raw setting `OCEAN ISLAND`
- `CIB`: raw settings `INTRAPLATE VOLCANICS`, `CONTINENTAL FLOOD BASALT`, or `RIFT VOLCANICS`

The mapping is broad by design. Fine raw settings remain available for sensitivity tests. `SUBMARINE RIDGE` is not treated as MORB because the category includes aseismic ridges and plateau-like features.

## Quality controls

- Exact source files are pinned by MD5.
- Trace-element values less than or equal to zero are treated as missing in the processed table.
- Citation sets associated with more than one primary label are excluded from `model_ready_broad`.
- At least 12 of 22 broad features must be observed for the broad baseline cohort.
- Location, citation, rock name, sample name, source file, and labels are forbidden model features.

## Known limitations

- The database is literature-compiled and geographically imbalanced.
- Missingness varies by class and publication era.
- The citation table predates the rock files by 18 months; 86 used Citation-IDs are not yet resolved in the downloaded bibliography snapshot.
- Multiple analyses from the same locality or paper are correlated; random row-level validation is optimistic.
- The broad ARC and CIB classes contain important petrogenetic substructure.
- External MORB data and a four-class external validation are still pending.
- Location-root validation is substantially harder than citation-grouped validation; OIB geographic transfer is currently the weakest class-level result.
- Missingness patterns contain weak class information, but adding explicit missingness indicators did not improve the value-based model on the common cohort.

## Recommended use

Use `citation_set` for literature-grouped validation and `location_root` for geographic stress testing. Treat `citation_overlap_component` as a conservative sensitivity analysis. Do not present random-stratified scores as the primary generalization estimate. The current primary baseline should omit explicit missingness indicators and should report both the broad and immobile feature sets.
