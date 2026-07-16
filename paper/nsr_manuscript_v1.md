# Research Article

# Geochemical memory limits machine-learning discrimination of basalt tectonic settings

**Running title:** Geochemical memory in basalt discrimination

**Bangkun Xu (author name and order to be confirmed)**^1,*

^1 **[Department, institution, city, postal code and country to be confirmed]**

*Corresponding author. E-mail: xubangkun439@gmail.com; telephone and fax: **[to be confirmed]**.

**Article type:** Research Article

**Teaser text (30 words):** This study shows that basalt classifiers transfer across end-member settings but fail systematically where inherited subduction or plume signals decouple rock chemistry from present tectonic position.

## ABSTRACT

Basalt compositions encode tectonic setting together with inherited mantle and crustal processes, making discrimination a problem of transfer rather than interpolation. We assembled 54 012 GEOROC and 6148 PetDB whole-rock basalts under auditable lithologic, iron and provenance contracts. Random-forest macro-F1 fell from 0.915 with random splitting to 0.760 when citation-overlap components were held out and 0.648 when locations were held out. A common 18-element model tested on 231 samples from two province proxies per class achieved balanced accuracy of 0.822 and macro-F1 of 0.764. Mid-ocean-ridge and arc recall were 1.000 and 0.967, whereas continental intraplate recall was 0.520. Errors were geologically structured: inherited subduction signatures shifted Big Pine rift basalts toward arcs, whereas East African rift basalts shifted toward ocean-island affinity. Tectonic discrimination is therefore most credible as an uncertainty-aware diagnosis of geochemical affinity and memory, not a universal hard label.

**Keywords:** basalt; tectonic discrimination; geochemistry; machine learning; external validation; mantle source

## INTRODUCTION

Whole-rock basalt chemistry has long been used to infer tectonic setting. Classical discrimination diagrams compress selected major and trace elements into low-dimensional boundaries intended to separate mid-ocean ridges, volcanic arcs and intraplate magmatism [1,2]. Their geological appeal is clear: incompatible-element enrichment, fluid-mobile element transfer, partial melting and fractional crystallization leave coherent multielement signals. Their statistical limits are equally important. Boundaries derived from restricted reference sets can fail when applied to new regions, and geochemical data are compositional, heterogeneous in analytical coverage and strongly clustered by publication and province [3]. Classification trees and later multivariate models expanded the usable information but did not remove these sampling and transfer problems [4].

Global repositories now make a more stringent test possible. GEOROC and PetDB combine measurements from many laboratories, times and scientific objectives [5]. These compilations are large enough for machine learning, yet samples from one paper, eruption or province are not independent. A random record split can place near-related rocks in both training and testing sets, rewarding interpolation within familiar studies rather than transfer to new geological systems. Measurement coverage is another potential shortcut: if a class has a systematically absent element, a model may learn what was measured instead of rock composition.

Previous machine-learning studies showed that multielement and isotopic data can discriminate tectonic settings [6,7], and recent reviews have emphasized the growing role of interpretable models in petrology [8]. The outstanding geological question is not whether a flexible classifier can fit a compiled database. It is whether the learned decision survives publication, province and database shifts, and whether failures correspond to recognizable petrogenetic processes. We therefore use balanced logistic regression as an auditable linear baseline and a balanced random forest as a limited nonlinear extension [9]. Permutation importance and SHAP values are used as complementary descriptions of model dependence [10], not as proof that an element uniquely causes a prediction.

Here we establish a provenance-first workflow for whole-rock basalt tectonic discrimination. We quantify optimism from random splitting, test structural missingness, compare bidirectional transfer between GEOROC and PetDB, and construct a four-class external test with two province proxies for each of arc (ARC), continental intraplate basalt (CIB), mid-ocean-ridge basalt (MORB) and ocean-island basalt (OIB). We then treat the systematic errors as geological observations. The central result is that end-member settings transfer well, whereas continental rifts and back-arc transitions expose a mismatch between present tectonic position and inherited geochemical memory.

## RESULTS AND DISCUSSION

### Data contracts separate sample chemistry from database structure

The June 2026 GEOROC Rock Types release contained 109 882 records labelled BASALT [11]. Lithologic, material, silica, major-element total, citation and label-consistency filters retained 54 012 broad-feature records: 18 083 ARC, 30 086 broad CIB and 5843 OIB. Broad CIB here combines intraplate volcanics, continental flood basalt and rift volcanics; the original fine label is retained so that this ontology can be audited rather than hidden.

Four independent PetDB whole-rock basalt exports were made for spreading centre, volcanic arc, ocean island and continental rift settings [12]. After duplicate-analysis removal, sample-level median aggregation, iron harmonization and conservative label linkage, 6148 samples remained (3650 MORB, 754 ARC, 1281 OIB and 463 CIB). The primary PetDB model used 21 chemical variables observed in every class. Th was excluded because it was absent from the entire PetDB OIB export. A 22-variable diagnostic was retained only to demonstrate that structural absence could generate a misleadingly powerful class cue. Geographic names, sample names, publications, filenames, analytical methods and all label fields were prohibited as model inputs.

### Random splits substantially overstate transfer performance

On GEOROC, the balanced random forest attained macro-F1 of 0.861 under a stratified random split, 0.747 when connected components of overlapping citation sets were held out and 0.654 when first-order locations were held out. The same pattern was sharper in the PetDB four-class task: macro-F1 decreased from 0.915 under random splitting to 0.760 under conservative citation-overlap grouping and 0.648 under location grouping (Fig. 1). The balanced logistic baseline fell from 0.822 to 0.718 and 0.646, respectively. Thus, most of the random forest's apparent advantage disappeared under spatial transfer. Added algorithmic flexibility helped within known domains but did not solve province shift.

[INSERT FIGURE 1 HERE]

The CIB class controlled much of this deterioration. In the PetDB location-held-out test, CIB recall was only 0.127, despite high random-split performance. This result is not adequately summarized as class imbalance. Continental rift basalts span lithospheric mantle, asthenosphere and plume contributions, variable degrees of extension, inherited subduction modification and crustal interaction. A single tectonic-location label therefore covers multiple compositional states.

Structural missingness was measurable but was not the main source of the grouped performance. On a common 43 005-record GEOROC subset, a missingness-only random forest produced macro-F1 of 0.289 under citation grouping and 0.258 under location grouping. Adding explicit missingness indicators to element values did not improve either split. By contrast, the PetDB all-22 diagnostic assigned strong OIB information to the entirely missing Th column, and a complete external OIB time series was then driven toward CIB. The production external contract therefore excluded Th, V, Cr and Ni and required a common, positive 18-element panel.

### Learned importance is geologically coherent but non-unique

Under conservative citation-overlap validation, permutation importance ranked TiO2 first (mean balanced-accuracy decrease 0.0918), followed by Sr (0.0633), K2O (0.0305), Zr (0.0253), MnO (0.0212), Nb (0.0211) and Y (0.0205; Fig. 2). Mean absolute SHAP importance independently placed Sr, TiO2, K2O, Nb and Y among the leading variables. The agreement is meaningful because the two methods ask different questions: permutation importance measures predictive loss on held-out groups, whereas SHAP partitions fitted predictions.

[INSERT FIGURE 2 HERE]

These elements form process-sensitive combinations rather than isolated tectonic tracers. TiO2, Nb, Zr and Y respond to source enrichment and melting degree; Sr and K2O also record slab-derived input, plagioclase behaviour, differentiation and crustal interaction [13]. Their joint importance is consistent with petrology but cannot uniquely separate these processes from whole-rock chemistry alone. Model explanations should therefore be read as hypotheses about source and differentiation, to be tested with isotope, mineral and age information.

Probability quality also depended on validation design. In the conservative PetDB split, the random forest had a multiclass Brier score of 0.236 and top-label expected calibration error of 0.075 [14]. These values are adequate for relative affinity but not for literal posterior probabilities. A 0.8 model probability is consequently reported as strong resemblance to a training class, not as an 80% geological truth.

### Database transfer exposes ontology and sampling asymmetry

Three-class transfer between GEOROC and PetDB was asymmetric even when both models used the same 21 variables and the target database was never used for fitting (Fig. 3). Under the broad CIB ontology, GEOROC-to-PetDB macro-F1 was 0.804, whereas PetDB-to-GEOROC macro-F1 was 0.484. Restricting GEOROC CIB to rift volcanics increased the latter to 0.619 but did not eliminate the asymmetry; the corresponding GEOROC-to-PetDB value was 0.788.

[INSERT FIGURE 3 HERE]

This directionality is expected from coverage. GEOROC supplied a much larger and broader source domain, whereas PetDB CIB was limited to continental-rift queries. Ontology alignment improved transfer because it removed some flood-basalt and general intraplate records, but the remaining gap shows that identical class names and feature columns do not guarantee exchangeable geological populations.

### A symmetric external test separates end members from continental-rift memory

The strict external test comprised 231 samples and two province proxies for every class (Table 1). No external publication or exact sample name occurred in the PetDB training table. The common 18-element balanced random forest achieved accuracy of 0.745, balanced accuracy of 0.822 and macro-F1 of 0.764. The logistic baseline achieved 0.442, 0.576 and 0.486, respectively (Fig. 4). The nonlinear gain is large enough to justify a random forest as the main classifier, but the model remains modest compared with deep or stacked ensembles and preserves direct permutation and SHAP audits.

[INSERT TABLE 1 HERE]

| Class | Province proxy | Source | n | Random-forest recall |
|---|---|---|---:|---:|
| ARC | South New Hebrides arc front | PANGAEA.922011 | 12 | 1.000 |
| ARC | Costa Rica volcanic front | GEOROC 2JETOA | 18 | 0.944 |
| CIB | Rio Grande Rift/Jemez Lineament | Rowe et al. Table S1 | 28 | 0.464 |
| CIB | Big Pine volcanic field | GEOROC 2JETOA | 70 | 0.543 |
| MORB | Southwest Indian Ridge | Figshare 25295671 | 36 | 1.000 |
| MORB | South Mid-Atlantic Ridge, 18.0-20.6°S | Zhong et al. Table S1 | 12 | 1.000 |
| OIB | Mauna Loa 2022 eruption | Rhoads et al. Table S1 | 16 | 1.000 |
| OIB | La Palma 2021 eruption | Day et al. Table S1 | 39 | 0.718 |

**Table 1.** Strict external whole-rock basalt cohorts. The two-province design is a minimum symmetry check, not enough clusters for stable province-level confidence intervals. The Southwest Indian Ridge repository cohort remains provisional because a linked journal article was not established; the second MORB cohort is publisher-primary and publication-linked.

[INSERT FIGURE 4 HERE]

MORB recall was 1.000 in both ridge provinces, and ARC recall was 1.000 in South New Hebrides and 0.944 in Costa Rica. OIB transfer depended on eruptive state: Mauna Loa recall was 1.000, whereas La Palma recall was 0.718. At La Palma, recall increased from 0.421 for days 1-20 fractionated tephrites to 1.000 for post-day-20 primitive basanites. The label did not change during the eruption; the observed change demonstrates that differentiation and recharge can move samples within the classifier's affinity space [21].

CIB was the persistent weak class. Rio Grande Rift recall was 0.464 and Big Pine recall was 0.543; most Big Pine errors were ARC predictions. This is a geological, not merely numerical, failure. Rio Grande Rift magmatism retains signatures of hydrated and metasomatized lithospheric mantle related to earlier Farallon subduction [18]. Big Pine basalts require heterogeneous mantle sources and, in individual eruptions, variable crustal assimilation and crystal cargo [19,20]. A classifier trained on current tectonic categories therefore detects inherited slab-related or crustal signals and assigns them to the arc compositional domain.

### Transitional provinces should be reported as affinities, not forced labels

An entire East African Rift province was removed from training before prediction. Among 347 samples labelled CIB in the database, the random forest assigned mean OIB probability of 0.609 and mean CIB probability of 0.075; 307 samples were predicted as OIB. This strong shift is compatible with variable contributions from plume-influenced asthenosphere and enriched lithosphere during continental breakup [15-17], but it is not evidence that the database label is wrong. It shows that a tectonic-position label and source-affinity signal can legitimately diverge (Fig. 5).

[INSERT FIGURE 5 HERE]

A second transition test used 21 Vate Trough samples from initial back-arc rifting. They were deliberately excluded from scored ARC or MORB accuracy. The random forest predicted 76.2% as MORB, 14.3% as ARC and 4.8% each as CIB and OIB. The MORB-dominant result is consistent with decompression melting and weak slab contribution during early back-arc opening [22]. Treating these samples as ordinary classification errors would discard the process signal that makes them scientifically informative.

Together, the external failures define an interpretable continuum. Some continental rifts shift toward ARC because lithospheric mantle remembers subduction; others shift toward OIB because plume or enriched asthenospheric contributions dominate. Early back-arc basins shift toward MORB as decompression melting becomes important. This structure argues for a layered output: present tectonic position, multielement source affinity, transition state and distance outside the training domain. Additional algorithmic complexity is justified only if it improves these geological components under province-held-out testing.

## CONCLUSION

Multi-element machine learning captures transferable basalt signals, but its credibility is controlled more by geological validation than by random-split accuracy. In PetDB, random-forest macro-F1 decreased from 0.915 under random splitting to 0.760 under citation-overlap grouping and 0.648 under location grouping. In a strict 231-sample external test with two province proxies per class, the common 18-element random forest reached balanced accuracy of 0.822 and macro-F1 of 0.764. It transferred nearly perfectly to MORB and arc end members, but continental intraplate recall was only 0.520.

The errors reveal why. Basalt chemistry records source enrichment, inherited metasomatism, melting and differentiation as well as present tectonic setting. Big Pine and Rio Grande rift basalts can retain arc-like memory; East African Rift basalts can show OIB affinity; Vate Trough back-arc basalts can approach MORB. TiO2, Sr, K2O, Nb and Y provide a robust multielement basis for these distinctions, but no importance method makes them unique causal tracers. The appropriate product is therefore not a universal tectonic label. It is an auditable, uncertainty-aware diagnosis that reports class affinity, domain shift and geological transition, with logistic regression as a transparent baseline and a random forest providing limited, necessary nonlinearity.

## MATERIALS AND METHODS

GEOROC Rock Types v2026-06 [11] and four PetDB 2.0 whole-rock basalt exports [12] were frozen with source queries and checksums. Records were retained when material and rock-name fields indicated whole-rock volcanic basalt, SiO2 was 40-55 wt%, a harmonized major-element total was 85-105 wt% (90-105 wt% for strict external cohorts), and the tectonic label was unique. Total iron was prioritized as reported FeOT, then calculated as 0.8998 × Fe2O3T or FeO + 0.8998 × Fe2O3. Duplicate analyses were removed and remaining analyses were aggregated to sample medians.

Positive concentrations were log10 transformed. Missing values were median-imputed within each training fold. The PetDB internal model used 21 variables observed in every class. External validation used 18 variables common to all strict cohorts: SiO2, TiO2, Al2O3, FeOT, CaO, MgO, MnO, K2O, Na2O, P2O5, Rb, Sr, Y, Zr, Nb, La, Ce and Nd. No metadata or missingness indicators entered the main model.

Balanced logistic regression and class-balanced random forests were evaluated by five-fold stratified random, citation-set grouped, citation-overlap-component grouped and location-grouped splits. Metrics were calculated from out-of-fold predictions. Permutation importance was measured on citation-overlap-held-out folds; descriptive SHAP values used a fitted 160-tree forest. Calibration used multiclass Brier score and ten-bin top-label expected calibration error without post-test recalibration.

External cohorts were selected before prediction, required complete positive common18 values, and were audited against PetDB publications and exact sample names. The 231 scored samples represented two province proxies per class. East African Rift and Vate Trough samples were analysed separately as whole-province or mechanism pressure tests. Code, fold assignments, prediction tables, hashes and detailed inclusion/exclusion audits are retained in the project archive.

## DATA AND SOFTWARE AVAILABILITY

GEOROC source data are available under DOI 10.25625/2JETOA and sample metadata under DOI 10.25625/4EZ7ID. PetDB source queries and EarthChem export archives are preserved with checksums. Publisher and repository sources for each external cohort are listed in Table 1 and the Supplementary Data. All processing scripts, label mappings, fold assignments, out-of-fold probabilities, external predictions, manifests and file hashes are frozen in the project workspace. A public archival DOI for the reproducibility package will be inserted before submission; no claim in this draft depends on an unavailable fitted model object.

## SUPPLEMENTARY DATA

Supplementary Data will include the full data dictionary, provenance manifest, exclusion ledger, label ontology, class-wise missingness matrix, grouped fold assignments, complete metrics, calibration tables, permutation and SHAP outputs, external sample audit, province-level prediction table and checksums for all source files.

## ACKNOWLEDGEMENTS

The authors acknowledge the investigators and data curators who contributed measurements to GEOROC, PetDB, PANGAEA, Figshare, Mendeley Data and the cited publisher supplements. **[Additional personal or institutional acknowledgements to be confirmed.]**

## FUNDING

**[Funding statement and grant numbers to be confirmed before submission.]**

## AUTHOR CONTRIBUTIONS

Bangkun Xu: Conceptualization, Methodology, Investigation, Data curation, Formal analysis, Visualization, Writing - original draft, Writing - review and editing. **[Roles and names of any additional authors must be confirmed before submission.]**

## CONFLICT OF INTEREST

Conflict of interest statement. None declared.

## REFERENCES

1. Pearce JA and Cann JR. Tectonic setting of basic volcanic rocks determined using trace element analyses. *Earth Planet Sci Lett* 1973; 19: 290-300.

2. Wood DA. The application of a Th-Hf-Ta diagram to problems of tectonomagmatic classification and to establishing the nature of crustal contamination of basaltic lavas. *Earth Planet Sci Lett* 1980; 50: 11-30.

3. Vermeesch P. Tectonic discrimination diagrams revisited. *Geochem Geophys Geosyst* 2006; 7: Q06017.

4. Vermeesch P. Tectonic discrimination of basalts with classification trees. *Geochim Cosmochim Acta* 2006; 70: 1839-48.

5. Lehnert KA, Su Y and Langmuir CH et al. A global geochemical database structure for rocks. *Geochem Geophys Geosyst* 2000; 1: 1012.

6. Petrelli M and Perugini D. Solving petrological problems through machine learning: the study case of tectonic discrimination using geochemical and isotopic data. *Contrib Mineral Petrol* 2016; 171: 81.

7. Ueki K, Hino H and Kuwatani T. Geochemical discrimination and characteristics of magmatic tectonic settings: a machine-learning-based approach. *Geochem Geophys Geosyst* 2018; 19: 1327-47.

8. Petrelli M, Caricchi L and Perugini D. Machine learning in petrology: state-of-the-art and future perspectives. *J Petrol* 2024; 65: egae036.

9. Breiman L. Random forests. *Mach Learn* 2001; 45: 5-32.

10. Lundberg SM and Lee S-I. A unified approach to interpreting model predictions. *Adv Neural Inf Process Syst* 2017; 30: 4765-74.

11. GEOROC Data Group. GEOROC compilation: Rock Types, version 2026-06. 2026, doi: 10.25625/2JETOA.

12. EarthChem. PetDB 2.0 whole-rock basalt query exports for spreading centre, volcanic arc, ocean island and continental rift settings. 2026. https://www.earthchem.org/petdb (14 July 2026, date last accessed).

13. Sun S-S and McDonough WF. Chemical and isotopic systematics of oceanic basalts: implications for mantle composition and processes. In: Saunders AD and Norry MJ (eds). *Magmatism in the Ocean Basins*. London: Geological Society, 1989, 313-45.

14. Brier GW. Verification of forecasts expressed in terms of probability. *Mon Weather Rev* 1950; 78: 1-3.

15. Furman T. Geochemistry of East African Rift basalts: an overview. *J Afr Earth Sci* 2007; 48: 147-60.

16. Rooney TO, Brown EL and Bastow ID et al. Magmatism during the continent-ocean transition. *Earth Planet Sci Lett* 2023; 614: 118189.

17. Brounce M, Scoggins S and Fischer TP et al. Volatiles and redox along the East African Rift. *Geochem Geophys Geosyst* 2024; 25: e2024GC011657.

18. Rowe MC, Lassiter JC and Goff K. Basalt volatile fluctuations during continental rifting: an example from the Rio Grande Rift, USA. *Geochem Geophys Geosyst* 2015; 16: 1254-73.

19. Blondes MS, Reiners PW and Ducea MN et al. Temporal-compositional trends over short and long time-scales in basalts of the Big Pine Volcanic Field, California. *Earth Planet Sci Lett* 2008; 269: 140-54.

20. Gao R, Lassiter JC and Ramirez G. Origin of temporal compositional trends in monogenetic vent eruptions: insights from the crystal cargo in the Papoose Canyon sequence, Big Pine Volcanic Field, CA. *Earth Planet Sci Lett* 2017; 457: 227-37.

21. Day JMD, Troll VR and Aulinas M et al. Mantle source characteristics and magmatic processes during the 2021 La Palma eruption. *Earth Planet Sci Lett* 2022; 597: 117793.

22. Haase KM, Gress MU and Lima SM et al. Evolution of magmatism in the New Hebrides Island Arc and in initial back-arc rifting, SW Pacific. *Geochem Geophys Geosyst* 2020; 21: e2020GC008946.

23. Zhong Y, Liu W and Sun Z et al. Geochemistry and mineralogy of basalts from the South Mid-Atlantic Ridge (18.0-20.6°S): evidence of a heterogeneous mantle source. *Minerals* 2019; 9: 659.

24. Rhoads EA, Kutyrev A and Bindeman IN et al. Rhenium-osmium and oxygen isotope homogeneity during the 2022 Mauna Loa eruption and implications for basaltic magma storage. *Bull Volcanol* 2025; 87: 38.

25. Zhu C. Supplementary Tables S1 to S6.xlsx. Figshare. 2024, doi: 10.6084/m9.figshare.25295671.v1.

## FIGURE LEGENDS

**Figure 1. Group-aware validation of the PetDB four-class task.** Macro-F1 and balanced accuracy from five-fold out-of-fold predictions using random, citation-set, citation-overlap-component and first-order-location splits. The 21-variable panel contains only elements observed in all four PetDB classes.

**Alt text:** Paired bar charts show high random-split performance for random forest and logistic regression, followed by lower citation-group performance and the lowest performance when locations are held out.

**Figure 2. Conservative permutation importance.** Mean decrease in balanced accuracy after permuting each variable within citation-overlap-held-out folds; error bars show between-fold standard deviation. Importance describes model dependence and is not a unique causal attribution.

**Alt text:** A horizontal bar chart ranks TiO2 and Sr as the two strongest variables, followed by K2O, Zr, MnO, Nb and Y, with uncertainty bars across five held-out folds.

**Figure 3. Bidirectional cross-database transfer.** Three-class ARC-CIB-OIB transfer between GEOROC and PetDB using the same 21 variables. The target database was not used for fitting. Broad and rift-aligned CIB ontologies are compared.

**Alt text:** Paired bar charts show stronger transfer from GEOROC to PetDB than from PetDB to GEOROC; aligning the continental intraplate class to rift volcanics improves the reverse direction but does not remove the gap.

**Figure 4. Four-setting external transfer and back-arc pressure test.** Row-normalized confusion matrices, class recall and unscored Vate Trough prediction shares for balanced logistic regression and balanced random forest. The scored test has 231 samples and two province proxies for each class.

**Alt text:** Confusion matrices and recall bars show near-perfect arc and mid-ocean-ridge classification, strong ocean-island classification and weaker continental intraplate classification; Vate Trough predictions are dominated by mid-ocean-ridge affinity.

**Figure 5. East African Rift as a rift-plume geochemical transition.** The entire East African Rift was removed from training. Panel A summarizes mean four-class probabilities by country; panel B shows element medians shifting the database CIB cohort toward the OIB training domain.

**Alt text:** Stacked country bars are dominated by ocean-island probability, while a horizontal ranking shows that SiO2, FeOT and Nb provide the largest median shifts from other continental intraplate basalts toward ocean-island basalts.
