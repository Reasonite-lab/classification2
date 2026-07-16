from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def metric(frame: pd.DataFrame, **filters: str) -> pd.Series:
    selected = frame
    for column, value in filters.items():
        selected = selected.loc[selected[column].eq(value)]
    assert len(selected) == 1, filters
    return selected.iloc[0]


def main() -> None:
    text = (ROOT / "paper" / "manuscript_draft.md").read_text(encoding="utf-8")
    profile = json.loads(
        (ROOT / "reports" / "data_quality" / "primary3_processing_profile.json").read_text(encoding="utf-8")
    )
    overall = pd.read_csv(ROOT / "reports" / "modeling" / "baseline_overall_metrics.csv")
    per_class = pd.read_csv(ROOT / "reports" / "modeling" / "baseline_per_class_metrics.csv")
    sensitivity = pd.read_csv(ROOT / "reports" / "modeling" / "feature_bias_sensitivity_summary.csv")
    primary4_profile = json.loads(
        (
            ROOT
            / "reports"
            / "data_quality"
            / "petdb_primary4_processing_profile.json"
        ).read_text(encoding="utf-8")
    )
    primary4_overall = pd.read_csv(
        ROOT / "reports" / "modeling" / "petdb_primary4_overall_metrics.csv"
    )
    primary4_per_class = pd.read_csv(
        ROOT / "reports" / "modeling" / "petdb_primary4_per_class_metrics.csv"
    )
    primary4_missingness = pd.read_csv(
        ROOT / "reports" / "modeling" / "petdb_primary4_missingness_summary.csv"
    )
    cross_database = pd.read_csv(
        ROOT / "reports" / "modeling" / "cross_database_overall_metrics.csv"
    )
    cross_database_per_class = pd.read_csv(
        ROOT / "reports" / "modeling" / "cross_database_per_class_metrics.csv"
    )
    primary4_importance = pd.read_csv(
        ROOT / "reports" / "modeling" / "petdb_primary4_permutation_importance_summary.csv"
    )
    primary4_calibration = pd.read_csv(
        ROOT / "reports" / "modeling" / "petdb_primary4_calibration_summary.csv"
    )
    major10_internal = pd.read_csv(
        ROOT / "reports" / "modeling" / "petdb_major10_internal_metrics.csv"
    )
    pangaea_external = pd.read_csv(
        ROOT / "reports" / "modeling" / "pangaea_morb_external_summary.csv"
    )
    common18_internal = pd.read_csv(
        ROOT / "reports" / "modeling" / "petdb_common18_figshare_internal_metrics.csv"
    )
    figshare_external = pd.read_csv(
        ROOT / "reports" / "modeling" / "figshare_morb_external_summary.csv"
    )
    multiclass_external = pd.read_csv(
        ROOT / "reports" / "modeling" / "multiclass_external_metrics.csv"
    )
    multiclass_external_per_class = pd.read_csv(
        ROOT / "reports" / "modeling" / "multiclass_external_per_class.csv"
    )
    multiclass_backarc = pd.read_csv(
        ROOT / "reports" / "modeling" / "multiclass_external_backarc_summary.csv"
    )
    rift_stress = pd.read_csv(
        ROOT / "reports" / "modeling" / "east_african_rift_stress_summary.csv"
    )
    rift_audit = json.loads(
        (ROOT / "reports" / "modeling" / "east_african_rift_audit_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    required_literals = {
        f"{profile['input_rows']:,}": "raw row count",
        f"{profile['processed_rows']:,}": "processed row count",
        f"{profile['model_ready_broad_rows']:,}": "broad model-ready count",
        f"{profile['model_ready_immobile_rows']:,}": "immobile/common cohort count",
    }
    strategy_names = {
        "random_stratified": "random",
        "citation_set_grouped": "citation",
        "citation_overlap_conservative": "citation overlap",
        "location_root_grouped": "location",
    }
    for strategy, label in strategy_names.items():
        row = metric(overall, cv_strategy=strategy, model="random_forest_balanced")
        for column in ["accuracy", "balanced_accuracy", "macro_f1", "log_loss"]:
            required_literals[f"{row[column]:.3f}"] = f"{label} {column}"

    for strategy in ["citation_set_grouped", "location_root_grouped"]:
        for label in ["ARC", "CIB", "OIB"]:
            row = metric(
                per_class,
                cv_strategy=strategy,
                model="random_forest_balanced",
                label=label,
            )
            required_literals[f"{row['recall']:.3f}"] = f"{strategy} {label} recall"

    for strategy in ["citation_set_grouped", "location_root_grouped"]:
        for input_spec in [
            "broad_values",
            "broad_values_plus_missing",
            "immobile_values",
            "broad_missingness_only",
        ]:
            row = metric(sensitivity, cv_strategy=strategy, input_spec=input_spec)
            required_literals[f"{row['macro_f1_mean']:.3f}"] = f"{strategy} {input_spec} macro-F1"

    for key, label in [
        ("sample_rows", "PetDB primary4 sample count"),
        ("exclusive_high_confidence", "PetDB primary4 high-confidence count"),
        ("model_ready_broad", "PetDB primary4 broad count"),
        ("model_ready_immobile", "PetDB primary4 immobile count"),
    ]:
        required_literals[f"{primary4_profile['combined'][key]:,}"] = label
    for strategy in strategy_names:
        row = metric(
            primary4_overall,
            cv_strategy=strategy,
            model="random_forest_balanced",
        )
        required_literals[f"{row['balanced_accuracy']:.3f}"] = (
            f"PetDB primary4 {strategy} balanced accuracy"
        )
        required_literals[f"{row['macro_f1']:.3f}"] = (
            f"PetDB primary4 {strategy} macro-F1"
        )
    for strategy in ["citation_overlap_conservative", "location_root_grouped"]:
        for label in ["ARC", "CIB", "MORB", "OIB"]:
            row = metric(
                primary4_per_class,
                cv_strategy=strategy,
                model="random_forest_balanced",
                label=label,
            )
            required_literals[f"{row['recall']:.3f}"] = (
                f"PetDB primary4 {strategy} {label} recall"
            )
    for variant in ["all_features", "common_features"]:
        row = metric(
            primary4_missingness,
            variant=variant,
            representation="missingness_only",
            cv_strategy="citation_overlap_conservative",
        )
        required_literals[f"{row['balanced_accuracy_mean']:.3f}"] = (
            f"PetDB primary4 {variant} missingness balanced accuracy"
        )
        required_literals[f"{row['macro_f1_mean']:.3f}"] = (
            f"PetDB primary4 {variant} missingness macro-F1"
        )
    for direction in ["GEOROC_to_PetDB", "PetDB_to_GEOROC"]:
        row = metric(
            cross_database,
            ontology_variant="broad_cib",
            direction=direction,
            model="random_forest_balanced",
        )
        for column in ["accuracy", "balanced_accuracy", "macro_f1"]:
            required_literals[f"{row[column]:.3f}"] = f"cross-database broad_cib {direction} {column}"
        for label in ["ARC", "CIB", "OIB"]:
            class_row = metric(
                cross_database_per_class,
                ontology_variant="broad_cib",
                direction=direction,
                model="random_forest_balanced",
                label=label,
            )
            required_literals[f"{class_row['recall']:.3f}"] = (
                f"cross-database broad_cib {direction} {label} recall"
            )

    for direction in ["GEOROC_to_PetDB", "PetDB_to_GEOROC"]:
        row = metric(
            cross_database,
            ontology_variant="rift_aligned_cib",
            direction=direction,
            model="random_forest_balanced",
        )
        required_literals[f"{row['macro_f1']:.3f}"] = (
            f"cross-database rift_aligned_cib {direction} macro-F1"
        )
        class_row = metric(
            cross_database_per_class,
            ontology_variant="rift_aligned_cib",
            direction=direction,
            model="random_forest_balanced",
            label="CIB",
        )
        required_literals[f"{class_row['recall']:.3f}"] = (
            f"cross-database rift_aligned_cib {direction} CIB recall"
        )

    for feature in ["TIO2(WT%)", "SR(PPM)", "K2O(WT%)", "ZR(PPM)"]:
        row = metric(primary4_importance, feature=feature)
        required_literals[f"{row['importance_mean']:.3f}"] = f"{feature} importance mean"
        required_literals[f"{row['importance_sd_between_folds']:.3f}"] = f"{feature} importance SD"

    calibration_row = metric(
        primary4_calibration,
        cv_strategy="citation_overlap_conservative",
        model="random_forest_balanced",
    )
    for column in ["multiclass_brier", "top_label_ece", "mean_top_label_confidence"]:
        required_literals[f"{calibration_row[column]:.3f}"] = f"PetDB calibration {column}"

    internal_row = metric(major10_internal, model="random_forest_balanced")
    required_literals[f"{internal_row['balanced_accuracy']:.3f}"] = "PetDB major10 balanced accuracy"
    required_literals[f"{internal_row['macro_f1']:.3f}"] = "PetDB major10 macro-F1"
    for model_name in ["random_forest_balanced", "logistic_balanced"]:
        external_row = metric(
            pangaea_external,
            model=model_name,
            source_dataset="ALL_ACCEPTED",
        )
        required_literals[f"{external_row['morb_recall']:.3f}"] = f"PANGAEA {model_name} MORB recall"
        required_literals[f"{external_row['mean_prob_morb']:.3f}"] = f"PANGAEA {model_name} mean MORB probability"

    for model_name in ["random_forest_balanced", "logistic_balanced"]:
        internal_row = metric(common18_internal, model=model_name)
        required_literals[f"{internal_row['balanced_accuracy']:.3f}"] = (
            f"PetDB common18 {model_name} balanced accuracy"
        )
        required_literals[f"{internal_row['macro_f1']:.3f}"] = (
            f"PetDB common18 {model_name} macro-F1"
        )
        external_row = metric(
            figshare_external,
            model=model_name,
            cohort="SWIR_STRICT_PROVISIONAL",
        )
        required_literals[f"{external_row['morb_recall']:.3f}"] = (
            f"Figshare SWIR {model_name} MORB recall"
        )
        required_literals[f"{external_row['mean_prob_morb']:.3f}"] = (
            f"Figshare SWIR {model_name} mean MORB probability"
        )
        required_literals[f"{external_row['median_prob_morb']:.3f}"] = (
            f"Figshare SWIR {model_name} median MORB probability"
        )

    required_literals["92"] = "strict four-class external sample count"
    for model_name in ["random_forest_balanced", "logistic_balanced"]:
        external_row = metric(multiclass_external, model=model_name)
        for column in ["accuracy", "balanced_accuracy", "macro_f1", "log_loss"]:
            required_literals[f"{external_row[column]:.3f}"] = (
                f"four-class external {model_name} {column}"
            )
    for label in ["ARC", "CIB", "MORB", "OIB"]:
        class_row = metric(
            multiclass_external_per_class,
            model="random_forest_balanced",
            label=label,
        )
        required_literals[f"{class_row['recall']:.3f}"] = (
            f"four-class external random forest {label} recall"
        )
    backarc_row = metric(multiclass_backarc, model="random_forest_balanced")
    required_literals[f"{backarc_row['predicted_share_morb']:.3f}"] = (
        "Vate Trough random forest MORB share"
    )
    required_literals[f"{backarc_row['mean_prob_morb']:.3f}"] = (
        "Vate Trough random forest mean MORB probability"
    )

    required_literals[f"{rift_audit['cib_recall']:.3f}"] = "East African Rift location-held-out CIB recall"
    required_literals[f"{rift_audit['cib_to_oib_share']:.3f}"] = "East African Rift CIB-to-OIB share"
    rift_row = metric(
        rift_stress,
        model="random_forest_balanced",
        actual_cohort="CIB",
    )
    for column in [
        "mean_prob_cib",
        "mean_prob_oib",
        "mean_transition_index",
        "mean_transition_index_ci_low",
        "mean_transition_index_ci_high",
        "share_prob_oib_gt_cib",
    ]:
        required_literals[f"{rift_row[column]:.3f}"] = f"East African Rift stress {column}"

    missing = [f"{label}: {literal}" for literal, label in required_literals.items() if literal not in text]
    assert not missing, "Manuscript is missing validated claims:\n" + "\n".join(missing)

    for doi in [
        "10.25625/2JETOA",
        "10.25625/4EZ7ID",
        "10.1029/1999GC000026",
        "10.1029/2005GC001092",
        "10.1016/j.gca.2005.12.016",
        "10.1007/s00410-016-1292-2",
        "10.1029/2017GC007401",
        "10.1093/petrology/egae036",
        "10.1594/PANGAEA.727391",
        "10.1594/PANGAEA.707361",
        "10.1029/2024JB029166",
        "10.1016/j.epsl.2023.118189",
        "10.1029/2024GC011657",
        "10.1016/j.jafrearsci.2006.06.009",
        "10.6084/m9.figshare.25295671.v1",
        "10.17632/fntfnf92tg.1",
        "10.1594/PANGAEA.922011",
        "10.1029/2020GC008946",
        "10.1002/2014GC005649",
        "10.1007/s00445-025-01825-0",
    ]:
        assert doi in text, doi

    assert "高级草稿" in text
    assert "跨数据库转移诊断" in text
    assert "PANGAEA MORB 试验性外测" in text
    assert "Figshare–Mendeley 西南印度洋脊全岩 MORB 外测" in text
    assert "统一四类外部转移与弧后机制压力测试" in text
    print("Manuscript headline claims match the validated result artifacts.")


if __name__ == "__main__":
    main()
