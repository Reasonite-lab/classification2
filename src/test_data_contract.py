from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss


ROOT = Path(__file__).resolve().parents[1]


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    manifest = pd.read_csv(ROOT / "reports" / "data_quality" / "raw_file_manifest.csv")
    assert len(manifest) == 9
    for row in manifest.itertuples(index=False):
        path = ROOT / "data" / "raw" / "georoc_2jetoa" / row.file
        assert path.exists(), path
        assert path.stat().st_size == row.bytes, path
        assert md5(path) == row.md5, path

    citation_path = ROOT / "data" / "raw" / "2024-12-4EZ7ID_CITATIONS.tab"
    assert md5(citation_path) == "cdebb38d75a8dd05a2ea56ffccbd39ab"
    config = yaml.safe_load(
        (ROOT / "config" / "study_config.yaml").read_text(encoding="utf-8")
    )
    broad_features = config["feature_sets"]["broad_v1"]

    petdb_archive = (
        ROOT
        / "data"
        / "raw"
        / "petdb_morb"
        / "earthchem_export_jul14_26_14620.zip"
    )
    assert petdb_archive.stat().st_size == 57677488
    assert md5(petdb_archive) == "210133d92712637ccd017fed1a567b93"
    assert sha256(petdb_archive) == (
        "24cbc5c90410248790c3560a82aa599a39a4032a24ab762b72db198a6363eb05"
    )
    petdb_profile = json.loads(
        (ROOT / "reports" / "data_quality" / "petdb_morb_raw_profile.json").read_text(
            encoding="utf-8"
        )
    )
    assert petdb_profile["archive"]["hashes_verified"] is True
    assert petdb_profile["header"]["columns"] == 1176
    assert petdb_profile["header"]["all_target_features_present"] is True
    assert petdb_profile["main_analysis_table"]["rows"] == 22680
    assert petdb_profile["main_analysis_table"]["unique_sample_urls"] == 7668
    assert petdb_profile["main_analysis_table"]["unique_citation_urls"] == 431
    assert petdb_profile["main_analysis_table"]["exact_duplicate_rows"] == 32
    petdb_sample_profile = petdb_profile["sample_metadata_table"]
    assert petdb_sample_profile["row_aligned_name_match_count"] == 22680
    assert petdb_sample_profile["row_aligned_coordinate_match_count"] == 22680
    assert petdb_sample_profile["exclusive_spreading_center_sample_urls"] == 7557
    assert petdb_sample_profile["mixed_or_secondary_setting_sample_urls"] == 111
    assert petdb_sample_profile["unknown_setting_sample_urls"] == 0
    assert petdb_sample_profile["name_linkage_sensitivity"][
        "exclusive_spreading_center_sample_urls"
    ] == 7511
    assert petdb_sample_profile["name_linkage_sensitivity"][
        "sample_urls_affected_by_ambiguous_names"
    ] == 335
    assert sum(
        petdb_sample_profile[key]
        for key in [
            "exclusive_spreading_center_sample_urls",
            "mixed_or_secondary_setting_sample_urls",
            "unknown_setting_sample_urls",
        ]
    ) == petdb_profile["main_analysis_table"]["unique_sample_urls"]
    petdb_coverage = pd.read_csv(
        ROOT / "reports" / "data_quality" / "petdb_morb_feature_coverage.csv"
    )
    assert len(petdb_coverage) == 29
    assert petdb_coverage["numeric_invalid"].eq(0).all()
    assert (ROOT / "notebooks" / "04_petdb_morb_raw_audit.ipynb").exists()

    petdb_processing = json.loads(
        (
            ROOT
            / "reports"
            / "data_quality"
            / "petdb_morb_processing_profile.json"
        ).read_text(encoding="utf-8")
    )
    assert petdb_processing["exact_duplicate_rows_removed"] == 32
    assert petdb_processing["analysis_rows_after_exact_deduplication"] == 22648
    assert petdb_processing["sample_urls"] == 7668
    assert petdb_processing["tectonic_linkage"]["exclusive_spreading_center"] == 7557
    assert petdb_processing["cohorts"] == {
        "strict_candidate": 4249,
        "model_ready_broad": 3650,
        "model_ready_immobile": 2302,
        "model_ready_broad_no_severe_alteration": 3630,
        "model_ready_immobile_no_severe_alteration": 2292,
        "model_ready_broad_major_range_sensitivity": 3630,
        "model_ready_immobile_major_range_sensitivity": 2284,
    }
    petdb_samples = pd.read_parquet(
        ROOT / "data" / "processed" / "petdb_morb_v0_1.parquet"
    )
    assert len(petdb_samples) == 7668
    assert petdb_samples["record_id"].is_unique
    assert petdb_samples["SAMPLE_URL"].is_unique
    assert set(petdb_samples["label_primary4"]) == {"MORB"}
    assert int(petdb_samples["flag_exclusive_spreading_center"].sum()) == 7557
    assert int(petdb_samples["strict_candidate"].sum()) == 4249
    assert int(petdb_samples["model_ready_broad"].sum()) == 3650
    assert int(petdb_samples["model_ready_immobile"].sum()) == 2302
    assert set(broad_features).issubset(petdb_samples.columns)
    assert set(config["feature_sets"]["immobile_v1"]).issubset(petdb_samples.columns)
    petdb_strict = petdb_samples.loc[petdb_samples["strict_candidate"]]
    assert petdb_strict["flag_exclusive_spreading_center"].all()
    assert petdb_strict["SIO2(WT%)"].between(40, 55).all()
    assert petdb_strict["major_total_calc"].between(85, 105).all()
    assert petdb_strict["citation_set"].ne("<MISSING>").all()
    petdb_trace_features = [
        feature
        for feature in set(
            broad_features + config["feature_sets"]["immobile_v1"]
        )
        if "(PPM)" in feature
    ]
    assert not (petdb_strict[petdb_trace_features] <= 0).any().any()
    petdb_bibliography = pd.read_csv(
        ROOT / "references" / "petdb_morb_bibliography.csv"
    )
    assert len(petdb_bibliography) == 431
    assert petdb_bibliography["citation_id"].nunique() == 431
    petdb_duplicate_indices = pd.read_csv(
        ROOT
        / "reports"
        / "data_quality"
        / "petdb_morb_exact_duplicate_indices.csv"
    )
    assert len(petdb_duplicate_indices) == 32

    primary4_export_config = yaml.safe_load(
        (ROOT / "config" / "petdb_primary4_exports.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert [item["label"] for item in primary4_export_config["exports"]] == [
        "MORB",
        "ARC",
        "OIB",
        "CIB",
    ]
    for item in primary4_export_config["exports"]:
        archive = (
            ROOT
            / "data"
            / "raw"
            / item["slug"]
            / f"{item['export_id']}.zip"
        )
        assert archive.stat().st_size == item["archive_bytes"]
        assert sha256(archive) == item["sha256"]
        assert md5(archive) == item["md5"]

    primary4_profile = json.loads(
        (
            ROOT
            / "reports"
            / "data_quality"
            / "petdb_primary4_processing_profile.json"
        ).read_text(encoding="utf-8")
    )
    assert primary4_profile["combined"] == {
        "sample_rows": 13718,
        "exclusive_high_confidence": 13566,
        "model_ready_broad": 6148,
        "model_ready_immobile": 4005,
        "class_counts_all": {"MORB": 7668, "OIB": 3304, "ARC": 1694, "CIB": 1052},
        "class_counts_broad": {"MORB": 3650, "OIB": 1281, "ARC": 754, "CIB": 463},
        "class_counts_immobile": {"MORB": 2302, "OIB": 789, "ARC": 579, "CIB": 335},
    }
    primary4_samples = pd.read_parquet(
        ROOT / "data" / "processed" / "petdb_primary4_v0_1.parquet"
    )
    assert len(primary4_samples) == 13718
    assert primary4_samples["record_id"].is_unique
    assert set(primary4_samples["label_primary4"]) == {"ARC", "CIB", "MORB", "OIB"}
    assert int(primary4_samples["flag_exclusive_target_setting"].sum()) == 13566
    assert int(primary4_samples["model_ready_broad"].sum()) == 6148
    assert int(primary4_samples["model_ready_immobile"].sum()) == 4005
    assert primary4_samples.loc[
        primary4_samples["flag_exclusive_target_setting"], "SAMPLE_URL"
    ].is_unique
    assert primary4_samples.loc[
        primary4_samples["label_primary4"].eq("OIB"), "TH(PPM)"
    ].isna().all()

    primary4_manifest = json.loads(
        (
            ROOT
            / "reports"
            / "modeling"
            / "petdb_primary4_validation_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert primary4_manifest["rows"] == 6148
    assert primary4_manifest["excluded_structurally_absent_features"] == ["TH(PPM)"]
    assert len(primary4_manifest["features"]) == 21
    primary4_predictions = pd.read_parquet(
        ROOT / "reports" / "modeling" / "petdb_primary4_oof_predictions.parquet"
    )
    primary4_overall = pd.read_csv(
        ROOT / "reports" / "modeling" / "petdb_primary4_overall_metrics.csv"
    )
    probability_columns = ["prob_arc", "prob_cib", "prob_morb", "prob_oib"]
    for (strategy, model_name), frame in primary4_predictions.groupby(
        ["cv_strategy", "model"]
    ):
        reported = primary4_overall[
            primary4_overall["cv_strategy"].eq(strategy)
            & primary4_overall["model"].eq(model_name)
        ].iloc[0]
        assert abs(reported["accuracy"] - accuracy_score(frame["actual"], frame["predicted"])) < 1e-12
        assert abs(
            reported["balanced_accuracy"]
            - balanced_accuracy_score(frame["actual"], frame["predicted"])
        ) < 1e-12
        assert abs(
            reported["macro_f1"]
            - f1_score(frame["actual"], frame["predicted"], average="macro")
        ) < 1e-12
        assert abs(
            reported["log_loss"]
            - log_loss(
                frame["actual"],
                frame[probability_columns],
                labels=["ARC", "CIB", "MORB", "OIB"],
            )
        ) < 1e-12
    primary4_missing_folds = pd.read_csv(
        ROOT
        / "reports"
        / "modeling"
        / "petdb_primary4_missingness_fold_metrics.csv"
    )
    assert len(primary4_missing_folds) == 40
    assert set(primary4_missing_folds["variant"]) == {
        "all_features",
        "common_features",
    }
    assert (ROOT / "reports" / "modeling" / "petdb_primary4_all22_overall_metrics.csv").exists()
    cross_manifest = json.loads(
        (
            ROOT
            / "reports"
            / "modeling"
            / "cross_database_validation_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert cross_manifest["georoc_rows"] == 54012
    assert cross_manifest["petdb_rows"] == 2498
    assert cross_manifest["georoc_ontology_variants"]["broad_cib"]["rows"] == 54012
    assert cross_manifest["georoc_ontology_variants"]["rift_aligned_cib"]["rows"] == 27349
    assert cross_manifest["georoc_ontology_variants"]["rift_aligned_cib"]["class_counts"]["CIB"] == 3421
    assert cross_manifest["excluded_structurally_absent_features"] == ["TH(PPM)"]
    assert len(cross_manifest["shared_features"]) == 21
    cross_predictions = pd.read_parquet(
        ROOT / "reports" / "modeling" / "cross_database_predictions.parquet"
    )
    cross_overall = pd.read_csv(
        ROOT / "reports" / "modeling" / "cross_database_overall_metrics.csv"
    )
    assert len(cross_predictions) == 2 * ((54012 + 2498) + (27349 + 2498))
    for (ontology_variant, direction, model_name), frame in cross_predictions.groupby(
        ["ontology_variant", "direction", "model"]
    ):
        reported = cross_overall[
            cross_overall["ontology_variant"].eq(ontology_variant)
            & cross_overall["direction"].eq(direction)
            & cross_overall["model"].eq(model_name)
        ].iloc[0]
        assert abs(
            reported["balanced_accuracy"]
            - balanced_accuracy_score(frame["actual"], frame["predicted"])
        ) < 1e-12
        assert abs(
            reported["macro_f1"]
            - f1_score(frame["actual"], frame["predicted"], average="macro")
        ) < 1e-12
        assert abs(
            reported["log_loss"]
            - log_loss(
                frame["actual"],
                frame[["prob_arc", "prob_cib", "prob_oib"]],
                labels=["ARC", "CIB", "OIB"],
            )
        ) < 1e-12

    error_manifest = json.loads(
        (ROOT / "reports" / "modeling" / "petdb_primary4_error_audit_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert error_manifest["prediction_rows"] == 12296
    assert error_manifest["errors_by_strategy"]["citation_overlap_conservative"] == 928
    assert error_manifest["errors_by_strategy"]["location_root_grouped"] == 1155

    rift_audit = json.loads(
        (ROOT / "reports" / "modeling" / "east_african_rift_audit_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert rift_audit["rows"] == 361
    assert rift_audit["actual_counts"] == {"CIB": 347, "MORB": 14}
    assert rift_audit["confusion"]["CIB"] == {"ARC": 19, "CIB": 5, "MORB": 19, "OIB": 304}
    assert abs(rift_audit["cib_recall"] - 5 / 347) < 1e-12
    assert abs(rift_audit["cib_to_oib_share"] - 304 / 347) < 1e-12

    rift_stress_manifest = json.loads(
        (ROOT / "reports" / "modeling" / "east_african_rift_stress_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert rift_stress_manifest["training_rows"] == 5787
    assert rift_stress_manifest["test_rows"] == 361
    assert len(rift_stress_manifest["features"]) == 21
    rift_predictions = pd.read_parquet(
        ROOT / "reports" / "modeling" / "east_african_rift_stress_predictions.parquet"
    )
    rift_summary = pd.read_csv(
        ROOT / "reports" / "modeling" / "east_african_rift_stress_summary.csv"
    )
    assert len(rift_predictions) == 2 * 361
    assert not rift_predictions.duplicated(["record_id", "model"]).any()
    assert (
        rift_predictions[["prob_arc", "prob_cib", "prob_morb", "prob_oib"]].sum(axis=1).sub(1).abs()
        < 1e-9
    ).all()
    for model_name, frame in rift_predictions.groupby("model"):
        cib = frame.loc[frame["actual"].eq("CIB")]
        reported = rift_summary[
            rift_summary["model"].eq(model_name)
            & rift_summary["actual_cohort"].eq("CIB")
        ].iloc[0]
        assert len(cib) == reported["n"] == 347
        assert abs(reported["mean_prob_cib"] - cib["prob_cib"].mean()) < 1e-12
        assert abs(reported["mean_prob_oib"] - cib["prob_oib"].mean()) < 1e-12
        assert abs(reported["mean_transition_index"] - cib["transition_index"].mean()) < 1e-12
        assert abs(
            reported["share_prob_oib_gt_cib"] - cib["prob_oib"].gt(cib["prob_cib"]).mean()
        ) < 1e-12

    interpretability_manifest = json.loads(
        (ROOT / "reports" / "modeling" / "petdb_primary4_interpretability_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert interpretability_manifest["cohort_rows"] == 6148
    assert interpretability_manifest["excluded_structurally_absent_features"] == ["TH(PPM)"]
    assert len(interpretability_manifest["features"]) == 21
    assert interpretability_manifest["shap_sample_rows"] == 800
    permutation_folds = pd.read_csv(
        ROOT / "reports" / "modeling" / "petdb_primary4_permutation_importance_folds.csv"
    )
    permutation_summary = pd.read_csv(
        ROOT / "reports" / "modeling" / "petdb_primary4_permutation_importance_summary.csv"
    )
    assert len(permutation_folds) == 5 * 21
    assert permutation_folds.groupby("feature")["fold"].nunique().eq(5).all()
    for row in permutation_summary.itertuples(index=False):
        values = permutation_folds.loc[
            permutation_folds["feature"].eq(row.feature), "importance_mean"
        ]
        assert abs(row.importance_mean - values.mean()) < 1e-12
        assert abs(row.importance_sd_between_folds - values.std(ddof=1)) < 1e-12
        assert abs(row.min_fold - values.min()) < 1e-12
        assert abs(row.max_fold - values.max()) < 1e-12
    shap_class = pd.read_csv(
        ROOT / "reports" / "modeling" / "petdb_primary4_shap_global_by_class.csv"
    )
    shap_overall = pd.read_csv(
        ROOT / "reports" / "modeling" / "petdb_primary4_shap_global_overall.csv"
    )
    shap_direction = pd.read_csv(
        ROOT / "reports" / "modeling" / "petdb_primary4_shap_direction_spearman.csv"
    )
    assert len(shap_class) == 21 * 4
    assert len(shap_overall) == 21
    assert len(shap_direction) == 21 * 4
    assert shap_direction["spearman_log_value_vs_shap"].between(-1, 1).all()

    calibration_manifest = json.loads(
        (ROOT / "reports" / "modeling" / "petdb_primary4_figure_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    primary_calibration = calibration_manifest["primary_calibration"]
    calibration_frame = primary4_predictions[
        primary4_predictions["cv_strategy"].eq("citation_overlap_conservative")
        & primary4_predictions["model"].eq("random_forest_balanced")
    ]
    probability = calibration_frame[probability_columns].to_numpy(float)
    actual_index = np.array([["ARC", "CIB", "MORB", "OIB"].index(value) for value in calibration_frame["actual"]])
    one_hot = np.eye(4)[actual_index]
    brier = float(np.mean(np.sum((probability - one_hot) ** 2, axis=1)))
    confidence = probability.max(axis=1)
    correct = probability.argmax(axis=1) == actual_index
    bin_id = np.clip(np.digitize(confidence, np.linspace(0, 1, 11), right=True) - 1, 0, 9)
    ece = sum(
        (bin_id == index).mean()
        * abs(correct[bin_id == index].mean() - confidence[bin_id == index].mean())
        for index in range(10)
        if (bin_id == index).any()
    )
    assert abs(primary_calibration["multiclass_brier"] - brier) < 1e-12
    assert abs(primary_calibration["top_label_ece"] - ece) < 1e-12
    assert abs(primary_calibration["mean_top_label_confidence"] - confidence.mean()) < 1e-12
    for figure in [
        "petdb_primary4_validation_performance",
        "petdb_primary4_grouped_confusion",
        "petdb_primary4_calibration",
        "cross_database_transfer",
        "petdb_primary4_permutation_importance",
        "petdb_primary4_shap_global_heatmap",
        "petdb_primary4_shap_direction_heatmap",
        "pangaea_morb_external_probabilities",
        "petdb_primary4_error_audit",
        "east_african_rift_transition",
        "figshare_morb_external_validation",
    ]:
        for suffix in [".png", ".pdf"]:
            path = ROOT / "figures" / f"{figure}{suffix}"
            assert path.exists() and path.stat().st_size > 10_000, path

    pangaea_hashes = {
        "pangaea_956077.tsv": (362212, "44fa35e5e2e0198531edf85ed6b1dfd1383da6674835d36ff3257c23e7dbf076"),
        "pangaea_727391.tsv": (15914, "8c9b93b961794edeaf626d04cf9bdee435259954cdaaaaf1a28df419ce27f55b"),
        "pangaea_707361.tsv": (2453, "d1e16a4368390273efa337c2250c148189af997676e809bde3208d9b87f71298"),
    }
    for filename, (bytes_expected, hash_expected) in pangaea_hashes.items():
        path = ROOT / "data" / "raw" / filename
        assert path.stat().st_size == bytes_expected
        assert sha256(path) == hash_expected
    pangaea_audit = pd.read_csv(
        ROOT / "reports" / "data_quality" / "pangaea_morb_candidate_audit.csv"
    ).set_index("dataset_id")
    assert len(pangaea_audit) == 3
    assert bool(pangaea_audit.loc["PANGAEA.956077", "publication_citation_detected_in_petdb"])
    assert pangaea_audit.loc["PANGAEA.956077", "petdb_exact_normalized_name_overlap"] == 0
    assert pangaea_audit.loc["PANGAEA.727391", "samples_with_at_least_10_shared_features"] == 16
    assert pangaea_audit.loc["PANGAEA.707361", "samples_with_at_least_10_shared_features"] == 2
    pangaea_manifest = json.loads(
        (ROOT / "reports" / "modeling" / "pangaea_morb_external_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert pangaea_manifest["training_rows"] == 6148
    assert pangaea_manifest["external_test_rows"] == 18
    assert len(pangaea_manifest["features"]) == 10
    assert set(pangaea_manifest["accepted_sources"]) == {"PANGAEA.727391", "PANGAEA.707361"}
    assert set(pangaea_manifest["excluded_source"]) == {"PANGAEA.956077"}
    pangaea_samples = pd.read_parquet(
        ROOT / "data" / "processed" / "pangaea_morb_external_v0_1.parquet"
    )
    assert len(pangaea_samples) == 21
    assert int(pangaea_samples["model_ready_external"].sum()) == 18
    assert pangaea_samples.loc[pangaea_samples["model_ready_external"], "record_id"].is_unique
    pangaea_predictions = pd.read_csv(
        ROOT / "reports" / "modeling" / "pangaea_morb_external_predictions.csv"
    )
    pangaea_summary = pd.read_csv(
        ROOT / "reports" / "modeling" / "pangaea_morb_external_summary.csv"
    )
    assert len(pangaea_predictions) == 2 * 18
    for (model_name, source), frame in pangaea_predictions.groupby(["model", "source_dataset"]):
        reported = pangaea_summary[
            pangaea_summary["model"].eq(model_name)
            & pangaea_summary["source_dataset"].eq(source)
        ].iloc[0]
        assert abs(reported["morb_recall"] - frame["predicted"].eq("MORB").mean()) < 1e-12
        assert abs(reported["mean_prob_morb"] - frame["prob_morb"].mean()) < 1e-12
        assert abs(reported["median_prob_morb"] - frame["prob_morb"].median()) < 1e-12
    for model_name, frame in pangaea_predictions.groupby("model"):
        reported = pangaea_summary[
            pangaea_summary["model"].eq(model_name)
            & pangaea_summary["source_dataset"].eq("ALL_ACCEPTED")
        ].iloc[0]
        assert abs(reported["morb_recall"] - frame["predicted"].eq("MORB").mean()) < 1e-12
        assert abs(reported["mean_prob_morb"] - frame["prob_morb"].mean()) < 1e-12

    figshare_raw = (
        ROOT
        / "data"
        / "raw"
        / "figshare_25295671"
        / "Supplementary_Tables_S1_to_S6.xlsx"
    )
    assert figshare_raw.stat().st_size == 58126
    assert md5(figshare_raw) == "a9a814369a577fb3acef128148d785bb"
    assert sha256(figshare_raw) == (
        "82dcf0d827dab645e2de8c8587fbf58921ac3a5b2123364ae7afc199819141bf"
    )
    extraction_manifest = json.loads(
        (
            ROOT
            / "reports"
            / "data_quality"
            / "figshare_25295671_extraction_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert extraction_manifest["dataset_doi"] == "10.6084/m9.figshare.25295671.v1"
    assert extraction_manifest["extracted_sample_rows"] == 44
    assert extraction_manifest["region_counts"] == {
        "East Pacific Rise": 8,
        "Southwest Indian Ridge": 36,
    }
    assert extraction_manifest["epr_provenance_note"] == (
        "a Data for EPR MORB are from Li et al. (2019)"
    )
    figshare_audit = pd.read_csv(
        ROOT
        / "reports"
        / "data_quality"
        / "figshare_25295671_morb_candidate_audit.csv"
    ).set_index("source_region")
    assert len(figshare_audit) == 2
    assert figshare_audit["petdb_exact_normalized_name_overlap"].eq(0).all()
    assert figshare_audit.loc["East Pacific Rise", "strict_external_candidate"] == 0
    assert figshare_audit.loc["Southwest Indian Ridge", "strict_external_candidate"] == 36
    assert figshare_audit["model_ready_common18"].sum() == 44
    figshare_samples = pd.read_parquet(
        ROOT / "data" / "processed" / "figshare_morb_external_v0_1.parquet"
    )
    assert len(figshare_samples) == 44
    assert figshare_samples["normalized_sample_name"].is_unique
    assert int(figshare_samples["strict_external_candidate"].sum()) == 36
    assert figshare_samples["flag_complete_major10"].all()
    assert figshare_samples["major_total_calc"].between(85, 105).all()
    figshare_manifest = json.loads(
        (
            ROOT / "reports" / "modeling" / "figshare_morb_external_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert figshare_manifest["training_rows"] == 6148
    assert figshare_manifest["external_rows"] == 44
    assert figshare_manifest["strict_swir_rows"] == 36
    assert len(figshare_manifest["features"]) == 18
    figshare_predictions = pd.read_csv(
        ROOT / "reports" / "modeling" / "figshare_morb_external_predictions.csv"
    )
    figshare_summary = pd.read_csv(
        ROOT / "reports" / "modeling" / "figshare_morb_external_summary.csv"
    )
    assert len(figshare_predictions) == 2 * 44
    assert figshare_predictions.groupby("model")["record_id"].nunique().eq(44).all()
    cohort_masks = {
        "SWIR_STRICT_PROVISIONAL": figshare_predictions[
            "strict_external_candidate"
        ].astype(bool),
        "EPR_SECONDARY_SENSITIVITY": figshare_predictions["source_region"].eq(
            "East Pacific Rise"
        ),
        "ALL_REPOSITORY_SENSITIVITY": pd.Series(
            True, index=figshare_predictions.index
        ),
    }
    for model_name in figshare_predictions["model"].unique():
        model_frame = figshare_predictions.loc[
            figshare_predictions["model"].eq(model_name)
        ]
        for cohort_name, mask in cohort_masks.items():
            frame = figshare_predictions.loc[
                figshare_predictions["model"].eq(model_name) & mask
            ]
            reported = figshare_summary[
                figshare_summary["model"].eq(model_name)
                & figshare_summary["cohort"].eq(cohort_name)
            ].iloc[0]
            assert reported["n"] == len(frame)
            assert abs(
                reported["morb_recall"] - frame["predicted"].eq("MORB").mean()
            ) < 1e-12
            assert abs(reported["mean_prob_morb"] - frame["prob_morb"].mean()) < 1e-12
            assert abs(
                reported["median_prob_morb"] - frame["prob_morb"].median()
            ) < 1e-12
        assert len(model_frame) == 44

    common18_internal = pd.read_csv(
        ROOT / "reports" / "modeling" / "petdb_common18_figshare_internal_metrics.csv"
    )
    assert len(common18_internal) == 2
    assert common18_internal["n"].eq(6148).all()
    assert common18_internal["feature_set"].eq("common18_figshare").all()

    external_audit = json.loads(
        (ROOT / "reports" / "data_quality" / "arc_cib_external_candidate_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert external_audit["pangaea_922011"]["arc_front_n"] == 12
    assert external_audit["pangaea_922011"]["back_arc_transition_n"] == 21
    assert external_audit["pangaea_922011"]["short_ambiguous_collision_n"] == 4
    assert external_audit["rio_grande_rift"]["natural_whole_rock_n"] == 29
    assert external_audit["rio_grande_rift"]["strict_silica_40_55_model_ready_n"] == 28
    assert sha256(ROOT / "data" / "raw" / "external_candidates" / "pangaea_922011.tsv") == (
        "b9c72023f63745a640a607db0bd246fe3ba172d964c5232144e42884e7beb47b"
    )
    assert sha256(
        ROOT
        / "data"
        / "raw"
        / "external_candidates"
        / "ggge20719-sup-0002-2014GC005649-ts01.xlsx"
    ) == "094248e2e89e8654d50e6b9a6587f9ef6effa33e3135b8ba6115693172edd1c9"

    arc_backarc = pd.read_parquet(
        ROOT / "data" / "processed" / "pangaea_922011_arc_backarc_external_v0_1.parquet"
    )
    rgr = pd.read_parquet(
        ROOT / "data" / "processed" / "rio_grande_rift_cib_external_v0_1.parquet"
    )
    assert len(arc_backarc) == 33
    assert int(arc_backarc["model_ready_common22"].sum()) == 33
    assert len(rgr) == 29
    assert int(rgr["strict_external_candidate"].sum()) == 28

    mauna_audit = json.loads(
        (ROOT / "reports" / "data_quality" / "mauna_loa_oib_external_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert mauna_audit["mauna_loa_2022_samples"] == 16
    assert mauna_audit["model_ready_common22"] == 16
    assert not mauna_audit["publication_detected_in_petdb"]
    assert sha256(
        ROOT
        / "data"
        / "raw"
        / "external_candidates"
        / "445_2025_1825_MOESM1_ESM.xlsx"
    ) == "25ae9c4c5f3521c976ab82e81358521e2b3141ad98f9a8a212c32e1b1a60975e"
    mauna = pd.read_parquet(
        ROOT / "data" / "processed" / "mauna_loa_2022_oib_external_v0_1.parquet"
    )
    assert len(mauna) == 16
    assert mauna["model_ready_common22"].all()
    assert (
        mauna["FEOT(WT%)"] - 0.8998 * mauna["FE2O3T_REPORTED(WT%)"]
    ).abs().max() < 1e-12

    external_predictions = pd.read_csv(
        ROOT / "reports" / "modeling" / "multiclass_external_predictions.csv"
    )
    external_metrics = pd.read_csv(
        ROOT / "reports" / "modeling" / "multiclass_external_metrics.csv"
    )
    external_per_class = pd.read_csv(
        ROOT / "reports" / "modeling" / "multiclass_external_per_class.csv"
    )
    assert len(external_predictions) == 231 * 2
    assert not external_predictions.duplicated(["record_id", "model"]).any()
    assert external_predictions.groupby("model")["actual_label"].value_counts().to_dict() == {
        ("logistic_balanced", "MORB"): 48,
        ("logistic_balanced", "CIB"): 98,
        ("logistic_balanced", "OIB"): 55,
        ("logistic_balanced", "ARC"): 30,
        ("random_forest_balanced", "MORB"): 48,
        ("random_forest_balanced", "CIB"): 98,
        ("random_forest_balanced", "OIB"): 55,
        ("random_forest_balanced", "ARC"): 30,
    }
    external_probabilities = external_predictions[
        ["prob_arc", "prob_cib", "prob_morb", "prob_oib"]
    ]
    assert ((external_probabilities.sum(axis=1) - 1).abs() < 1e-9).all()
    for model_name, frame in external_predictions.groupby("model"):
        reported = external_metrics.loc[external_metrics["model"].eq(model_name)].iloc[0]
        probability = frame[["prob_arc", "prob_cib", "prob_morb", "prob_oib"]]
        assert abs(reported["accuracy"] - accuracy_score(frame["actual_label"], frame["predicted"])) < 1e-12
        assert abs(
            reported["balanced_accuracy"]
            - balanced_accuracy_score(frame["actual_label"], frame["predicted"])
        ) < 1e-12
        assert abs(
            reported["macro_f1"]
            - f1_score(frame["actual_label"], frame["predicted"], average="macro")
        ) < 1e-12
        assert abs(
            reported["log_loss"]
            - log_loss(frame["actual_label"], probability, labels=["ARC", "CIB", "MORB", "OIB"])
        ) < 1e-12
    rf_external = external_per_class.loc[
        external_per_class["model"].eq("random_forest_balanced")
    ].set_index("label")
    assert abs(rf_external.loc["ARC", "recall"] - 29 / 30) < 1e-12
    assert rf_external.loc["MORB", "recall"] == 1
    assert abs(rf_external.loc["OIB", "recall"] - 44 / 55) < 1e-12
    assert abs(rf_external.loc["CIB", "recall"] - 51 / 98) < 1e-12
    assert (ROOT / "figures" / "multiclass_external_validation.pdf").exists()

    ueki_source = ROOT / "data" / "raw" / "ueki2018_arxiv_source.tar"
    assert ueki_source.stat().st_size == 3694465
    assert md5(ueki_source) == "f287b1ce6593ab3b62c35f1dd230ecbc"
    ueki_metadata = json.loads(
        (ROOT / "data" / "raw" / "ueki2018_arxiv_source_metadata.json").read_text(encoding="utf-8")
    )
    assert len(ueki_metadata["archive_members"]) == 13
    with tarfile.open(ueki_source) as archive:
        assert archive.getnames() == ueki_metadata["archive_members"]
    assert not any(
        member.lower().endswith((".csv", ".tsv", ".xlsx", ".xls", ".r", ".rdata"))
        for member in ueki_metadata["archive_members"]
    )

    profile = json.loads(
        (ROOT / "reports" / "data_quality" / "primary3_processing_profile.json").read_text(encoding="utf-8")
    )
    data = pd.read_parquet(ROOT / "data" / "processed" / "georoc_primary3_v0_1.parquet")
    assert len(data) == profile["processed_rows"] == 62113
    assert int(data["model_ready_broad"].sum()) == profile["model_ready_broad_rows"] == 54012
    assert set(data["label_primary3"]) == {"ARC", "CIB", "OIB"}
    assert data["record_id"].is_unique
    assert data["record_id"].notna().all()
    assert data["major_total_plausible"].all()
    assert data["major_total_calc"].between(85.0, 105.0).all()
    assert set(broad_features).issubset(data.columns)
    assert all(pd.api.types.is_numeric_dtype(data[column]) for column in broad_features)

    forbidden = {
        "TECTONIC SETTING", "LOCATION", "CITATIONS", "ROCK NAME", "SAMPLE NAME",
        "label_primary3", "label_fine", "source_file", "record_id",
    }
    assert forbidden.isdisjoint(broad_features)

    ready = data.loc[data["model_ready_broad"]]
    assert (ready["broad_observed_n"] >= 12).all()
    assert ready.groupby("citation_set")["label_primary3"].nunique().max() == 1
    trace_features = [feature for feature in broad_features if "(PPM)" in feature]
    assert not (ready[trace_features] <= 0).any().any()

    metrics = pd.read_csv(ROOT / "reports" / "modeling" / "baseline_overall_metrics.csv")
    expected_strategies = {
        "random_stratified", "citation_set_grouped", "location_root_grouped",
        "citation_overlap_conservative",
    }
    expected_models = {"dummy_prior", "logistic_balanced", "random_forest_balanced"}
    assert set(metrics["cv_strategy"]) == expected_strategies
    assert set(metrics["model"]) == expected_models
    assert len(metrics) == len(expected_strategies) * len(expected_models)
    assert metrics[["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]].apply(
        lambda column: column.between(0, 1).all()
    ).all()

    oof = pd.read_parquet(ROOT / "reports" / "modeling" / "baseline_oof_predictions.parquet")
    assert len(oof) == 54012 * len(expected_strategies) * len(expected_models)
    assert not oof.duplicated(["record_id", "cv_strategy", "model"]).any()
    assert set(oof["fold"]) == {1, 2, 3, 4, 5}
    probabilities = oof[["prob_arc", "prob_cib", "prob_oib"]]
    assert probabilities.apply(lambda column: column.between(0, 1).all()).all()
    assert ((probabilities.sum(axis=1) - 1.0).abs() < 1e-9).all()

    labels = ["ARC", "CIB", "OIB"]
    recalculated = []
    for (strategy, model), group in oof.groupby(["cv_strategy", "model"], sort=True):
        recalculated.append(
            {
                "cv_strategy": strategy,
                "model": model,
                "n": len(group),
                "accuracy": accuracy_score(group["actual"], group["predicted"]),
                "balanced_accuracy": balanced_accuracy_score(group["actual"], group["predicted"]),
                "macro_f1": f1_score(group["actual"], group["predicted"], average="macro"),
                "weighted_f1": f1_score(group["actual"], group["predicted"], average="weighted"),
                "log_loss": log_loss(group["actual"], group[["prob_arc", "prob_cib", "prob_oib"]], labels=labels),
            }
        )
    recalculated = pd.DataFrame(recalculated).sort_values(["cv_strategy", "model"]).reset_index(drop=True)
    reported = metrics.sort_values(["cv_strategy", "model"]).reset_index(drop=True)
    assert (recalculated[["cv_strategy", "model", "n"]] == reported[["cv_strategy", "model", "n"]]).all().all()
    numeric_metrics = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "log_loss"]
    assert ((recalculated[numeric_metrics] - reported[numeric_metrics]).abs() < 1e-12).all().all()

    sensitivity_folds = pd.read_csv(
        ROOT / "reports" / "modeling" / "feature_bias_sensitivity_fold_metrics.csv"
    )
    sensitivity_summary = pd.read_csv(
        ROOT / "reports" / "modeling" / "feature_bias_sensitivity_summary.csv"
    )
    assert len(sensitivity_folds) == 2 * 4 * 5
    assert set(sensitivity_folds["cv_strategy"]) == {"citation_set_grouped", "location_root_grouped"}
    assert set(sensitivity_folds["input_spec"]) == {
        "broad_values", "broad_values_plus_missing", "immobile_values", "broad_missingness_only"
    }
    grouped = sensitivity_folds.groupby(["cv_strategy", "input_spec"], sort=True)
    expected_sensitivity = grouped.agg(
        n_folds=("fold", "nunique"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_sd=("accuracy", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_sd=("balanced_accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_sd=("macro_f1", "std"),
        log_loss_mean=("log_loss", "mean"),
    ).reset_index()
    sensitivity_summary = sensitivity_summary.sort_values(["cv_strategy", "input_spec"]).reset_index(drop=True)
    expected_sensitivity = expected_sensitivity.sort_values(["cv_strategy", "input_spec"]).reset_index(drop=True)
    assert (sensitivity_summary[["cv_strategy", "input_spec", "n_folds"]] == expected_sensitivity[["cv_strategy", "input_spec", "n_folds"]]).all().all()
    sensitivity_numeric = [
        "accuracy_mean", "accuracy_sd", "balanced_accuracy_mean", "balanced_accuracy_sd",
        "macro_f1_mean", "macro_f1_sd", "log_loss_mean",
    ]
    assert ((sensitivity_summary[sensitivity_numeric] - expected_sensitivity[sensitivity_numeric]).abs() < 1e-12).all().all()

    permutation_folds = pd.read_csv(ROOT / "reports" / "modeling" / "permutation_importance_folds.csv")
    permutation_summary = pd.read_csv(ROOT / "reports" / "modeling" / "permutation_importance_summary.csv")
    assert len(permutation_folds) == 5 * len(broad_features)
    assert permutation_folds.groupby("feature")["fold"].nunique().eq(5).all()
    expected_permutation = permutation_folds.groupby("feature", as_index=False).agg(
        importance_mean=("importance_mean", "mean"),
        importance_sd_between_folds=("importance_mean", "std"),
        min_fold=("importance_mean", "min"),
        max_fold=("importance_mean", "max"),
    )
    permutation_summary = permutation_summary.sort_values("feature").reset_index(drop=True)
    expected_permutation = expected_permutation.sort_values("feature").reset_index(drop=True)
    assert permutation_summary["feature"].equals(expected_permutation["feature"])
    permutation_numeric = ["importance_mean", "importance_sd_between_folds", "min_fold", "max_fold"]
    assert ((permutation_summary[permutation_numeric] - expected_permutation[permutation_numeric]).abs() < 1e-12).all().all()

    shap_by_class = pd.read_csv(ROOT / "reports" / "modeling" / "shap_global_by_class.csv")
    shap_overall = pd.read_csv(ROOT / "reports" / "modeling" / "shap_global_overall.csv")
    shap_direction = pd.read_csv(ROOT / "reports" / "modeling" / "shap_direction_spearman.csv")
    assert len(shap_by_class) == len(broad_features) * 3
    assert len(shap_direction) == len(broad_features) * 3
    assert not shap_by_class.duplicated(["feature", "label"]).any()
    assert not shap_direction.duplicated(["feature", "label"]).any()
    expected_shap_overall = shap_by_class.groupby("feature", as_index=False)["mean_abs_shap"].mean()
    expected_shap_overall = expected_shap_overall.rename(columns={"mean_abs_shap": "mean_abs_shap_over_classes"})
    shap_overall = shap_overall.sort_values("feature").reset_index(drop=True)
    expected_shap_overall = expected_shap_overall.sort_values("feature").reset_index(drop=True)
    assert shap_overall["feature"].equals(expected_shap_overall["feature"])
    assert ((shap_overall["mean_abs_shap_over_classes"] - expected_shap_overall["mean_abs_shap_over_classes"]).abs() < 1e-12).all()
    assert shap_direction["spearman_log_value_vs_shap"].between(-1, 1).all()

    print(
        "GEOROC/PetDB data, model, sensitivity, and interpretability contracts passed."
    )


if __name__ == "__main__":
    main()
