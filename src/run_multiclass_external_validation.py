from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)

from run_petdb_primary4_validation import LABELS, models


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "reports" / "modeling"
FEATURES = [
    "SIO2(WT%)",
    "TIO2(WT%)",
    "AL2O3(WT%)",
    "FEOT(WT%)",
    "CAO(WT%)",
    "MGO(WT%)",
    "MNO(WT%)",
    "K2O(WT%)",
    "NA2O(WT%)",
    "P2O5(WT%)",
    "RB(PPM)",
    "SR(PPM)",
    "Y(PPM)",
    "ZR(PPM)",
    "NB(PPM)",
    "LA(PPM)",
    "CE(PPM)",
    "ND(PPM)",
]
MODEL_NAMES = ["logistic_balanced", "random_forest_balanced"]


def log_features(frame: pd.DataFrame) -> pd.DataFrame:
    return np.log10(frame[FEATURES].where(frame[FEATURES] > 0))


def build_external() -> tuple[pd.DataFrame, pd.DataFrame]:
    morb = pd.read_parquet(PROCESSED / "figshare_morb_external_v0_1.parquet")
    morb = morb.loc[
        morb["strict_external_candidate"] & morb["model_ready_common18"]
    ].copy()
    morb["sample_name"] = morb["Sample"].astype(str)
    morb["record_id"] = "figshare:25295671:" + morb["sample_name"]
    morb["actual_label"] = "MORB"
    morb["mechanism_cohort"] = "SPREADING_CENTER"
    morb["source_area"] = "Southwest Indian Ridge"
    morb["province_proxy"] = "SWIR repository cohort"
    morb["external_source"] = "Figshare/Mendeley 25295671"

    morb_smar = pd.read_parquet(
        PROCESSED / "smar_2019_morb_external_v0_1.parquet"
    )
    morb_smar = morb_smar.loc[
        morb_smar["strict_external_candidate"]
        & morb_smar["model_ready_common18"]
    ].copy()
    morb_smar["province_proxy"] = "South Mid-Atlantic Ridge 18-20.6 S"
    morb_smar["external_source"] = "Zhong et al. 2019 Table S1"
    morb = pd.concat([morb, morb_smar], ignore_index=True, sort=False)

    arc_backarc = pd.read_parquet(
        PROCESSED / "pangaea_922011_arc_backarc_external_v0_1.parquet"
    )
    arc = arc_backarc.loc[
        arc_backarc["strict_external_candidate"]
        & arc_backarc["model_ready_common22"]
        & arc_backarc["mechanism_cohort"].eq("ARC_FRONT")
    ].copy()
    arc["province_proxy"] = "South New Hebrides arc front"
    arc["external_source"] = "PANGAEA.922011"

    arc_costa_rica = pd.read_parquet(
        PROCESSED / "georoc_costa_rica_arc_external_v0_1.parquet"
    )
    arc_costa_rica = arc_costa_rica.loc[
        arc_costa_rica["strict_external_candidate"]
        & arc_costa_rica["model_ready_common18"]
    ].copy()
    arc_costa_rica["province_proxy"] = "Costa Rica volcanic front"
    arc_costa_rica["external_source"] = "GEOROC 2JETOA 2026-06"
    arc = pd.concat([arc, arc_costa_rica], ignore_index=True, sort=False)

    cib_rgr = pd.read_parquet(
        PROCESSED / "rio_grande_rift_cib_external_v0_1.parquet"
    )
    cib_rgr = cib_rgr.loc[cib_rgr["strict_external_candidate"]].copy()
    cib_rgr["province_proxy"] = "Rio Grande Rift/Jemez Lineament"
    cib_rgr["external_source"] = "Rowe et al. 2015 Table S1"

    cib_big_pine = pd.read_parquet(
        PROCESSED / "georoc_big_pine_cib_external_v0_1.parquet"
    )
    cib_big_pine = cib_big_pine.loc[
        cib_big_pine["strict_external_candidate"]
        & cib_big_pine["model_ready_common18"]
    ].copy()
    cib_big_pine["province_proxy"] = "Big Pine volcanic field"
    cib_big_pine["external_source"] = "GEOROC 2JETOA 2026-06"
    cib = pd.concat([cib_rgr, cib_big_pine], ignore_index=True, sort=False)

    oib_mauna = pd.read_parquet(
        PROCESSED / "mauna_loa_2022_oib_external_v0_1.parquet"
    )
    oib_mauna = oib_mauna.loc[oib_mauna["strict_external_candidate"]].copy()
    oib_mauna["province_proxy"] = "Mauna Loa 2022 eruption"
    oib_mauna["external_source"] = "Rhoads et al. 2025 Table S1"

    oib_palma = pd.read_parquet(
        PROCESSED / "la_palma_2021_oib_external_v0_1.parquet"
    )
    oib_palma = oib_palma.loc[oib_palma["strict_external_candidate"]].copy()
    oib_palma["province_proxy"] = "La Palma 2021 eruption"
    oib_palma["external_source"] = "Day et al. 2022 Table S1"
    oib = pd.concat([oib_mauna, oib_palma], ignore_index=True, sort=False)

    strict = pd.concat([arc, cib, morb, oib], ignore_index=True, sort=False)
    transition = arc_backarc.loc[
        arc_backarc["mechanism_cohort"].eq("BACK_ARC_TRANSITION")
        & arc_backarc["model_ready_common22"]
    ].copy()
    transition["province_proxy"] = "Vate Trough initial back-arc rift"
    transition["external_source"] = "PANGAEA.922011"
    return strict, transition


def probability_matrix(model: object, X: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(X)
    classes = list(model.classes_)
    return np.column_stack([raw[:, classes.index(label)] for label in LABELS])


def prediction_frame(
    data: pd.DataFrame,
    model_name: str,
    predicted: np.ndarray,
    probability: np.ndarray,
) -> pd.DataFrame:
    columns = [
            "record_id",
            "external_source",
            "province_proxy",
            "sample_name",
            "source_area",
            "mechanism_cohort",
            "actual_label",
            "outside_train_1_99_n",
        ]
    if "publication_group" in data.columns:
        columns.append("publication_group")
    frame = data[columns].copy()
    frame["model"] = model_name
    frame["predicted"] = predicted
    for index, label in enumerate(LABELS):
        frame[f"prob_{label.lower()}"] = probability[:, index]
    frame["prob_actual"] = [
        probability[index, LABELS.index(str(label))]
        if str(label) in LABELS
        else np.nan
        for index, label in enumerate(frame["actual_label"])
    ]
    return frame


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(PROCESSED / "petdb_primary4_v0_1.parquet")
    train = train.loc[train["model_ready_broad"]].reset_index(drop=True)
    strict, transition = build_external()
    if strict["record_id"].duplicated().any():
        raise ValueError("Duplicate external record_id")
    if strict["actual_label"].value_counts().index.sort_values().tolist() != LABELS:
        raise ValueError("External strict cohort does not contain all four labels")

    y_train = train["label_primary4"].astype(str)
    X_train = log_features(train)
    X_strict = log_features(strict)
    X_transition = log_features(transition)
    lower = X_train.quantile(0.01)
    upper = X_train.quantile(0.99)
    for frame, X in [(strict, X_strict), (transition, X_transition)]:
        frame["outside_train_1_99_n"] = (
            (X.lt(lower, axis="columns") | X.gt(upper, axis="columns"))
            & X.notna()
        ).sum(axis=1)

    metric_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    transition_frames: list[pd.DataFrame] = []
    transition_rows: list[dict[str, object]] = []
    province_rows: list[dict[str, object]] = []
    templates = models()
    for model_name in MODEL_NAMES:
        model = clone(templates[model_name])
        model.fit(X_train, y_train)
        predicted = model.predict(X_strict)
        probability = probability_matrix(model, X_strict)
        metric_rows.append(
            {
                "model": model_name,
                "validation": "external_four_class_two_province_minimum",
                "n": len(strict),
                "accuracy": accuracy_score(strict["actual_label"], predicted),
                "balanced_accuracy": balanced_accuracy_score(
                    strict["actual_label"], predicted
                ),
                "macro_f1": f1_score(
                    strict["actual_label"], predicted, average="macro"
                ),
                "weighted_f1": f1_score(
                    strict["actual_label"], predicted, average="weighted"
                ),
                "log_loss": log_loss(
                    strict["actual_label"], probability, labels=LABELS
                ),
                "mean_prob_actual": float(
                    np.mean(
                        [
                            probability[index, LABELS.index(label)]
                            for index, label in enumerate(strict["actual_label"])
                        ]
                    )
                ),
            }
        )
        report = classification_report(
            strict["actual_label"],
            predicted,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        )
        for label in LABELS:
            mask = strict["actual_label"].eq(label).to_numpy()
            class_rows.append(
                {
                    "model": model_name,
                    "label": label,
                    "n": int(mask.sum()),
                    "precision": report[label]["precision"],
                    "recall": report[label]["recall"],
                    "f1": report[label]["f1-score"],
                    "mean_prob_actual": float(
                        probability[mask, LABELS.index(label)].mean()
                    ),
                    "median_prob_actual": float(
                        np.median(probability[mask, LABELS.index(label)])
                    ),
                    "mean_outside_train_1_99_n": float(
                        strict.loc[mask, "outside_train_1_99_n"].mean()
                    ),
                    "share_with_any_outside_train_1_99": float(
                        strict.loc[mask, "outside_train_1_99_n"].gt(0).mean()
                    ),
                }
            )
        matrix = confusion_matrix(strict["actual_label"], predicted, labels=LABELS)
        for actual_index, actual in enumerate(LABELS):
            for predicted_index, predicted_label in enumerate(LABELS):
                confusion_rows.append(
                    {
                        "model": model_name,
                        "actual": actual,
                        "predicted": predicted_label,
                        "n": int(matrix[actual_index, predicted_index]),
                        "row_share": float(
                            matrix[actual_index, predicted_index]
                            / matrix[actual_index].sum()
                        ),
                    }
                )
        prediction_frames.append(
            prediction_frame(strict, model_name, predicted, probability)
        )
        for (actual, province), indices in strict.groupby(
            ["actual_label", "province_proxy"], sort=True
        ).groups.items():
            positions = strict.index.get_indexer(indices)
            province_rows.append(
                {
                    "model": model_name,
                    "actual_label": actual,
                    "province_proxy": province,
                    "n": len(positions),
                    "recall": float(np.mean(predicted[positions] == actual)),
                    "mean_prob_actual": float(
                        probability[positions, LABELS.index(actual)].mean()
                    ),
                    "median_prob_actual": float(
                        np.median(probability[positions, LABELS.index(actual)])
                    ),
                    "mean_outside_train_1_99_n": float(
                        strict.iloc[positions]["outside_train_1_99_n"].mean()
                    ),
                }
            )

        transition_predicted = model.predict(X_transition)
        transition_probability = probability_matrix(model, X_transition)
        transition_frames.append(
            prediction_frame(
                transition,
                model_name,
                transition_predicted,
                transition_probability,
            )
        )
        transition_rows.append(
            {
                "model": model_name,
                "cohort": "VATE_TROUGH_BACK_ARC_TRANSITION",
                "n": len(transition),
                **{
                    f"predicted_share_{label.lower()}": float(
                        np.mean(transition_predicted == label)
                    )
                    for label in LABELS
                },
                **{
                    f"mean_prob_{label.lower()}": float(
                        transition_probability[:, index].mean()
                    )
                    for index, label in enumerate(LABELS)
                },
            }
        )

    metrics = pd.DataFrame(metric_rows)
    per_class = pd.DataFrame(class_rows)
    confusion = pd.DataFrame(confusion_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    transition_predictions = pd.concat(transition_frames, ignore_index=True)
    transition_summary = pd.DataFrame(transition_rows)
    province_summary = pd.DataFrame(province_rows)
    big_pine = predictions.loc[
        predictions["province_proxy"].eq("Big Pine volcanic field")
    ].copy()
    big_pine_publication = (
        big_pine.groupby(["model", "publication_group"], dropna=False)
        .agg(
            n=("record_id", "size"),
            cib_recall=("predicted", lambda values: values.eq("CIB").mean()),
            arc_share=("predicted", lambda values: values.eq("ARC").mean()),
            mean_prob_cib=("prob_cib", "mean"),
            mean_prob_arc=("prob_arc", "mean"),
            mean_outside_train_1_99_n=("outside_train_1_99_n", "mean"),
        )
        .reset_index()
    )
    metrics.to_csv(OUT / "multiclass_external_metrics.csv", index=False)
    per_class.to_csv(OUT / "multiclass_external_per_class.csv", index=False)
    confusion.to_csv(OUT / "multiclass_external_confusion.csv", index=False)
    predictions.to_csv(OUT / "multiclass_external_predictions.csv", index=False)
    transition_predictions.to_csv(
        OUT / "multiclass_external_backarc_predictions.csv", index=False
    )
    transition_summary.to_csv(
        OUT / "multiclass_external_backarc_summary.csv", index=False
    )
    province_summary.to_csv(
        OUT / "multiclass_external_by_province.csv", index=False
    )
    big_pine_publication.to_csv(
        OUT / "multiclass_external_big_pine_by_publication.csv", index=False
    )

    internal = pd.read_csv(OUT / "petdb_common18_figshare_internal_metrics.csv")
    manifest = {
        "training_dataset": "data/processed/petdb_primary4_v0_1.parquet",
        "training_rows": len(train),
        "features": FEATURES,
        "models": MODEL_NAMES,
        "external_class_counts": strict["actual_label"].value_counts().to_dict(),
        "external_province_proxies": strict.groupby("actual_label")[
            "province_proxy"
        ].unique().map(list).to_dict(),
        "external_n": len(strict),
        "morb_provenance_status": (
            "The SWIR repository cohort remains provisional because no linked journal DOI "
            "was found. The second MORB proxy is publisher-primary and linked to "
            "Zhong et al. (2019), doi:10.3390/min9110659."
        ),
        "domain_check": (
            "Per sample, count observed common18 log10 feature values outside the PetDB "
            "training 1st-99th percentile envelope."
        ),
        "independence_limit": (
            "All four labels now contain two province proxies, which is the minimum "
            "symmetric external design. Two clusters per label remain insufficient for "
            "stable province-cluster confidence intervals; province-level outcomes are "
            "therefore reported descriptively rather than treated as independent samples."
        ),
        "transition_policy": (
            "Vate Trough samples are unscored and reported separately as an initial "
            "back-arc-rift mechanism pressure test."
        ),
        "internal_reference": internal.to_dict("records"),
    }
    (OUT / "multiclass_external_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    print(per_class.to_string(index=False))
    print(transition_summary.to_string(index=False))


if __name__ == "__main__":
    main()
