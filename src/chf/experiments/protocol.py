import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from chf.conformal import calibrate_threshold, evaluate_prediction_sets
from chf.data import ControlledSyntheticDataset, make_controlled_multiclass
from chf.metrics import classification_metrics, conditional_coverage_metrics
from chf.scaling import (
    probabilities_from_logits,
    probability_diagnostics,
    tune_confts,
    tune_temperature,
)


def split_id(split: Any) -> str:
    """Return a stable identifier for a four-way split."""
    digest = hashlib.sha256()
    for name in ("train", "tune", "calibration", "test"):
        digest.update(name.encode("utf-8"))
        digest.update(np.asarray(getattr(split, name), dtype=np.int64).tobytes())
    return digest.hexdigest()[:16]


def code_version(repository_root: Path) -> str:
    """Return the current Git revision, marking an uncommitted worktree."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return f"{commit}-dirty" if dirty else commit
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def dataset_from_config(config: Mapping[str, Any]) -> ControlledSyntheticDataset:
    """Recreate the controlled dataset from a resolved experiment config."""
    dataset_config = config["dataset"]
    feature_config = dataset_config["features"]
    return make_controlled_multiclass(
        n_samples=int(dataset_config["n_samples"]),
        n_classes=int(dataset_config["n_classes"]),
        n_strong=int(feature_config["strong"]),
        n_weak=int(feature_config["weak"]),
        n_redundant=int(feature_config["redundant"]),
        n_noise=int(feature_config["noise"]),
        seed=int(config["seed"]),
        strong_signal=float(dataset_config.get("strong_signal", 1.0)),
        weak_signal=float(dataset_config.get("weak_signal", 0.25)),
        redundant_noise=float(dataset_config.get("redundant_noise", 0.15)),
    )


def evaluate_logits(
    *,
    logits_tune: np.ndarray,
    logits_calibration: np.ndarray,
    logits_test: np.ndarray,
    labels_tune: np.ndarray,
    labels_calibration: np.ndarray,
    labels_test: np.ndarray,
    config: Mapping[str, Any],
    calibration_uniforms: np.ndarray,
    test_uniforms: np.ndarray,
    seed: int,
) -> list[dict[str, Any]]:
    """Evaluate one fitted-model/data combination under the shared CP protocol.

    Temperature choices use only ``tune``. Each frozen choice receives a fresh
    threshold from ``calibration`` and is evaluated once on ``test``. Explicit
    uniforms make every randomized APS/RAPS comparison paired.
    """
    conformal_config = config["conformal"]
    alpha = float(conformal_config["alpha"])
    temperature_grid = [float(value) for value in config["temperature"]["grid"]]
    ece_bins = int(config.get("metrics", {}).get("ece_bins", 15))
    min_sscv_size = int(
        config.get("metrics", {}).get("min_sscv_stratum_size", 25)
    )
    raps_lambda = float(conformal_config["raps_lambda"])
    raps_k_reg = int(conformal_config["raps_k_reg"])
    ts_result = tune_temperature(logits_tune, labels_tune, temperature_grid)
    rows: list[dict[str, Any]] = []

    for score_name in conformal_config["scores"]:
        confts_result = tune_confts(
            logits_tune,
            labels_tune,
            alpha,
            score_name,
            temperature_grid,
            seed=seed + 30_001,
            threshold_fraction=float(
                config["temperature"].get("confts_threshold_fraction", 0.5)
            ),
            k_reg=raps_k_reg,
            lambda_reg=raps_lambda,
            reject_zero_probabilities=bool(
                config["temperature"].get("reject_zero_probabilities", True)
            ),
        )
        scaling_candidates = (
            ("base", 1.0, np.nan, np.nan, ()),
            ("ts", ts_result.temperature, ts_result.nll, np.nan, ()),
            (
                "confts",
                confts_result.temperature,
                np.nan,
                confts_result.loss,
                confts_result.rejected_temperatures,
            ),
        )
        for (
            scaling_name,
            temperature,
            tuning_nll,
            confts_loss,
            rejected,
        ) in scaling_candidates:
            probability_calibration = probabilities_from_logits(
                logits_calibration, temperature
            )
            probability_test = probabilities_from_logits(logits_test, temperature)
            tau = calibrate_threshold(
                probability_calibration,
                labels_calibration,
                alpha,
                score_name,
                u_values=calibration_uniforms,
                k_reg=raps_k_reg,
                lambda_reg=raps_lambda,
            )
            prediction_set_result = evaluate_prediction_sets(
                probability_test,
                labels_test,
                tau,
                score_name,
                u_values=test_uniforms,
                k_reg=raps_k_reg,
                lambda_reg=raps_lambda,
            )
            classification = classification_metrics(
                probability_test, labels_test, ece_bins=ece_bins
            )
            conditional = conditional_coverage_metrics(
                prediction_set_result.included,
                labels_test,
                target_coverage=1 - alpha,
                min_sscv_stratum_size=min_sscv_size,
            )
            calibration_diagnostics = probability_diagnostics(
                probability_calibration
            )
            test_diagnostics = probability_diagnostics(probability_test)
            rows.append(
                {
                    "scaling": scaling_name,
                    "temperature": temperature,
                    "score": score_name,
                    "alpha": alpha,
                    "raps_lambda": raps_lambda,
                    "raps_k_reg": raps_k_reg,
                    "threshold": tau,
                    "accuracy": classification.accuracy,
                    "macro_f1": classification.macro_f1,
                    "nll": classification.nll,
                    "ece": classification.ece,
                    "coverage": prediction_set_result.coverage,
                    "mean_size": prediction_set_result.mean_size,
                    "median_size": prediction_set_result.median_size,
                    "size_p90": prediction_set_result.size_p90,
                    "empty_set_rate": prediction_set_result.empty_set_rate,
                    "full_set_rate": prediction_set_result.full_set_rate,
                    "sscv": conditional.sscv,
                    "class_coverage_gap": conditional.class_coverage_gap,
                    "class_coverage_max_deviation": conditional.class_coverage_max_deviation,
                    "class_coverages": json.dumps(conditional.class_coverages),
                    "tuning_nll": tuning_nll,
                    "confts_loss": confts_loss,
                    "rejected_temperatures": json.dumps(rejected),
                    "calibration_zero_count": calibration_diagnostics.zero_count,
                    "calibration_exactly_one_count": calibration_diagnostics.exactly_one_count,
                    "test_zero_count": test_diagnostics.zero_count,
                    "test_exactly_one_count": test_diagnostics.exactly_one_count,
                    "test_mean_max_probability": test_diagnostics.mean_max_probability,
                }
            )
    return rows
