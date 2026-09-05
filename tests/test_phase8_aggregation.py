import json
from pathlib import Path

import pandas as pd
import yaml

from chf.experiments.phase8_runner import aggregate_phase8


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _write_unit(root: Path, seed: int) -> dict[str, object]:
    relative = f"dry_bean/seed_{seed}"
    unit = root / relative
    proposed_dir = unit / "proposed_selection"
    proposed_dir.mkdir(parents=True)
    all_size = 1.5 + seed / 10_000
    result_rows = [
        {
            "dataset": "dry_bean",
            "model": "small_neural_network",
            "seed": seed,
            "method": "all_features",
            "target_size": 6,
            "selected_indices": json.dumps([0, 1, 2, 3, 4, 5]),
            "phase8_primary_selected": True,
            "alpha": 0.1,
            "scaling": "base",
            "score": "aps",
            "mean_size": all_size,
            "accuracy": 0.9,
            "mean_size_reduction_vs_all": 0.0,
            "accuracy_loss_vs_all": 0.0,
            "coverage_delta_vs_all": 0.0,
            "tuning_n_removed": None,
        }
    ]
    for method_index, method in enumerate(
        ("conformal_harm_one_shot", "conformal_harm_recursive")
    ):
        for removed in range(6):
            selected = list(range(6 - removed))
            primary = removed == 2
            reduction = 0.01 * removed + 0.001 * method_index
            result_rows.append(
                {
                    "dataset": "dry_bean",
                    "model": "small_neural_network",
                    "seed": seed,
                    "method": method,
                    "target_size": len(selected),
                    "selected_indices": json.dumps(selected),
                    "phase8_primary_selected": primary,
                    "alpha": 0.1,
                    "scaling": "base",
                    "score": "aps",
                    "mean_size": all_size - reduction,
                    "accuracy": 0.9 - 0.001 * removed,
                    "mean_size_reduction_vs_all": reduction,
                    "accuracy_loss_vs_all": 0.001 * removed,
                    "coverage_delta_vs_all": -0.0005 * removed,
                    "tuning_n_removed": removed,
                }
            )
    result_rows.append(
        {
            "dataset": "dry_bean",
            "model": "small_neural_network",
            "seed": seed,
            "method": "crfe",
            "target_size": 4,
            "selected_indices": json.dumps([0, 1, 2, 3]),
            "phase8_primary_selected": True,
            "alpha": 0.1,
            "scaling": "base",
            "score": "aps",
            "mean_size": all_size - 0.01,
            "accuracy": 0.897,
            "mean_size_reduction_vs_all": 0.01,
            "accuracy_loss_vs_all": 0.003,
            "coverage_delta_vs_all": -0.001,
            "tuning_n_removed": None,
        }
    )
    pd.DataFrame(result_rows).to_csv(unit / "phase8_results.csv", index=False)
    (unit / "phase8_unit_complete.json").write_text(
        json.dumps({"status": "PASS"}), encoding="utf-8"
    )

    choice_rows = []
    for method in ("conformal_harm_one_shot", "conformal_harm_recursive"):
        for limit in (0.0, 0.005, 0.01, 0.02):
            choice_rows.append(
                {
                    "dataset": "dry_bean",
                    "seed": seed,
                    "model": "small_neural_network",
                    "method": method,
                    "accuracy_loss_limit": limit,
                    "selected_indices": json.dumps([0, 1, 2, 3]),
                }
            )
    pd.DataFrame(choice_rows).to_csv(
        unit / "phase8_accuracy_loss_choices.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "model": "small_neural_network",
                "method": "crfe",
                "feature_ranking": json.dumps([5, 4, 3, 2, 1, 0]),
            }
        ]
    ).to_csv(unit / "baseline_selections.csv", index=False)
    path_rows = []
    for method in ("one_shot", "recursive"):
        for removed in range(6):
            path_rows.append(
                {
                    "model": "small_neural_network",
                    "method": method,
                    "n_removed": removed,
                    "n_features": 6 - removed,
                    "removed_feature": None if removed == 0 else f"f{removed}",
                }
            )
    pd.DataFrame(path_rows).to_csv(
        proposed_dir / "progressive_consensus_paths.csv", index=False
    )
    return {
        "dataset_key": "dry_bean",
        "seed": seed,
        "output_subdirectory": relative,
    }


def test_phase8_aggregation_builds_paired_sensitivity_and_rank_outputs(
    tmp_path,
) -> None:
    spec = yaml.safe_load(
        (REPOSITORY_ROOT / "configs" / "phase8_robustness.yaml").read_text(
            encoding="utf-8"
        )
    )
    plan = pd.DataFrame([_write_unit(tmp_path, seed) for seed in spec["seeds"]])

    aggregate_phase8(spec=spec, output_dir=tmp_path, expected_plan=plan)

    size = pd.read_csv(tmp_path / "phase8_paired_size_effects.csv")
    standards = pd.read_csv(tmp_path / "phase8_matched_standard_effects.csv")
    ranks = pd.read_csv(tmp_path / "phase8_rank_stability_summary.csv")
    assert size["n_pairs"].eq(10).all()
    assert set(standards["reference_method"]) == {"crfe"}
    assert standards["n_pairs"].eq(10).all()
    assert set(ranks["stability_type"]) == {
        "complete_harmfulness_ranking",
        "progressive_removal_path",
    }
    assert (tmp_path / "phase8_accuracy_loss_sensitivity_effects.csv").exists()
    assert (tmp_path / "phase8_subset_size_sensitivity_effects.csv").exists()
