import argparse
from pathlib import Path

from chf.experiments.phase8_runner import (
    aggregate_phase8,
    load_phase8_spec,
    phase8_run_plan,
    run_phase8,
    write_phase8_plan,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--plan-only",
        action="store_true",
        help="Validate and write the exact zero-fit execution plan",
    )
    action.add_argument(
        "--run",
        action="store_true",
        help="Launch the selected expensive dataset/seed units",
    )
    action.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Rebuild statistics from completed unit artifacts without fitting",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only manifest-compatible candidate/subset checkpoints",
    )
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[1]
    spec = load_phase8_spec(args.config)
    output_dir = args.output_dir or repository_root / "outputs" / spec["experiment_name"]
    if args.plan_only:
        plan = write_phase8_plan(
            spec,
            repository_root,
            output_dir,
            datasets=args.datasets,
            seeds=args.seeds,
        )
        print(plan.to_string(index=False))
        print("\nPhase 8 plan validated; no dataset was loaded and no model was fit.")
        return
    if args.aggregate_only:
        plan = phase8_run_plan(
            spec,
            repository_root,
            datasets=args.datasets,
            seeds=args.seeds,
        )
        aggregate_phase8(spec=spec, output_dir=output_dir, expected_plan=plan)
        print("Phase 8 statistics rebuilt without fitting any classifier.")
        return
    run_phase8(
        spec=spec,
        repository_root=repository_root,
        output_dir=output_dir,
        datasets=args.datasets,
        seeds=args.seeds,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
