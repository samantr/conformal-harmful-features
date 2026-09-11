"""Generate Phase 9 manuscript tables and figures from frozen Phase 8 outputs.

This is a deterministic reporting transformation. It never fits a model, selects a
feature, recalibrates a predictor, changes a grid, or writes to Phase 8 inputs.

Usage:
  python paper/generate_phase9_tables_figures.py \
    --inputs /path/to/final_summaries \
    --aggregate /path/to/phase8_all_results.csv \
    --audit paper/phase9_audit \
    --out paper/phase9_outputs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASET = {
    "dry_bean": "Dry Bean",
    "covertype": "Covertype",
    "human_activity_recognition": "HAR",
}
MODEL = {
    "logistic_regression": "LR",
    "small_neural_network": "Small NN",
}
METHOD = {
    "all_features": "All features",
    "conformal_harm_one_shot": "One-shot",
    "conformal_harm_recursive": "Recursive",
}
STANDARD = {
    "mutual_information": "MI",
    "permutation_importance": "Permutation",
    "rfe": "RFE",
    "shap": "SHAP",
    "crfe": "CRFE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def source(directory: Path, stem: str, suffix: str = ".csv") -> Path:
    matches = sorted(directory.glob(f"{stem}*{suffix}"))
    if len(matches) != 1:
        raise ValueError(f"Expected one {stem} source, found {matches}")
    return matches[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label_rows(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column, mapping in (("dataset", DATASET), ("model", MODEL), ("method", METHOD)):
        if column in result:
            result[column] = result[column].map(mapping).fillna(result[column])
    if "reference_method" in result:
        result["reference_method"] = result["reference_method"].map(STANDARD).fillna(
            result["reference_method"]
        )
    return result


def write_table(frame: pd.DataFrame, stem: Path, caption: str, label: str) -> None:
    frame.to_csv(stem.with_suffix(".csv"), index=False)
    def tex(value: object) -> str:
        if pd.isna(value):
            return "--"
        if isinstance(value, (float, np.floating)):
            return f"{value:.6f}"
        text = str(value)
        for old, new in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
            text = text.replace(old, new)
        return text

    columns = "l" * len(frame.columns)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{columns}}}",
        r"\hline",
        " & ".join(tex(column) for column in frame.columns) + r" \\",
        r"\hline",
    ]
    lines.extend(" & ".join(tex(value) for value in row) + r" \\" for row in frame.itertuples(index=False, name=None))
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    stem.with_suffix(".tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_figure(figure: plt.Figure, stem: Path) -> None:
    for suffix, options in (
        (".png", {"format": "png", "dpi": 220}),
        (".pdf", {"format": "pdf"}),
    ):
        target = stem.with_suffix(suffix)
        temporary = target.with_suffix(suffix + ".tmp")
        figure.savefig(temporary, bbox_inches="tight", **options)
        if temporary.stat().st_size < 1_000:
            raise RuntimeError(f"Incomplete figure export: {temporary}")
        os.replace(temporary, target)
    plt.close(figure)


def effect_label(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["dataset"].map(DATASET)
        + " · "
        + frame["model"].map(MODEL)
        + " · "
        + frame["method"].map(METHOD)
    )


def forest(
    ax: plt.Axes, frame: pd.DataFrame, title: str, *, show_labels: bool = True
) -> None:
    y = np.arange(len(frame))
    x = frame["mean_difference"].to_numpy()
    lower = x - frame["ci_lower"].to_numpy()
    upper = frame["ci_upper"].to_numpy() - x
    ax.errorbar(x, y, xerr=np.vstack([lower, upper]), fmt="o", capsize=2.5, lw=1)
    ax.axvline(0, color="0.4", lw=0.8)
    ax.set_title(title)
    ax.set_yticks(y, effect_label(frame) if show_labels else [""] * len(frame))
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)


def make_f1(
    size_effects: pd.DataFrame,
    accuracy_effects: pd.DataFrame,
    t2: pd.DataFrame,
    output: Path,
) -> None:
    order = ["dry_bean", "covertype", "human_activity_recognition"]
    sort = {value: index for index, value in enumerate(order)}
    for frame in (size_effects, accuracy_effects):
        frame["_dataset_order"] = frame.dataset.map(sort)
        frame["_model_order"] = frame.model.map({"small_neural_network": 0, "logistic_regression": 1})
        frame["_method_order"] = frame.method.map(
            {"conformal_harm_one_shot": 0, "conformal_harm_recursive": 1}
        )
        frame.sort_values(["_dataset_order", "_model_order", "_method_order"], inplace=True)
    coverage = t2[t2.method.ne("all_features")].copy()
    coverage["mean_difference"] = coverage["coverage_delta_vs_all"]
    coverage["_dataset_order"] = coverage.dataset.map(sort)
    coverage["_model_order"] = coverage.model.map(
        {"small_neural_network": 0, "logistic_regression": 1}
    )
    coverage["_method_order"] = coverage.method.map(
        {"conformal_harm_one_shot": 0, "conformal_harm_recursive": 1}
    )
    coverage.sort_values(["_dataset_order", "_model_order", "_method_order"], inplace=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.2), sharey=True)
    forest(axes[0], size_effects, "Set-size reduction (95% paired t CI)")
    forest(axes[1], accuracy_effects, "Accuracy difference (95% paired t CI)")
    y = np.arange(len(coverage))
    axes[2].scatter(coverage.mean_difference, y, s=25)
    axes[2].axvline(0, color="0.4", lw=0.8)
    axes[2].set_title("Coverage difference (descriptive mean)")
    axes[2].grid(axis="x", alpha=0.25)
    axes[2].set_yticks(y, effect_label(coverage))
    axes[2].invert_yaxis()
    fig.suptitle("Primary Base APS effects at α = 0.10")
    fig.text(
        0.5,
        0.01,
        "Positive set-size values favor removal. Coverage is shown without a newly computed interval.",
        ha="center",
        fontsize=8,
    )
    save_figure(fig, output / "figure_f1_primary_effects")


def make_f2(overlap: pd.DataFrame, stability: pd.DataFrame, output: Path) -> None:
    overlap = overlap.copy()
    overlap["row"] = effect_label(overlap.rename(columns={"proposed": "method"}))
    overlap["standard_label"] = overlap.standard.map(STANDARD)
    left = overlap.pivot(index="row", columns="standard_label", values="jaccard_mean")
    columns = [name for name in STANDARD.values() if name in left.columns]
    left = left[columns]
    stable = stability[
        stability.method.isin(METHOD) & stability.top_k.eq(1)
    ].copy()
    stable["row"] = effect_label(stable)
    right = stable.pivot_table(index="row", columns="stability_type", values="jaccard_mean")
    common = sorted(set(left.index) | set(right.index))
    left = left.reindex(common)
    right = right.reindex(common)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, max(6, 0.35 * len(common))),
        sharey=True,
        constrained_layout=True,
        gridspec_kw={"width_ratios": [4, 1.2]},
    )
    image = None
    for ax, matrix, title in (
        (axes[0], left, "Within-seed removed-set Jaccard"),
        (axes[1], right, "Across-seed top-1 Jaccard"),
    ):
        values = matrix.to_numpy(dtype=float)
        image = ax.imshow(values, aspect="auto", vmin=0, vmax=0.40)
        ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=35, ha="right")
        ax.set_yticks(range(len(matrix.index)), matrix.index)
        ax.set_title(title)
    axes[1].tick_params(labelleft=False)
    assert image is not None
    fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02, label="Mean Jaccard")
    fig.suptitle("Selection distinctness and identity stability are separate properties")
    save_figure(fig, output / "figure_f2_overlap_stability")


def sensitivity_figure(
    allowance: pd.DataFrame,
    removal: pd.DataFrame,
    model: str,
    output: Path,
    supplement: bool,
) -> None:
    metrics = [
        ("mean_size_reduction_vs_all", "Set-size reduction"),
        ("accuracy_loss_vs_all", "Accuracy loss"),
        ("coverage_delta_vs_all", "Coverage difference"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex="col")
    colors = dict(zip(DATASET, plt.get_cmap("tab10").colors, strict=False))
    styles = {"conformal_harm_one_shot": "-", "conformal_harm_recursive": "--"}
    for row, (metric, title) in enumerate(metrics):
        for column, (frame, x, xlabel) in enumerate(
            (
                (allowance, "accuracy_loss_limit", "Allowed tuning accuracy loss"),
                (removal, "n_removed", "Frozen removed-feature count"),
            )
        ):
            ax = axes[row, column]
            subset = frame[(frame.model == model) & (frame.metric == metric)]
            for (dataset, method), group in subset.groupby(["dataset", "method"], sort=False):
                group = group.sort_values(x)
                ax.plot(
                    group[x],
                    group.mean_difference,
                    marker="o",
                    ms=3,
                    lw=1,
                    color=colors[dataset],
                    linestyle=styles[method],
                    label=f"{DATASET[dataset]} · {METHOD[method]}",
                )
            ax.axhline(0, color="0.5", lw=0.7)
            ax.set_ylabel(title)
            if row == 2:
                ax.set_xlabel(xlabel)
            ax.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle(f"Frozen sensitivity curves — {MODEL[model]}")
    fig.subplots_adjust(bottom=0.13, hspace=0.3)
    name = "figure_s3_sensitivity_lr" if supplement else "figure_f3_sensitivity_small_nn"
    save_figure(fig, output / name)


def make_f4(primary_grid: pd.DataFrame, output: Path) -> None:
    combinations = [
        (dataset, model)
        for dataset in ("dry_bean", "covertype", "human_activity_recognition")
        for model in ("small_neural_network", "logistic_regression")
    ]
    fig, axes = plt.subplots(6, 2, figsize=(11, 18))
    x_methods = ["all_features", "conformal_harm_one_shot", "conformal_harm_recursive"]
    scalings = ["base", "ts", "confts"]
    for row, (dataset, model) in enumerate(combinations):
        group = primary_grid[(primary_grid.dataset == dataset) & (primary_grid.model == model)]
        for column, metric in enumerate(("mean_size", "coverage")):
            ax = axes[row, column]
            for scaling in scalings:
                means = (
                    group[group.scaling == scaling]
                    .groupby("method")[metric]
                    .mean()
                    .reindex(x_methods)
                )
                ax.plot(range(3), means, marker="o", label=scaling.upper())
            ax.set_xticks(range(3), [METHOD[value] for value in x_methods], rotation=15)
            ax.set_ylabel("Mean set size" if metric == "mean_size" else "Coverage")
            ax.set_title(f"{DATASET[dataset]} · {MODEL[model]}")
            ax.grid(alpha=0.2)
            if metric == "coverage":
                ax.axhline(0.90, color="firebrick", lw=0.8, linestyle=":")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.suptitle("Scaling interaction: Base APS, TS and ConfTS at α = 0.10")
    fig.subplots_adjust(bottom=0.05, hspace=0.5)
    save_figure(fig, output / "figure_f4_scaling_interaction")


def make_f5(subject_effects: pd.DataFrame, subjects: pd.DataFrame, output: Path) -> None:
    size = subject_effects[subject_effects.metric.eq("mean_size_reduction_vs_all")].copy()
    coverage = subject_effects[subject_effects.metric.eq("coverage_difference_vs_all")].copy()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.7))
    forest(axes[0], size, "Subject-weighted size reduction")
    forest(
        axes[1],
        coverage,
        "Subject-weighted coverage difference",
        show_labels=False,
    )
    order = ["all_features", "conformal_harm_one_shot", "conformal_harm_recursive"]
    positions = []
    values = []
    labels = []
    position = 1
    for model in ("small_neural_network", "logistic_regression"):
        for method in order:
            group = subjects[(subjects.model == model) & (subjects.method == method)]
            positions.append(position)
            values.append(group.coverage.to_numpy())
            labels.append(
                f"{'NN' if model == 'small_neural_network' else 'LR'}\n"
                f"{METHOD[method]}"
            )
            position += 1
        position += 0.7
    axes[2].boxplot(values, positions=positions, widths=0.55, showfliers=False)
    axes[2].set_xticks(positions, labels, rotation=20, ha="right", fontsize=7.5)
    axes[2].axhline(0.90, color="firebrick", lw=0.9, linestyle=":")
    axes[2].set_ylabel("Subject-cell coverage")
    axes[2].set_title("Primary subject-cell coverage distribution")
    axes[2].grid(axis="y", alpha=0.2)
    fig.suptitle("HAR subject-level uncertainty under subject-disjoint evaluation")
    fig.subplots_adjust(bottom=0.22, wspace=0.35)
    fig.text(
        0.5,
        0.005,
        "Intervals use the frozen two-way seed/subject bootstrap; grid rows and windows are not independent subjects.",
        ha="center",
        fontsize=8,
    )
    save_figure(fig, output / "figure_f5_har_subject_effects")


def main() -> None:
    args = parse_args()
    tables = args.out / "tables"
    figures = args.out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
            "pdf.fonttype": 42,
        }
    )

    paths = {
        name: source(args.inputs, name, suffix)
        for name, suffix in {
            "phase8_protocol": ".json",
            "phase8_precision_extension_aggregate": ".json",
            "phase8_paired_size_effects": ".csv",
            "phase8_paired_accuracy_effects": ".csv",
            "phase8_matched_standard_effects": ".csv",
            "phase8_accuracy_loss_sensitivity_effects": ".csv",
            "phase8_subset_size_sensitivity_effects": ".csv",
            "phase8_rank_stability_summary": ".csv",
            "phase8_h3_aps_summary": ".csv",
            "phase8_har_subject_effects": ".csv",
            "phase8_numerical_diagnostics": ".csv",
        }.items()
    }
    paths.update(
        {
            "aggregate": args.aggregate,
            "primary_absolute": args.audit / "primary_absolute.csv",
            "h1_overlap": args.audit / "h1_overlap_verified.csv",
            "har_primary_subjects": args.audit / "har_primary_subjects.csv",
            "har_subject_effects_reproduced": args.audit
            / "har_subject_effects_reproduced.csv",
            "audit": args.audit / "audit.json",
            "har_verification": args.audit / "har_verification.json",
        }
    )

    aggregate_columns = [
        "dataset",
        "seed",
        "model",
        "method",
        "phase8_primary_selected",
        "alpha",
        "scaling",
        "score",
        "mean_size",
        "accuracy",
        "coverage",
    ]
    aggregate = pd.concat(
        pd.read_csv(args.aggregate, usecols=aggregate_columns, chunksize=5000),
        ignore_index=True,
    )
    primary_grid = aggregate[
        aggregate.method.isin(METHOD)
        & (aggregate.method.eq("all_features") | aggregate.phase8_primary_selected)
        & aggregate.alpha.eq(0.10)
        & aggregate.score.eq("aps")
    ].copy()

    protocol = json.loads(paths["phase8_protocol"].read_text())
    extension = json.loads(paths["phase8_precision_extension_aggregate"].read_text())
    design = []
    for dataset in ("dry_bean", "covertype", "human_activity_recognition"):
        group = aggregate[aggregate.dataset == dataset]
        design.append(
            {
                "dataset": DATASET[dataset],
                "paired_seeds": group.seed.nunique(),
                "seed_range": f"{group.seed.min()}–{group.seed.max()}",
                "dataset_seed_units": group[["dataset", "seed"]].drop_duplicates().shape[0],
                "aggregate_rows": len(group),
                "subject_rows": 146500 if dataset == "human_activity_recognition" else 0,
                "split_regime": "Subject-disjoint" if dataset == "human_activity_recognition" else "Stratified",
                "final_precision": "Met",
                "scientific_code": protocol["scientific_code_version"],
            }
        )
    t1 = pd.DataFrame(design)
    if extension["audited_units"] != t1.dataset_seed_units.sum():
        raise ValueError("T1 unit count does not reconcile")
    write_table(t1, tables / "table_t1_design_audit", "Frozen Phase 8 design and audit population.", "tab:design-audit")

    absolute = pd.read_csv(paths["primary_absolute"])
    all_reference = absolute[absolute.method.eq("all_features")][
        ["dataset", "model", "mean_size", "accuracy", "coverage"]
    ].rename(
        columns={
            "mean_size": "all_mean_size",
            "accuracy": "all_accuracy",
            "coverage": "all_coverage",
        }
    )
    t2 = absolute.merge(all_reference, on=["dataset", "model"], validate="many_to_one")
    t2["coverage_delta_vs_all"] = t2.coverage - t2.all_coverage
    size_effects = pd.read_csv(paths["phase8_paired_size_effects"])
    accuracy_effects = pd.read_csv(paths["phase8_paired_accuracy_effects"])
    size_columns = ["dataset", "model", "method", "mean_difference", "ci_lower", "ci_upper", "cohens_dz", "holm_sign_flip_pvalue"]
    accuracy_columns = size_columns.copy()
    t2 = t2.merge(
        size_effects[size_columns].rename(
            columns={column: f"size_{column}" for column in size_columns[3:]}
        ),
        on=["dataset", "model", "method"],
        how="left",
        validate="one_to_one",
    ).merge(
        accuracy_effects[accuracy_columns].rename(
            columns={column: f"accuracy_{column}" for column in accuracy_columns[3:]}
        ),
        on=["dataset", "model", "method"],
        how="left",
        validate="one_to_one",
    )
    write_table(label_rows(t2), tables / "table_t2_primary_results", "Primary Base APS results at alpha 0.10.", "tab:primary-results")

    matched = pd.read_csv(paths["phase8_matched_standard_effects"])
    write_table(label_rows(matched), tables / "table_t3a_matched_baselines", "Matched-size baseline comparisons.", "tab:matched-baselines")
    stability = pd.read_csv(paths["phase8_rank_stability_summary"])
    write_table(label_rows(stability), tables / "table_t3b_rank_stability", "Across-seed feature-identity stability.", "tab:rank-stability")

    h3 = pd.read_csv(paths["phase8_h3_aps_summary"])
    scaling_absolute = (
        primary_grid.groupby(["dataset", "model", "method", "scaling"])[["mean_size", "coverage"]]
        .mean()
        .reset_index()
    )
    t4 = h3.merge(scaling_absolute, on=["dataset", "model", "method", "scaling"], how="left", validate="one_to_one")
    write_table(label_rows(t4), tables / "table_t4_scaling_interaction", "APS scaling interaction and absolute outcomes.", "tab:scaling-interaction")

    subject_effects = pd.read_csv(paths["har_subject_effects_reproduced"])
    diagnostics = pd.read_csv(paths["phase8_numerical_diagnostics"])
    write_table(label_rows(subject_effects), tables / "table_t5a_har_subject_effects", "HAR two-way seed/subject bootstrap effects.", "tab:har-subject-effects")
    write_table(label_rows(diagnostics), tables / "table_t5b_numerical_diagnostics", "Frozen numerical-boundary diagnostics.", "tab:numerical-diagnostics")

    overlap = pd.read_csv(paths["h1_overlap"])
    allowance = pd.read_csv(paths["phase8_accuracy_loss_sensitivity_effects"])
    removal = pd.read_csv(paths["phase8_subset_size_sensitivity_effects"])
    subjects = pd.read_csv(paths["har_primary_subjects"])
    make_f1(size_effects, accuracy_effects, t2, figures)
    make_f2(overlap, stability, figures)
    sensitivity_figure(allowance, removal, "small_neural_network", figures, False)
    sensitivity_figure(allowance, removal, "logistic_regression", figures, True)
    make_f4(primary_grid, figures)
    make_f5(subject_effects, subjects, figures)

    manifest = {
        "status": "PASS",
        "scope": "Frozen-output reporting only; no model fitting, recalibration, retuning, feature reselection, new seeds, or new tests.",
        "primary_filter": "Base APS, alpha 0.10, all features plus phase8_primary_selected proposed paths.",
        "input_hashes": {
            name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for name, path in paths.items()
        },
        "tables": sorted(path.name for path in tables.iterdir()),
        "figures": sorted(path.name for path in figures.iterdir()),
        "boundaries": [
            "F1 coverage panel is descriptive and has no newly computed interval.",
            "F3 preserves every frozen allowance/removal cell; LR is separated as a supplement for legibility.",
            "F4 shows coverage beside size and retains HAR undercoverage.",
            "F5 uses the frozen two-way bootstrap summaries; subject/grid rows are not independent replicates.",
            "H3 intervals are supplied Phase 8 values; their missing generator is not replaced here.",
        ],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "tables": len(manifest["tables"]), "figures": len(manifest["figures"])}, indent=2))


if __name__ == "__main__":
    main()
