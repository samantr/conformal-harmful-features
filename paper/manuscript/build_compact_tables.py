"""Build compact Phase 9 main-text tables from frozen reporting tables.

This script is reporting-only. It reads the already generated Phase 9 canonical CSVs
and audited H1 overlap CSV, performs no model fitting/calibration/selection/inference,
and writes compact manuscript CSV/TeX views. Full generated tables remain unchanged.
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FULL = ROOT / "paper" / "phase9_outputs" / "tables"
AUDIT = ROOT / "paper" / "phase9_audit"
OUT = Path(__file__).resolve().parent / "tables"
OUT.mkdir(parents=True, exist_ok=True)

DATASET = {"dry_bean": "Dry Bean", "covertype": "Covertype", "human_activity_recognition": "HAR"}
MODEL = {"logistic_regression": "LR", "small_neural_network": "Small NN"}
PATH = {"conformal_harm_one_shot": "One-shot", "conformal_harm_recursive": "Recursive"}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def num(value: str) -> float:
    return float(value)


def p_fmt(value: float) -> str:
    return f"{value:.8f}" if value < 1e-4 else f"{value:.6f}"


def ci(lo: float, hi: float) -> str:
    return f"[{lo:.6f}, {hi:.6f}]"


def write_csv(path: Path, records: list[dict[str, object]]) -> None:
    if not records:
        raise ValueError(f"No records for {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def tex_escape(value: object) -> str:
    text = str(value)
    for old, new in (("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        text = text.replace(old, new)
    return text


def write_tex(path: Path, caption: str, label: str, headers: list[str], body: list[list[object]]) -> None:
    lines = [r"\begin{table*}[t]", r"\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", r"\small", r"\begin{tabular}{" + "l" * len(headers) + "}", r"\hline", " & ".join(headers) + r" \\", r"\hline"]
    lines += [" & ".join(tex_escape(value) for value in record) + r" \\" for record in body]
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compact_t2() -> None:
    src = rows(FULL / "table_t2_primary_results.csv")
    all_ref = {(r["dataset"], r["model"]): r for r in src if r["method"] == "All features"}
    selected = [r for r in src if r["method"] in {"One-shot", "Recursive"}]
    assert len(all_ref) == 6 and len(selected) == 12
    records, body = [], []
    for r in selected:
        ref = all_ref[(r["dataset"], r["model"])]
        record = {"dataset": r["dataset"], "model": r["model"], "path": r["method"], "all_mean_size": num(ref["mean_size"]), "selected_mean_size": num(r["mean_size"]), "size_reduction": num(r["size_mean_difference"]), "size_ci_lower": num(r["size_ci_lower"]), "size_ci_upper": num(r["size_ci_upper"]), "size_holm_sign_flip_p": num(r["size_holm_sign_flip_pvalue"]), "all_accuracy": num(ref["accuracy"]), "selected_accuracy": num(r["accuracy"]), "accuracy_difference": num(r["accuracy_mean_difference"]), "accuracy_ci_lower": num(r["accuracy_ci_lower"]), "accuracy_ci_upper": num(r["accuracy_ci_upper"]), "accuracy_holm_sign_flip_p": num(r["accuracy_holm_sign_flip_pvalue"]), "all_coverage": num(ref["coverage"]), "selected_coverage": num(r["coverage"])}
        records.append(record)
        body.append([r["dataset"], r["model"], r["method"], f'{record["all_mean_size"]:.4f} $\\to$ {record["selected_mean_size"]:.4f}', f'{record["size_reduction"]:.4f} {ci(record["size_ci_lower"], record["size_ci_upper"])}; p={p_fmt(record["size_holm_sign_flip_p"])}', f'{record["all_accuracy"]:.4f} $\\to$ {record["selected_accuracy"]:.4f}', f'{record["accuracy_difference"]:.4f} {ci(record["accuracy_ci_lower"], record["accuracy_ci_upper"])}; p={p_fmt(record["accuracy_holm_sign_flip_p"])}', f'{record["all_coverage"]:.4f} $\\to$ {record["selected_coverage"]:.4f}'])
    write_csv(OUT / "table_t2_primary_compact.csv", records)
    write_tex(OUT / "table_t2_primary_compact.tex", r"Primary Base-APS results at $\alpha=0.10$ with tuning accuracy-loss allowance 0.01. Set-size reduction is all-feature minus selected mean size (positive favors removal); accuracy difference is selected minus all-feature accuracy (positive favors removal). Intervals are paired Student-$t$ 95\% CIs and $p$ values are Holm-adjusted exact sign-flip tests. Coverage is descriptive; no new coverage interval or noninferiority claim is implied.", "tab:t2-primary-compact", ["Dataset","Model","Path",r"Mean size all $\to$ selected",r"$\Delta$ size [95\% CI]; Holm p",r"Accuracy all $\to$ selected",r"$\Delta$ accuracy [95\% CI]; Holm p",r"Coverage all $\to$ selected"], body)


def compact_h1() -> None:
    overlap = rows(AUDIT / "h1_overlap_verified.csv")
    stability = rows(FULL / "table_t3b_rank_stability.csv")
    stable = {(r["dataset"], r["model"], r["method"]): r for r in stability if r["stability_type"] == "progressive_removal_path" and r["top_k"] == "1"}
    grouped: dict[tuple[str,str,str], list[dict[str,str]]] = {}
    for r in overlap:
        key = (DATASET[r["dataset"]], MODEL[r["model"]], PATH[r["proposed"]])
        grouped.setdefault(key, []).append(r)
    assert len(grouped) == 12 and all(len(v) == 5 for v in grouped.values())
    records, body = [], []
    for key in sorted(grouped):
        d,m,path = key; group = grouped[key]
        n = {int(r["n"]) for r in group}; removed = {num(r["mean_removed"]) for r in group}
        assert len(n) == 1 and len(removed) == 1
        s = stable[(d,m,path)]; vals = [num(r["jaccard_mean"]) for r in group]
        record = {"dataset":d,"model":m,"path":path,"within_seed_overlap_n":next(iter(n)),"mean_removed":next(iter(removed)),"jaccard_vs_standards_min":min(vals),"jaccard_vs_standards_max":max(vals),"across_seed_pair_count":int(s["seed_pairs"]),"top1_jaccard_across_seeds":num(s["jaccard_mean"])}
        records.append(record); body.append([d,m,path,record["within_seed_overlap_n"],f'{record["mean_removed"]:.2f}',f'{record["jaccard_vs_standards_min"]:.4f}--{record["jaccard_vs_standards_max"]:.4f}',record["across_seed_pair_count"],f'{record["top1_jaccard_across_seeds"]:.4f}'])
    write_csv(OUT / "table_t3_h1_compact.csv", records)
    write_tex(OUT / "table_t3_h1_compact.tex", r"Selection distinctness and identity stability. Within-seed Jaccard ranges summarize the five size-matched conventional selectors at the same seed and removed-feature count; zero-removal cases are excluded only from this overlap estimand and the conditional $n$ is shown. Across-seed top-1 Jaccard is computed over overlapping seed pairs (45 for Covertype, 190 for the 20-seed datasets), which are not independent replications.", "tab:t3-h1-compact", ["Dataset","Model","Path","Overlap n","Mean removed","Within-seed Jaccard range","Seed pairs","Across-seed top-1 Jaccard"], body)


def compact_baselines() -> None:
    src = [r for r in rows(FULL / "table_t3a_matched_baselines.csv") if r["metric"] == "mean_size"]
    grouped: dict[tuple[str,str,str],list[dict[str,str]]] = {}
    for r in src: grouped.setdefault((r["dataset"],r["model"],r["method"]),[]).append(r)
    assert len(grouped) == 12 and all(len(v) == 5 for v in grouped.values())
    records, body = [], []
    for key in sorted(grouped):
        d,m,path = key; group = grouped[key]
        wins = [r["reference_method"] for r in group if num(r["mean_difference"]) > 0 and num(r["holm_sign_flip_pvalue"]) < 0.05]
        effects = [num(r["mean_difference"]) for r in group]; n = {int(r["n_pairs"]) for r in group}; assert len(n) == 1
        record = {"dataset":d,"model":m,"path":path,"n_pairs":next(iter(n)),"target_size_min":min(int(r["target_size_min"]) for r in group),"target_size_max":max(int(r["target_size_max"]) for r in group),"significant_size_wins_of_5":len(wins),"significant_comparators":",".join(wins) if wins else "None","size_gain_min":min(effects),"size_gain_max":max(effects)}
        records.append(record); body.append([d,m,path,record["n_pairs"],f'{record["target_size_min"]}--{record["target_size_max"]}',f'{len(wins)}/5',record["significant_comparators"],f'{record["size_gain_min"]:.4f}--{record["size_gain_max"]:.4f}'])
    write_csv(OUT / "table_t3_matched_baselines_compact.csv", records)
    write_tex(OUT / "table_t3_matched_baselines_compact.tex", r"Size-matched comparator summary. Positive effects mean smaller mean prediction sets for the proposed subset. Wins count the five prespecified selectors whose supplied Holm-adjusted exact sign-flip $p<0.05$ and whose effect favors the proposed method; this is a descriptive reduction of the five corrected contrasts, not a new test. Full intervals, effect sizes and $p$ values are retained in the supplement.", "tab:t3-baseline-compact", ["Dataset","Model","Path","n","Matched retained-feature range","Wins","Comparators beaten","Point-effect range"], body)


def compact_t4() -> None:
    src = [r for r in rows(FULL / "table_t4_scaling_interaction.csv") if r["scaling"] == "confts"]
    assert len(src) == 12
    records, body = [], []
    for r in src:
        record = {"dataset":r["dataset"],"model":r["model"],"path":r["method"],"n":int(r["n"]),"feature_gain_under_confts":num(r["feature_size_gain"]),"feature_ci_lower":num(r["feature_ci_lo"]),"feature_ci_upper":num(r["feature_ci_hi"]),"all_feature_confts_scaling_gain":num(r["scaling_size_gain_all"]),"scaling_ci_lower":num(r["scaling_ci_lo"]),"scaling_ci_upper":num(r["scaling_ci_hi"]),"combined_gain_vs_base_all":num(r["combined_size_gain_vs_base_all"]),"combined_ci_lower":num(r["comb_ci_lo"]),"combined_ci_upper":num(r["comb_ci_hi"]),"additive_excess":num(r["interaction"]),"interaction_ci_lower":num(r["int_ci_lo"]),"interaction_ci_upper":num(r["int_ci_hi"]),"selected_confts_mean_size":num(r["mean_size"]),"selected_confts_coverage":num(r["coverage"])}
        records.append(record); body.append([r["dataset"],r["model"],r["method"],f'{record["feature_gain_under_confts"]:.4f} {ci(record["feature_ci_lower"],record["feature_ci_upper"])}',f'{record["all_feature_confts_scaling_gain"]:.4f} {ci(record["scaling_ci_lower"],record["scaling_ci_upper"])}',f'{record["combined_gain_vs_base_all"]:.4f} {ci(record["combined_ci_lower"],record["combined_ci_upper"])}',f'{record["additive_excess"]:.4f} {ci(record["interaction_ci_lower"],record["interaction_ci_upper"])}',f'{record["selected_confts_mean_size"]:.4f}',f'{record["selected_confts_coverage"]:.4f}'])
    write_csv(OUT / "table_t4_scaling_compact.csv", records)
    write_tex(OUT / "table_t4_scaling_compact.tex", r"Frozen APS feature-removal $\times$ ConfTS interaction at $\alpha=0.10$. Feature gain is all-ConfTS minus selected-ConfTS size; scaling gain is all-Base minus all-ConfTS; combined gain is all-Base minus selected-ConfTS. Positive values indicate smaller sets. Additive excess is $B+C-A-D$; negative values indicate sub-additivity. H3 means were independently reconciled, but interval values remain supplied Phase 8 evidence because their original generator is unavailable.", "tab:t4-scaling-compact", ["Dataset","Model","Path",r"Feature gain [95\% CI]",r"Scaling gain [95\% CI]",r"Combined gain [95\% CI]",r"Additive excess [95\% CI]","Selected size","Selected coverage"], body)


def compact_t5() -> None:
    src = rows(FULL / "table_t5a_har_subject_effects.csv")
    grouped: dict[tuple[str,str],dict[str,dict[str,str]]] = {}
    for r in src: grouped.setdefault((r["model"],r["method"]),{})[r["metric"]] = r
    assert len(grouped) == 4 and all(set(v) == {"mean_size_reduction_vs_all","coverage_difference_vs_all"} for v in grouped.values())
    records, body = [], []
    for (model,path), metrics in sorted(grouped.items()):
        s = metrics["mean_size_reduction_vs_all"]; c = metrics["coverage_difference_vs_all"]
        for field in ("n_cells","n_seeds","n_subjects","bootstrap_repetitions"): assert s[field] == c[field]
        record = {"dataset":"HAR","model":model,"path":path,"n_cells":int(s["n_cells"]),"n_seeds":int(s["n_seeds"]),"n_unique_subjects":int(s["n_subjects"]),"size_reduction":num(s["mean_difference"]),"size_ci_lower":num(s["ci_lower"]),"size_ci_upper":num(s["ci_upper"]),"coverage_difference":num(c["mean_difference"]),"coverage_ci_lower":num(c["ci_lower"]),"coverage_ci_upper":num(c["ci_upper"]),"bootstrap_repetitions":int(s["bootstrap_repetitions"])}
        records.append(record); body.append([model,path,record["n_cells"],record["n_seeds"],record["n_unique_subjects"],f'{record["size_reduction"]:.4f} {ci(record["size_ci_lower"],record["size_ci_upper"])}',f'{record["coverage_difference"]:.4f} {ci(record["coverage_ci_lower"],record["coverage_ci_upper"])}',record["bootstrap_repetitions"]])
    write_csv(OUT / "table_t5_har_subject_compact.csv", records)
    write_tex(OUT / "table_t5_har_subject_compact.tex", r"HAR subject-disjoint subject-weighted effects. Positive size reduction favors removal; positive coverage difference means higher selected-subset coverage. Intervals are the exactly reproduced 95\% percentile intervals from the frozen two-way seed/subject bootstrap. Each contrast has 100 seed $\times$ subject cells from 20 seeds and 26 unique held-out subject identifiers; these cells and the 146,500 grid rows are not independent subjects.", "tab:t5-har-compact", ["Model","Path","Cells","Seeds","Unique subjects",r"Size reduction [95\% CI]",r"Coverage difference [95\% CI]","Bootstrap reps"], body)


if __name__ == "__main__":
    compact_t2(); compact_h1(); compact_baselines(); compact_t4(); compact_t5()
    expected = {"table_t2_primary_compact.csv":12,"table_t3_h1_compact.csv":12,"table_t3_matched_baselines_compact.csv":12,"table_t4_scaling_compact.csv":12,"table_t5_har_subject_compact.csv":4}
    for name,count in expected.items():
        actual = len(rows(OUT / name))
        if actual != count: raise AssertionError(f"{name}: {actual} != {count}")
    print("PASS: compact manuscript tables built from frozen reporting sources")
