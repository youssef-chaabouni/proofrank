import argparse
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

from proofrank.pairwise.io import print_markdown_table
from proofrank.pairwise.graph_utils import cluster_density_metrics, transitivity_violations
from proofrank.pairwise.loader import load_pairwise_records


DEFAULT_TARGET_MODELS = [
    "deepseek/deepseek_v32_think", "gemini/gemini-3-flash", "gemini/gemini-31-pro",
    "openai/gpt-54", "stepfun/3.5-flash", "glm/glm-5",
    "xai/grok-41-fast-reasoning", "moonshot/k25",
    "qwen/qwen35_397b_a17b_high", "openai/oss-120b",
]
DEFAULT_JUDGES = ["openai_oss-120b"]

DENSITY_METRIC_KEYS = [
    "n_models", "n_pairs", "n_known_pairs", "positive_rate",
    "n_components", "largest_component_size", "largest_component_frac",
    "intra_cluster_density", "modularity_like", "n_singletons",
]

def group_by_problem(records):
    same = defaultdict(dict)
    models = defaultdict(set)
    for r in records:
        if r["majority"] is None:
            continue
        key = (r["judge"], r["problem"])
        sol_idx_1 = int(r['source_file'].split('_sol')[1])
        sol_idx_2 = int(r['source_file'].split('_sol')[2].split('.')[0])
        same[key][(sol_idx_1, sol_idx_2)] = r["majority"]
        models[key].update([sol_idx_1, sol_idx_2])
    return same, models

def run_transitivity(records, args):
    same_map, models_map = group_by_problem(records)

    detail_rows = []
    all_violations = []

    for (judge, problem), same in same_map.items():
        models = sorted(models_map[(judge, problem)])
        n_ant, n_viol, viols = transitivity_violations(models, same)
        detail_rows.append({
            "judge": judge,
            "problem": problem,
            "n_models": len(models),
            "n_labeled_pairs": len(same),
            "n_antecedents": n_ant,
            "n_violations": n_viol,
            "transitivity_rate": (1.0 - n_viol / n_ant) if n_ant else None,
        })
        for x, y, z in viols:
            all_violations.append({
                "judge": judge, "problem": problem, "a": x, "b": y, "c": z,
            })

    by_judge = defaultdict(list)
    for row in detail_rows:
        by_judge[row["judge"]].append(row)

    summary_rows = []
    for judge, rows in by_judge.items():
        tot_ant = sum(r["n_antecedents"] for r in rows)
        tot_viol = sum(r["n_violations"] for r in rows)
        summary_rows.append({
            "judge": judge,
            "n_problems": len(rows),
            "total_antecedents": tot_ant,
            "total_violations": tot_viol,
            "transitivity_rate": (1.0 - tot_viol / tot_ant) if tot_ant else None,
        })

    print("## Transitivity summary")
    print_markdown_table(summary_rows, headers=[
        "judge", "n_problems", "total_antecedents",
        "total_violations", "transitivity_rate",
    ])

    show = getattr(args, "show_violations", 0)
    if show > 0 and all_violations:
        print(f"\n## First {min(show, len(all_violations))} violations")
        print_markdown_table(all_violations[:show],
                             headers=["judge", "problem", "a", "b", "c"])

def run_density(records, args) -> None:
    same_map, models_map = group_by_problem(records)

    detail_rows = []
    for (judge, problem), same in same_map.items():
        models = sorted(models_map[(judge, problem)])
        metrics = cluster_density_metrics(models, same)
        detail_rows.append({"judge": judge, "problem": problem, **metrics})

    by_judge = defaultdict(list)
    for row in detail_rows:
        by_judge[row["judge"]].append(row)

    summary_rows = []
    for judge, rows in by_judge.items():
        entry = {"judge": judge, "n_problems": len(rows)}
        for key in DENSITY_METRIC_KEYS:
            entry[f"mean_{key}"] = np.mean(
                [r[key] for r in rows if r[key] is not None]
            )
        summary_rows.append(entry)

    print()
    print("## Density summary (means over problems)")
    print_markdown_table(summary_rows, headers=[
        "judge", "n_problems", "mean_positive_rate",
        "mean_n_components", "mean_largest_component_frac",
        "mean_intra_cluster_density", "mean_modularity_like",
        "mean_n_singletons",
    ])

def run_consistency(records, args):
    detail_rows = []
    for r in records:
        votes = r["votes"]
        n = len(votes)
        if n == 0:
            continue
        maj = r["majority"]
        if maj is None:
            agree = max(votes.count(0), votes.count(1))
        else:
            agree = sum(1 for v in votes if v == maj)

        detail_rows.append({
            "judge": r["judge"],
            "problem": r["problem"],
            "model_a": r["model_a"],
            "model_b": r["model_b"],
            "n_votes": n,
            "n_agree_with_majority": agree,
            "consistency": agree / n,
            "majority": maj if maj is not None else "tie",
            "votes": "".join(str(v) for v in votes),
        })

    by_judge = defaultdict(list)
    for row in detail_rows:
        by_judge[row["judge"]].append(row)

    summary_rows = []
    for judge, rows in by_judge.items():
        cons = [r["consistency"] for r in rows]
        summary_rows.append({
            "judge": judge,
            "n_records": len(rows),
            "mean_consistency": np.mean(cons),
            "median_consistency": float(statistics.median(cons)) if cons else None,
            "unanimous_rate": sum(1 for c in cons if c == 1.0) / len(rows) if rows else None,
            "ties_rate": sum(1 for r in rows if r["majority"] == "tie") / len(rows) if rows else None,
        })

    print("\n## Consistency summary")
    print_markdown_table(summary_rows, headers=[
        "judge", "n_records", "mean_consistency",
        "median_consistency", "unanimous_rate", "ties_rate",
    ])

    bins = getattr(args, "histogram_bins", 5)
    edges = [0.5 + i * (0.5 / bins) for i in range(bins + 1)]
    hist_rows = []
    for judge, rows in by_judge.items():
        counter = Counter()
        for r in rows:
            c = r["consistency"]
            for i in range(bins):
                lo, hi = edges[i], edges[i + 1]
                in_bin = (lo <= c <= hi) if i == bins - 1 else (lo <= c < hi)
                if in_bin:
                    counter[i] += 1
                    break
        entry = {"judge": judge}
        for i in range(bins):
            closer = "]" if i == bins - 1 else ")"
            entry[f"[{edges[i]:.2f},{edges[i+1]:.2f}{closer}"] = counter.get(i, 0)
        hist_rows.append(entry)

    if hist_rows:
        print("\n## Consistency histogram (per judge)")
        headers = ["judge"] + [k for k in hist_rows[0] if k != "judge"]
        print_markdown_table(hist_rows, headers=headers)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--project", required=True)
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--judges", nargs="+", default=DEFAULT_JUDGES)
    parser.add_argument("--target-models", nargs="+", default=DEFAULT_TARGET_MODELS)
    parser.add_argument("--no-target-model-filter", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--show-violations", type=int, default=0)

    args = parser.parse_args()

    target = None if args.no_target_model_filter else set(args.target_models)

    print(f"Project: {args.project}")
    print(f"Judges: {', '.join(args.judges)}")
    print(f"Model filter: {'disabled' if target is None else f'{len(target)} models'}")

    records = load_pairwise_records(
        outputs_root=Path(args.outputs_root),
        project=args.project,
        judges=args.judges,
        target_models=None,
        verbose=args.verbose,
    )

    print(f"Loaded {len(records)} clustering records.")

    run_transitivity(records, args)
    run_density(records, args)
    run_consistency(records, args)



if __name__ == "__main__":
    sys.exit(main())