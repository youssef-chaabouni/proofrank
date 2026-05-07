import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from llmteach.pairwise.io import iter_json_files, print_markdown_table, write_csv
from llmteach.pairwise.loader import (
    ground_truth_record_builder,
    load_pairwise_records,
)


DEFAULT_TARGET_MODELS = [
    "deepseek/deepseek_v32_think",
    "gemini/gemini-3-flash",
    "gemini/gemini-31-pro",
    "openai/gpt-54",
    "stepfun/3.5-flash",
    "glm/glm-5",
    "xai/grok-41-fast-reasoning",
    "moonshot/k25",
    "qwen/qwen35_397b_a17b_high",
    "openai/oss-120b",
]

DEFAULT_PAIRWISE_JUDGES = [
    "xai_grok-41-fast-reasoning",
    "openai_oss-120b",
]

DEFAULT_REFERENCE_JUDGES = [
    "proofrank-1",
    "proofrank-2",
    "proofrank-3",
]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare pairwise LLM-judge outputs for a specific project "
                    "against human baseline judges."
    )
    p.add_argument("--project", required=True,
                   help="Project name under outputs/, e.g. 'pairwise_computation_tie'.")
    p.add_argument("--outputs-root", default="outputs")
    p.add_argument("--data-root", default="website/data")
    p.add_argument("--pairwise-judges", nargs="+", default=DEFAULT_PAIRWISE_JUDGES)
    p.add_argument("--reference-judges", nargs="+", default=DEFAULT_REFERENCE_JUDGES)
    p.add_argument("--target-models", nargs="+", default=DEFAULT_TARGET_MODELS)
    p.add_argument("--no-target-model-filter", action="store_true")
    p.add_argument("--part-id", type=int, default=None,
                   help="Grading schema part_id. Inferred from project name if omitted.")
    p.add_argument("--min-human-annotations", type=int, default=1)
    p.add_argument("--tie-tol", type=float, default=1e-9)
    p.add_argument("--details-csv", default=None)
    p.add_argument("--summary-csv", default=None)
    p.add_argument("--show-mismatches", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def infer_part_id(project: str, explicit: Optional[int]) -> int:
    if explicit is not None:
        return explicit
    pl = project.lower()
    if "complexity" in pl:
        return 4
    if "computation" in pl:
        return 5
    raise ValueError(
        "Could not infer part_id from project name. Please pass --part-id explicitly."
    )

def load_human_judge_data(
    data_root: Path,
    reference_judges: List[str],
    verbose: bool = False,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """judge_data[judge_id][problem_id] -> full JSON entry."""
    judge_data: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for judge in reference_judges:
        per_judge: Dict[str, Dict[str, Any]] = {}
        for _, raw in iter_json_files(data_root / judge, verbose=verbose,
                                      desc=f"Loading {judge}"):
            pid = raw.get("problem_id")
            if pid:
                per_judge[pid] = raw
        judge_data[judge] = per_judge
    return judge_data


def extract_part_score_from_attempt(attempt: Dict[str, Any], part_id: int) -> Optional[float]:
    grading = attempt.get("grading")
    if grading is None:
        return None
    details = grading.get("details", [])
    if not isinstance(details, list):
        return None
    for item in details:
        if item.get("part_id") == part_id and item.get("score") is not None:
            try:
                return float(item["score"])
            except Exception:
                return None
    return None


def find_human_score_for_pair(
    problem_entry: Dict[str, Any],
    model_a: str,
    model_b: str,
    part_id: int,
) -> Optional[float]:
    attempts = problem_entry.get("attempts", [])
    if not isinstance(attempts, list):
        return None

    for left, right in zip(attempts[:-1:2], attempts[1::2]):
        left_model, right_model = left.get("model_id"), right.get("model_id")
        if not left_model or not right_model:
            continue
        pair_score = extract_part_score_from_attempt(left, part_id)
        if pair_score is None:
            continue
        if left_model == model_a and right_model == model_b:
            return pair_score
        if left_model == model_b and right_model == model_a:
            return 1.0 - pair_score
    return None

def human_vote_label(score: float, tie_tol: float) -> str:
    if math.isclose(score, 0.5, abs_tol=tie_tol):
        return "tie"
    return "model_a" if score < 0.5 else "model_b"


def winner_from_human_mean(human_mean: float, tie_tol: float) -> str:
    if math.isclose(human_mean, 0.5, abs_tol=tie_tol):
        return "tie"
    return "model_a" if human_mean < 0.5 else "model_b"


def soft_correctness(predicted: str, human_mean: float, tie_tol: float) -> float:
    human_is_tie = math.isclose(human_mean, 0.5, abs_tol=tie_tol)
    if human_is_tie and predicted == "tie":
        return 1.0
    if human_is_tie:
        return 0.5
    if predicted == "tie":
        return 0.5
    if predicted == "model_a" and human_mean < 0.5:
        return 1.0
    if predicted == "model_b" and human_mean > 0.5:
        return 1.0
    return 0.0


def mean_or_none(values: List[float]) -> Optional[float]:
    return float(statistics.mean(values)) if values else None

def evaluate_records(
    pairwise_records: List[Dict[str, Any]],
    judge_data: Dict[str, Dict[str, Dict[str, Any]]],
    reference_judges: List[str],
    part_id: int,
    min_human_annotations: int,
    tie_tol: float,
) -> List[Dict[str, Any]]:
    evaluated: List[Dict[str, Any]] = []

    for rec in pairwise_records:
        annotations: List[float] = []
        for ref in reference_judges:
            entry = judge_data.get(ref, {}).get(rec["problem"])
            if entry is None:
                continue
            score = find_human_score_for_pair(entry, rec["model_a"], rec["model_b"], part_id)
            if score is not None:
                annotations.append(score)

        if len(annotations) < min_human_annotations:
            continue

        human_mean = float(statistics.mean(annotations))
        human_winner = winner_from_human_mean(human_mean, tie_tol)
        predicted = rec["winner"]

        vc_a = vc_b = vc_t = 0
        for s in annotations:
            lbl = human_vote_label(s, tie_tol)
            if lbl == "model_a":
                vc_a += 1
            elif lbl == "model_b":
                vc_b += 1
            else:
                vc_t += 1

        evaluated.append({
            "judge": rec["judge"],
            "project": rec["project"],
            "problem": rec["problem"],
            "model_a": rec["model_a"],
            "model_b": rec["model_b"],
            "predicted_winner": predicted,
            "majority_vote": rec["majority_vote"],
            "human_annotations": json.dumps(annotations),
            "n_human_annotations": len(annotations),
            "human_votes_model_a": vc_a,
            "human_votes_tie": vc_t,
            "human_votes_model_b": vc_b,
            "human_mean": human_mean,
            "human_baseline_winner": human_winner,
            "soft_score": soft_correctness(predicted, human_mean, tie_tol),
            "hard_correct": int(predicted == human_winner),
            "judgements_json": json.dumps(rec.get("judgements"), ensure_ascii=False, default=str),
            "outputs_json": json.dumps(rec.get("outputs"), ensure_ascii=False, default=str),
            "source_file": rec["source_file"],
        })
    return evaluated


def build_summary_rows(
    pairwise_records: List[Dict[str, Any]],
    evaluated_rows: List[Dict[str, Any]],
    requested_judges: List[str],
) -> List[Dict[str, Any]]:
    total_loaded = Counter(r["judge"] for r in pairwise_records)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in evaluated_rows:
        grouped[row["judge"]].append(row)

    all_judges = list(dict.fromkeys(
        requested_judges + sorted(set(total_loaded) - set(requested_judges))
    ))

    summary: List[Dict[str, Any]] = []
    for judge in all_judges:
        rows = grouped.get(judge, [])
        n_loaded = total_loaded.get(judge, 0)
        summary.append({
            "judge": judge,
            "loaded_pairs": n_loaded,
            "evaluated_pairs": len(rows),
            "coverage": (len(rows) / n_loaded) if n_loaded else None,
            "mean_soft_score": mean_or_none([r["soft_score"] for r in rows]),
            "hard_accuracy": mean_or_none([float(r["hard_correct"]) for r in rows]),
            "llm_tie_rate": mean_or_none(
                [1.0 if r["predicted_winner"] == "tie" else 0.0 for r in rows]
            ),
            "human_tie_rate": mean_or_none(
                [1.0 if r["human_baseline_winner"] == "tie" else 0.0 for r in rows]
            ),
            "avg_human_annotations": mean_or_none(
                [float(r["n_human_annotations"]) for r in rows]
            ),
        })
    return summary

def main() -> int:
    args = parse_args()

    try:
        part_id = infer_part_id(args.project, args.part_id)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    outputs_root = Path(args.outputs_root)
    data_root = Path(args.data_root)
    target = None if args.no_target_model_filter else set(args.target_models)

    print(f"Project: {args.project}")
    print(f"Part ID: {part_id}")
    print(f"Outputs root: {outputs_root}")
    print(f"Data root: {data_root}")
    print(f"Pairwise judges: {', '.join(args.pairwise_judges)}")
    print(f"Reference judges: {', '.join(args.reference_judges)}")
    print(f"Model filter: {'disabled' if target is None else f'{len(target)} models'}")
    print(f"Minimum human annotations: {args.min_human_annotations}")
    print()

    judge_data = load_human_judge_data(data_root, args.reference_judges, verbose=args.verbose)

    pairwise_records = load_pairwise_records(
        outputs_root=outputs_root,
        project=args.project,
        judges=args.pairwise_judges,
        target_models=target,
        verbose=args.verbose,
        record_builder=ground_truth_record_builder,
    )
    if not pairwise_records:
        print("No pairwise records found.")
        return 1

    evaluated = evaluate_records(
        pairwise_records=pairwise_records,
        judge_data=judge_data,
        reference_judges=args.reference_judges,
        part_id=part_id,
        min_human_annotations=args.min_human_annotations,
        tie_tol=args.tie_tol,
    )

    summary = build_summary_rows(pairwise_records, evaluated, args.pairwise_judges)

    print(f"Loaded pairwise rows: {len(pairwise_records)}")
    print(f"Evaluated rows with human baseline: {len(evaluated)}")
    print()

    print("## Summary")
    print_markdown_table(summary, headers=[
        "judge", "loaded_pairs", "evaluated_pairs", "coverage",
        "mean_soft_score", "hard_accuracy",
        "llm_tie_rate", "human_tie_rate", "avg_human_annotations",
    ])

    if args.show_mismatches > 0 and evaluated:
        worst = sorted(
            evaluated,
            key=lambda r: (r["soft_score"], r["judge"], r["problem"],
                           r["model_a"], r["model_b"]),
        )[: args.show_mismatches]
        print()
        print(f"## Worst {len(worst)} rows by soft score")
        print_markdown_table(worst, headers=[
            "judge", "problem", "model_a", "model_b",
            "predicted_winner", "human_baseline_winner",
            "human_mean", "soft_score", "hard_correct", "n_human_annotations",
        ])

    if args.details_csv:
        write_csv(Path(args.details_csv), evaluated, fieldnames=[
            "judge", "project", "problem", "model_a", "model_b",
            "predicted_winner", "majority_vote", "human_annotations",
            "n_human_annotations", "human_votes_model_a", "human_votes_tie",
            "human_votes_model_b", "human_mean", "human_baseline_winner",
            "soft_score", "hard_correct",
            "judgements_json", "outputs_json", "source_file",
        ])
        print()
        print(f"Saved detailed CSV to {args.details_csv}")

    if args.summary_csv:
        write_csv(Path(args.summary_csv), summary, fieldnames=[
            "judge", "loaded_pairs", "evaluated_pairs", "coverage",
            "mean_soft_score", "hard_accuracy",
            "llm_tie_rate", "human_tie_rate", "avg_human_annotations",
        ])
        print(f"Saved summary CSV to {args.summary_csv}")



if __name__ == "__main__":
    main()