import os
import json
import itertools
import numpy as np
import pandas as pd
from collections import defaultdict

DATA_ROOT = "website/data"
JUDGES = ["proofrank-1", "proofrank-2", "proofrank-3"]

def load_judge_data(judge_id):
    judge_path = os.path.join(DATA_ROOT, judge_id)
    if not os.path.exists(judge_path):
        print(f"Warning: Directory for {judge_id} not found at {judge_path}")
        return {}

    data_map = {}
    for root, dirs, files in os.walk(judge_path):
        for file in files:
            if file.endswith(".json"):
                full_path = os.path.join(root, file)
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        entry = json.load(f)
                        pid = entry.get('problem_id')
                        if pid:
                            data_map[pid] = entry
                except Exception as e:
                    print(f"Error reading {full_path}: {e}")
    return data_map

def normalize_part_id(pid):
    try:
        return int(pid)
    except (ValueError, TypeError):
        return pid

def get_score_map(problem_entry):
    scores = {}
    attempts = problem_entry.get('attempts', [])

    for attempt in attempts:
        model_id = attempt.get('model_id')
        if not model_id:
            continue

        grading = attempt.get('grading', {})
        if grading is None:
            continue

        details = grading.get('details', [])

        part_scores = {}
        for item in details:
            pid = normalize_part_id(item.get('part_id'))
            score = item.get('score')

            if pid is not None and score is not None:
                try:
                    part_scores[pid] = float(score)
                except (ValueError, TypeError):
                    continue

        scores[model_id] = part_scores

    return scores

def calculate_agreement(val1, val2):
    try:
        v1 = float(val1)
        v2 = float(val2)

        if abs(v1 - v2) > 1.0:
            print(
                f"Warning: Unexpected score difference {abs(v1 - v2)} "
                f"between values {v1} and {v2}. Check data integrity."
            )
            breakpoint()

        return 1.0 - abs(v1 - v2)

    except (ValueError, TypeError):
        return np.nan

def print_markdown_table(df):
    print(df.to_markdown(floatfmt=".3f"))

def build_metric_score_dataframe(judge_data, metrics_map):
    rows = []

    for judge_id, problem_map in judge_data.items():
        for problem_id, problem_entry in problem_map.items():
            score_map = get_score_map(problem_entry)

            for model_id, metric_scores in score_map.items():
                row = {
                    "judge": judge_id,
                    "problem_id": problem_id,
                    "model_id": model_id,
                }

                for metric_id, metric_name in metrics_map.items():
                    row[metric_name] = metric_scores.get(metric_id, np.nan)

                rows.append(row)

    return pd.DataFrame(rows)

def main():
    judge_data = {j: load_judge_data(j) for j in JUDGES}

    metrics_map = {
        1: "Verbosity",
        2: "Elegance",
        3: "Insight",
        4: "Complexity",
        5: "Camp. Challenge"
    }

    pairs = list(itertools.combinations(JUDGES, 2))
    raw_results = []
    n_pairs = defaultdict(int)

    print(f"Processing {len(pairs)} pairings...")

    for judge_a, judge_b in pairs:
        data_a_map = judge_data[judge_a]
        data_b_map = judge_data[judge_b]

        common_problems = set(data_a_map.keys()) & set(data_b_map.keys())

        metric_agreements = {mid: [] for mid in metrics_map.keys()}
        n_pairs[(judge_a, judge_b)] = len(common_problems)

        for pid in common_problems:
            scores_a = get_score_map(data_a_map[pid])
            scores_b = get_score_map(data_b_map[pid])

            if not scores_a or not scores_b:
                continue

            print(judge_a, judge_b, pid)

            common_models = set(scores_a.keys()) & set(scores_b.keys())

            for mid in common_models:
                print(mid)

                attempt_grades_a = scores_a[mid]
                attempt_grades_b = scores_b[mid]

                for metric_id in metrics_map.keys():
                    if metric_id in attempt_grades_a and metric_id in attempt_grades_b:
                        val_a = attempt_grades_a[metric_id]
                        val_b = attempt_grades_b[metric_id]

                        agreement = calculate_agreement(val_a, val_b)

                        if not np.isnan(agreement):
                            metric_agreements[metric_id].append(agreement)

        for mid, metric_name in metrics_map.items():
            agreements = metric_agreements[mid]

            if agreements:
                avg = np.mean(agreements)
            else:
                avg = np.nan

            raw_results.append({
                "judge_a": judge_a,
                "judge_b": judge_b,
                "Pair": f"{judge_a} vs {judge_b}",
                "Metric": metric_name,
                "Agreement": avg
            })

    if not raw_results:
        print("No overlapping data found.")
        return

    df = pd.DataFrame(raw_results)

    total_pairs = sum(n_pairs.values())

    overall = df.groupby("Metric").apply(
        lambda x: np.sum(
            x["Agreement"] * [
                n_pairs[(a, b)]
                for a, b in zip(x["judge_a"], x["judge_b"])
            ]
        ) / total_pairs
    )

    df = pd.concat([
        df,
        pd.DataFrame({
            "judge_a": "overall",
            "judge_b": "overall",
            "Pair": "overall",
            "Metric": overall.index,
            "Agreement": overall.values
        })
    ], ignore_index=True)

    pivot_df = df.pivot(index="Pair", columns="Metric", values="Agreement")

    pivot_df["Overall Agreement"] = pivot_df.mean(axis=1)

    pivot_df = pivot_df.sort_values(by="Overall Agreement", ascending=False)

    print("\n### Final Agreement Ranking Table\n")
    print_markdown_table(pivot_df)

    metric_score_df = build_metric_score_dataframe(judge_data, metrics_map)

    metric_columns = list(metrics_map.values())

    if metric_score_df.empty:
        print("\nNo metric score data available for correlation.")
        return

    pearson_corr = metric_score_df[metric_columns].corr(method="pearson")

    spearman_corr = metric_score_df[metric_columns].corr(method="spearman")

    print("\n### Overall Pearson Correlation Between Metrics\n")
    print_markdown_table(pearson_corr)

    print("\n### Overall Spearman Correlation Between Metrics\n")
    print_markdown_table(spearman_corr)

if __name__ == "__main__":
    main()