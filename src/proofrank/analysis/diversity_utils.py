import re
import os
from loguru import logger
import json
import numpy as np
from collections import defaultdict

PART_SUFFIX_RE = re.compile(r"^(.*)-part(\d+)$")
HUMAN_MODEL_ID = "human/human"


def is_human_model(model_id):
    return model_id == HUMAN_MODEL_ID or str(model_id).startswith("human/")


def get_base_problem_id(problem_id):
    problem_id = str(problem_id)
    match = PART_SUFFIX_RE.match(problem_id)
    if match:
        return match.group(1)
    return problem_id


def get_part_number(problem_id):
    problem_id = str(problem_id)
    match = PART_SUFFIX_RE.match(problem_id)
    if match:
        return int(match.group(2))
    return 0

def load_diversity_solution_order(diversity_samples_path):
    if not os.path.exists(diversity_samples_path):
        logger.warning(
            f"Diversity samples path does not exist: {diversity_samples_path}"
        )
        return {}

    with open(diversity_samples_path, "r", encoding="utf-8") as f:
        raw_samples = json.load(f)

    order_by_base_pid = defaultdict(list)

    for sample in raw_samples:
        problem_id = sample.get("problem_id")
        if not problem_id:
            continue

        model_id = sample.get("model_id", HUMAN_MODEL_ID)
        base_pid = get_base_problem_id(problem_id)

        order_by_base_pid[base_pid].append(
            {
                "idx": None,
                "model_id": model_id,
                "problem_id": problem_id,
                "base_problem_id": base_pid,
                "part_number": get_part_number(problem_id),
            }
        )

    for base_pid, entries in order_by_base_pid.items():
        for i, entry in enumerate(entries):
            entry["idx"] = i + 1

    logger.info(
        f"Loaded diversity solution order for {len(order_by_base_pid)} base problems "
        f"from {diversity_samples_path}"
    )

    return dict(order_by_base_pid)

def normalize_compiled_solution_entry(entry, one_based_idx):
    if isinstance(entry, dict):
        model_id = entry.get("model_id", HUMAN_MODEL_ID)
        problem_id = entry.get("problem_id")
    else:
        model_id = entry[0] if len(entry) > 0 else HUMAN_MODEL_ID
        problem_id = entry[1] if len(entry) > 1 else None

    return {
        "idx": one_based_idx,
        "model_id": model_id,
        "problem_id": problem_id,
        "base_problem_id": get_base_problem_id(problem_id),
        "part_number": get_part_number(problem_id),
    }


def get_compiled_solution_entries(row, diversity_solution_order=None):
    row_pid = row.get("problem_id")
    base_pid = get_base_problem_id(row_pid)

    meta = row.get("extra_metadata", {})

    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}

    solution_map = meta.get("compiled_solutions_ids", [])

    if solution_map:
        return [
            normalize_compiled_solution_entry(entry, i + 1)
            for i, entry in enumerate(solution_map)
        ]

    if diversity_solution_order and base_pid in diversity_solution_order:
        return [dict(entry) for entry in diversity_solution_order[base_pid]]

    logger.warning(
        f"No compiled_solutions_ids and no diversity sample order found for {base_pid}"
    )

    return []

def get_main_experiment_entries(row, diversity_solution_order):
    row_pid = row.get("problem_id")
    base_pid = get_base_problem_id(row_pid)
    model_id = row.get("solver_id")

    if not model_id:
        logger.warning(f"Missing solver_id for main diversity row: {row_pid}")
        return []

    all_entries_for_problem = diversity_solution_order.get(base_pid, [])

    model_entries = [
        dict(entry)
        for entry in all_entries_for_problem
        if entry["model_id"] == model_id
    ]

    model_entries = sorted(
        model_entries,
        key=lambda entry: entry["part_number"],
    )

    model_entries = model_entries[:4]

    # Reindex locally because the main prompt only contains this model's samples.
    for local_idx, entry in enumerate(model_entries, start=1):
        entry["idx"] = local_idx

    return model_entries

def normalize_clustering_output(outputs):
    if isinstance(outputs, list):
        return outputs[0] if outputs else ""
    return outputs


def parse_diversity_clusters(parser, outputs):
    output_text = normalize_clustering_output(outputs)
    success = parser.parse(output_text)

    if not success:
        return None

    clusters = []

    for cluster in parser.clusters:
        cluster_name = cluster["cluster_name"]

        members = []
        for member in cluster["members"]:
            try:
                members.append(int(member))
            except Exception:
                continue

        clusters.append(
            {
                "name": cluster_name,
                "members": members,
            }
        )

    return clusters


def cluster_counts_for_indices(clusters, allowed_indices):

    allowed_indices = set(allowed_indices)
    counts = defaultdict(int)

    for cluster in clusters:
        cluster_name = cluster["name"]
        matched_members = [
            member for member in cluster["members"]
            if member in allowed_indices
        ]

        if matched_members:
            counts[cluster_name] += len(matched_members)

    return dict(counts)


def cluster_names_for_indices(clusters, allowed_indices):
    return set(cluster_counts_for_indices(clusters, allowed_indices).keys())


def entropy_from_counts(counts):
    total = sum(counts.values())

    if total <= 1:
        return 0.0

    probs = [count / total for count in counts.values() if count > 0]
    return float(-sum(p * np.log2(p) for p in probs))


def cross_entropy_from_counts(model_counts, global_counts, eps=1e-12):
    model_total = sum(model_counts.values())
    global_total = sum(global_counts.values())

    if model_total == 0 or global_total == 0:
        return None

    score = 0.0

    for cluster_name, model_count in model_counts.items():
        p_model = model_count / model_total
        p_global = global_counts.get(cluster_name, 0) / global_total
        p_global = max(p_global, eps)
        score += -p_model * np.log2(p_global)

    return float(score)


def group_entries_by_model(entries, exclude_human=True):
    grouped = defaultdict(list)

    for entry in entries:
        model_id = entry["model_id"]

        if exclude_human and is_human_model(model_id):
            continue

        grouped[model_id].append(entry)

    for model_id in grouped:
        grouped[model_id] = sorted(
            grouped[model_id],
            key=lambda x: x["part_number"],
        )

    return grouped


def solution_is_correct(entry, correctness_map):
    return correctness_map.get((entry["model_id"], entry["problem_id"]), False)


def add_diversity_metric(
    raw_scores,
    metric_name,
    model_id,
    problem_id,
    value,
    topic_map,
    unique_topics,
    correctness_bucket=False,
    answer_correctness_bucket=False
):
    base_pid = get_base_problem_id(problem_id)
    topic = topic_map.get(base_pid) or topic_map.get(problem_id)

    if correctness_bucket:
        raw_scores["correct"][metric_name][model_id][base_pid] = value*100

        if topic and topic in unique_topics:
            raw_scores[f"{topic}_correct"][metric_name][model_id][base_pid] = value*100
    elif answer_correctness_bucket:
        raw_scores["answer_correct"][metric_name][model_id][base_pid] = value*100
    else:
        raw_scores["all"][metric_name][model_id][base_pid] = value*100

        if topic and topic in unique_topics:
            raw_scores[topic][metric_name][model_id][base_pid] = value*100