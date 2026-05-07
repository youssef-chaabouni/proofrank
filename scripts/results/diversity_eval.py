import os
import json
import itertools
import argparse
from typing import Dict, List
from collections import defaultdict
from llmteach.diversity import DiversityAnalysisParser
from llmteach.postprocess import fix_thinking


def load_grouped_data(filepath: str) -> Dict[str, List[str]]:
    """
    Reads dataset and extracts solutions, grouping them by their base problem ID.
    Ignores the `-partX` suffix so that solutions for the same task sit securely in a single list.
    Preserves document order to properly map back to the 'Proof 1', 'Proof 2' labeling.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    grouped = defaultdict(list)
    for item in data:
        base_id = item["problem_id"].split("-part")[0]
        grouped[base_id].append(item["solution"].strip())

    return grouped


def get_text_to_cluster_map(
    output_dir: str, problem_id: str, ordered_texts: List[str]
) -> Dict[str, str]:
    """
    Reads the cluster output for a specific problem_id and maps the literal
    solution text to its assigned cluster_id.
    """
    filepath = os.path.join(output_dir, f"{problem_id}.json")
    if not os.path.exists(filepath):
        return None, None

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    try:
        wrapper = json.loads(content)
        if "outputs" in wrapper:
            content = wrapper["outputs"]
        elif "choices" in wrapper:
            content = wrapper["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = content[0]
        content = fix_thinking(content)
    except json.JSONDecodeError:
        pass
    parser = DiversityAnalysisParser()
    success = parser.parse(content)

    if not success or not parser.clusters:
        return None, None

    text_to_cluster = {}
    cluster_names = {}
    for cluster in parser.clusters:
        c_id = cluster.get("cluster_id", "UNKNOWN_C_ID")
        cluster_names[c_id] = cluster.get("cluster_name", "Unnamed Cluster")
        for member in cluster.get("members", []):
            try:
                idx = int(member) - 1
                if 0 <= idx < len(ordered_texts):
                    text = ordered_texts[idx]
                    text_to_cluster[text] = c_id
            except (ValueError, TypeError):
                continue

    return text_to_cluster, cluster_names


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate diversity clustering alignment accuracy via subset pairwise matching."
    )
    parser.add_argument(
        "--full_data",
        type=str,
        default="./data/postprocess/matharena_proofs/human_sols_ok.json",
    )
    parser.add_argument(
        "--sample_data",
        type=str,
        default="./data/postprocess/matharena_proofs/human_sols_sample.json",
    )
    parser.add_argument(
        "--full_out",
        type=str,
        default="./outputs/summary_diversity_clustering_ideas/openai/oss-120b/human/human",
    )
    parser.add_argument(
        "--sample_out",
        type=str,
        default="./outputs/sample_summary_diversity_clustering_ideas/openai/oss-120b/human/human",
    )
    args = parser.parse_args()

    print("Loading datasets...")
    grouped_full = load_grouped_data(args.full_data)
    grouped_sample = load_grouped_data(args.sample_data)
    breakpoint()
    total_pairs = 0
    correct_pairs = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    num_pairs = defaultdict(int)
    missing_data_skipped = 0

    for subset_id, subset_texts in grouped_sample.items():
        base_id = subset_id.split("_subset")[0]

        if len(subset_texts) < 2:
            continue

        full_clusters, full_cluster_names = get_text_to_cluster_map(
            args.full_out, base_id, grouped_full.get(base_id, [])
        )
        sample_clusters, sample_cluster_names = get_text_to_cluster_map(
            args.sample_out, subset_id, subset_texts
        )

        if full_clusters is None or sample_clusters is None:
            missing_data_skipped += 1

            continue

        for textA, textB in itertools.combinations(subset_texts, 2):
            total_pairs += 1
            c_full_A = full_clusters.get(textA, id(textA))
            c_full_B = full_clusters.get(textB, id(textB))
            full_match = c_full_A == c_full_B

            c_sample_A = sample_clusters.get(textA, id(textA))
            c_sample_B = sample_clusters.get(textB, id(textB))
            sample_match = c_sample_A == c_sample_B
            num_pairs[base_id] += 1
            if full_match == sample_match:
                correct_pairs[base_id] += 1
            if sample_match and not full_match:
                fp[base_id] += 1
            if not sample_match and full_match:
                fn[base_id] += 1
            
            # else:
            #     print(f"\n[Mismatch] Subset: {subset_id}")
            #     print(
            #         f"Text A: {textA}... \n| Cluster Full: {full_cluster_names.get(c_full_A, c_full_A)} \n| Cluster Sample: {sample_cluster_names.get(c_sample_A, c_sample_A)}"
            #     )
            #     print(
            #         f"Text B: {textB}... \n| Cluster Full: {full_cluster_names.get(c_full_B, c_full_B)} \n| Cluster Sample: {sample_cluster_names.get(c_sample_B, c_sample_B)}"
            #     )
            #     breakpoint()

    print("=" * 45)
    print("Diversity Clustering Accuracy Evaluation")
    print("=" * 45)
    accuracies = {base_id: correct_pairs[base_id] / num_pairs[base_id]
        for base_id in num_pairs if num_pairs[base_id] > 0}

    if total_pairs > 0:
        accuracy = sum(correct_pairs.values()) / total_pairs
        print(f"Subsets Processed    : {len(grouped_sample) - missing_data_skipped}")
        print(f"Total Pairs Evaluated: {total_pairs}")
        print(f"Agreed Cluster Pairs : {sum(correct_pairs.values())}")
        print(f"Clustering Accuracy (meaned by problems): {sum(accuracies.values()) / len(accuracies):.4f} ( {sum(accuracies.values()) / len(accuracies) * 100:.2f}% )")
        #Add fn and fp rates
        print(f"False Positives      : {sum(fp.values())} ( {sum(fp.values()) / total_pairs * 100:.2f}% )")
        print(f"False Negatives      : {sum(fn.values())} ( {sum(fn.values()) / total_pairs * 100:.2f}% )")
        print(f"Clustering Accuracy  : {accuracy:.4f} ( {accuracy * 100:.2f}% )")
    else:
        print(
            "No paired matches could be evaluated. Double-check file paths and JSON formatting."
        )

    if missing_data_skipped > 0:
        print(
            f"\n[Warning] {missing_data_skipped} subset(s) were skipped due to missing or unparseable cluster output configs."
        )


if __name__ == "__main__":
    main()
