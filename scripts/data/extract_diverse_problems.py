import json
import os
from pathlib import Path
from tqdm import tqdm
from proofrank.diversity import DiversityAnalysisParser
from proofrank.postprocess import fix_thinking
from collections import defaultdict
import argparse
from copy import deepcopy

def store_diverse_problems(args):
    base_path = Path(args.human_clustering_dir)

    print(f"Scanning directory: {base_path} ...")
    json_files = base_path.rglob("*.json")


    valid_problems = []
    possible_clusters = []

    parser = DiversityAnalysisParser()


    for file_path in tqdm(json_files):

        if 'part' in file_path.name or 'human' not in file_path.parts:
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            raw_json = json.load(f)

        if 'outputs' not in raw_json:
            continue

        sample_json = raw_json['outputs']
        sample_json = fix_thinking(sample_json)
        
        problem_id = raw_json['problem_id']

        success = parser.parse(sample_json)
        
        if not success:
            continue
        
        if len(parser.clusters) >= 3:
            valid_problems.append(problem_id)
            possible_clusters.append(parser.clusters)

    with open(args.raw_problems, 'r') as f:
        problems = json.load(f)
        problems = [p for p in problems if p['problem_id'].replace('/', '_') in valid_problems]
    with open(args.diversity_output, 'w') as f:
        json.dump(problems, f, indent=4)

def store_diverse_techniques(args):

    with open(args.original_data, "r") as f:
        human_problems = json.load(f)

    human_sols_lookup = {}
    for p in human_problems:
        p_id = p.get("problem_id")
        sol = p.get("solution")
        if not p_id or not sol:
            continue
        
        base_id = p_id.replace('/', '_')
        if "-part" in p_id:
            base_id = base_id.split("-part")[0]
            
        if base_id not in human_sols_lookup:
            human_sols_lookup[base_id] = set()
        human_sols_lookup[base_id].add(sol.strip())

    with open(args.all_data, "r") as f:
        all_problems = json.load(f)

    all_sols_ordered = {}
    for p in all_problems:
        p_id = p.get("problem_id")
        sol = p.get("solution")
        if not p_id or not sol:
            continue
            
        base_id = p_id.replace('/', '_')
        if "-part" in p_id:
            base_id = base_id.split("-part")[0]
            
        if base_id not in all_sols_ordered:
            all_sols_ordered[base_id] = []
        all_sols_ordered[base_id].append(sol.strip())

    base_path_all = Path(args.full_clustering_dir)
    print(f"Scanning directory: {base_path_all} ...")
    json_files_all = base_path_all.rglob("*.json")

    parser = DiversityAnalysisParser()
    files_with_warnings_all = []

    human_only_clusters = []
    pid_counter = defaultdict(int)
    for file_path in tqdm(list(json_files_all)):
        if 'part' in file_path.name or 'gpt-5' in file_path.parts:
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            raw_json = json.load(f)

        if 'outputs' not in raw_json:
            files_with_warnings_all.append((str(file_path), ["Critical Failure: Could not parse output"]))
            continue 

        sample_json = raw_json['outputs']
        sample_json = fix_thinking(sample_json)
        
        problem_id = raw_json.get('problem_id')
        if not problem_id:
            continue

        base_id = problem_id.replace('/', '_')
        if "-part" in problem_id:
            base_id = base_id.split("-part")[0]

        success = parser.parse(sample_json)
        if not success:
            files_with_warnings_all.append((str(file_path), ["Critical Failure: Parsing JSON structure"]))
            continue
        
        available_all_sols = all_sols_ordered.get(base_id, [])
        known_human_sols = human_sols_lookup.get(base_id, set())

        for cluster in parser.clusters:
            
            if isinstance(cluster, dict):
                indices = cluster.get('members', [])
                cluster_name = cluster.get('cluster_name', 'Unnamed Cluster')
            else:
                indices = getattr(cluster, 'members', [])
                cluster_name = getattr(cluster, 'cluster_name', 'Unnamed Cluster')
                
            if not indices:
                continue
                
            is_human_only = True
            for idx in indices:
                list_idx = idx - 1 
                
                if 0 <= list_idx < len(available_all_sols):
                    sol_string = available_all_sols[list_idx]
                    if sol_string not in known_human_sols:
                        is_human_only = False
                        break
                else:
                    is_human_only = False
                    break
                    
            if is_human_only:
                human_only_clusters.append({
                    "problem_id": base_id,
                    "cluster_name": cluster_name,
                    "idx": pid_counter[base_id]
                })
                pid_counter[base_id] += 1

    with open(args.raw_problems, 'r') as f:
        problems = json.load(f)
        problem_dict = {p['problem_id'].replace('/', '_'): p for p in problems}

    problems_with_technique = []
    for c in human_only_clusters:
        #Append an identifier to the problem ID matching the index of the technique
        problems_with_technique.append(deepcopy(problem_dict[c['problem_id'].replace('/', '_')]))
        problems_with_technique[-1]['technique'] = c['cluster_name']
        if c["idx"] > 0:
            problems_with_technique[-1]['problem_id'] = c['problem_id'] + f"-tech{c['idx']}"

    with open(args.adaptivity_output, 'w') as f:
        json.dump(problems_with_technique, f, indent=4)

    print(f"\nDiscovered {len(human_only_clusters)} unique human-only clusters across the dataset.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--human-clustering-dir",
        type=str,
        default="./outputs/summary_diversity_clustering/openai/oss-120b/human/human",
        help="Directory containing the JSON files to process.",
    )
    parser.add_argument(
        "--full-clustering-dir",
        type=str,
        default="./outputs/summary_diversity_clustering_all/openai/oss-120b/human/human",
        help="Directory containing the JSON files to process (containing clusters from all solutions).",
    )
    parser.add_argument(
        "--original-data",
        type=str,
        default="./data/postprocess/matharena_proofs/human_sols_ok.json",
        help="Path to the original data file containing problem and solution information.",
    )
    parser.add_argument(
        "--diversity-output",
        type=str,
        default="./data/raw/diversity_samples/sample.json",
        help="Path to save the extracted diverse problems.",
    )
    parser.add_argument(
        "--diversity-threshold",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--raw-problems",
        type=str,
        default="./data/raw/matharena_proofs/sample.json",
        help="Path to the raw problems file to filter and save.",
    )
    parser.add_argument(
        "--all-data",
        type=str,
        default="./data/postprocess/matharena_proofs/all_sols_ok.json",
        help="Path to the data file containing all problems and solutions.",
    )
    parser.add_argument(
        "--adaptivity-output",
        type=str,
        default="./data/raw/technique_adaptivity/initial_sample.json",
        help="Path to save the problems with identified human-only techniques.",
    )
    args = parser.parse_args()
    
    store_diverse_problems(args)
    store_diverse_techniques(args)

if __name__ == "__main__":
    main()