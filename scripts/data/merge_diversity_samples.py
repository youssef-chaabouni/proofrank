import argparse
import json
from collections import defaultdict

TARGET_MODELS = {
    'deepseek/deepseek_v32_think',
    'gemini/gemini-3-flash',
    'gemini/gemini-31-pro',
    'openai/gpt-54',
    'stepfun/3.5-flash',
    'glm/glm-5',
    'xai/grok-41-fast-reasoning',
    'moonshot/k25',
    'qwen/qwen35_397b_a17b_high',
    "openai/oss-120b",
}

def reindex_multiple_solutions(samples):
    id_counter = defaultdict(lambda: defaultdict(int))
    for sample in samples:
        sample["original_problem_id"] = sample["problem_id"]
        if id_counter[sample["model_id"]][sample["original_problem_id"]] > 0:
            sample[
                "problem_id"
            ] += f"-part{id_counter[sample['model_id']][sample['original_problem_id']]}"
        id_counter[sample["model_id"]][sample["original_problem_id"]] += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_file", type=str, default="data/postprocess/matharena_proofs/test_samples.json")
    parser.add_argument("--supplement_file", type=str, default="data/postprocess/diversity_samples/test_samples.json")
    parser.add_argument("--human_file", type=str, default="data/postprocess/matharena_proofs/human_sols_ok.json")
    parser.add_argument("--output_file", type=str, default="data/postprocess/matharena_proofs/diversity_samples.json")

    args = parser.parse_args()

    with open(args.main_file, 'r') as f:
        main_data = json.load(f)
    with open(args.supplement_file, 'r') as f:
        supplement_data = json.load(f)
    with open(args.human_file, 'r') as f:
        human_data = json.load(f)
    
    model_counts =  defaultdict(int)

    relevant_problems = set([p['problem_id'].split('-part')[0] for p in supplement_data])
    filtered_main_data = [p for p in main_data if p['problem_id'].split('-part')[0] in relevant_problems]
    filtered_main_data = [p for p in filtered_main_data if p['model_id'] in TARGET_MODELS]
    filtered_human_data = [p for p in human_data if p['problem_id'].split('-part')[0] in relevant_problems]
    supplement_data = [p for p in supplement_data if p['model_id'] in TARGET_MODELS]
    for entry in supplement_data:
        entry['problem_id'] = entry['problem_id'].split('-part')[0]
        model_counts[entry['model_id']] += 1
    for entry in filtered_main_data:
        entry['problem_id'] = entry['problem_id'].split('-part')[0]
        model_counts[entry['model_id']] += 1
    for entry in filtered_human_data:
        entry['problem_id'] = entry['problem_id'].split('-part')[0]
        model_counts[entry['model_id']] += 1
    combined_data = filtered_main_data + supplement_data + filtered_human_data
    reindex_multiple_solutions(combined_data)

    print(f"Writng {len(combined_data)} samples to {args.output_file}...")

    for model in model_counts:
        print(f"Model {model} has {model_counts[model]} samples.") 

    with open(args.output_file, 'w') as f:
        json.dump(combined_data, f, indent=4)

if __name__ == "__main__":
    main()

