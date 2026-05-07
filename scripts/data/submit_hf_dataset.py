from datasets import Dataset, DatasetDict, Features, Value, Sequence
import json
import os

def main():
    with open("data/raw/matharena_proofs/sample.json", "r", encoding="utf-8") as f:
        main_data = json.load(f)
    with open("data/postprocess/matharena_proofs/human_sols_ok.json", "r", encoding="utf-8") as f:
        human_data = json.load(f)
    with open("data/raw/diversity_samples/sample.json", "r", encoding="utf-8") as f:
        diversity_data = json.load(f)
    with open("data/raw/technique_adaptivity/sample.json", "r", encoding="utf-8") as f:
        technique_data = json.load(f)

    diversity_data_ids = set(entry["problem_id"] for entry in diversity_data)


    main_split = []
    for entry in main_data:
        main_split.append(
            {
                "problem_id": entry["problem_id"],
                "problem": entry["problem"],
                "gold_answer": entry['gold_answer'],
                "in_diversity_eval": entry["problem_id"] in diversity_data_ids,
                "human_solution_summaries": [e['solution_summary'] for e in human_data if e["problem_id"].split("-part")[0] == entry["problem_id"]],
                "technique": None,
            }
        )

    adaptivity_split = []
    for entry in technique_data:
        adaptivity_split.append(
            {
                "problem_id": entry["problem_id"],
                "problem": entry["problem"],
                "gold_answer": entry['gold_answer'],
                "in_diversity_eval": False,
                "human_solution_summaries": [],
                "technique": entry["technique"],
            }
        )

    target_features = Features({
        "problem_id": Value("string"),
        "problem": Value("string"),
        "gold_answer": Value("string"),
        "in_diversity_eval": Value("bool"),
        "human_solution_summaries": Sequence(Value("string")),
        "technique": Value("string"),
    })

    DatasetDict({
        "main": Dataset.from_list(main_split, features=target_features),
        "adaptivity": Dataset.from_list(adaptivity_split, features=target_features),
    }).push_to_hub("Anon987281293/ProofRank", private=False, token=os.environ.get("HF_TOKEN_NEURIPS"))


if __name__ == "__main__":
    main()
