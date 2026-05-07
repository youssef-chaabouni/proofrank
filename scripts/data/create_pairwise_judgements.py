import pandas as pd
import argparse
import random
import json

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


def deterministic_random_order(model1, model2, problem_id):
    random.seed(f"{model1}_{model2}_{problem_id}")
    return random.random() < 0.5


def main():
    parser = argparse.ArgumentParser(description="Postprocess the grading results.")
    parser.add_argument("--project", type=str, required=True, help="Project name")
    parser.add_argument(
        "--target-models",
        type=str,
        nargs="*",
        default=DEFAULT_TARGET_MODELS,
        help="Models to include in pairwise comparisons",
    )
    args = parser.parse_args()

    all_solutions = pd.read_json(f"./data/postprocess/{args.project}/test_samples.json")
    all_solutions = all_solutions[
        all_solutions.solution.apply(lambda x: len(x.strip()) > 0)
    ]
    all_solutions["problem_id"] = all_solutions.problem_id.apply(
        lambda x: x.replace("/", "_").split("-part")[0]
    )
    sample_solutions = (
        all_solutions.groupby(["model_id", "problem_id"]).first().reset_index()
    )
    all_pairs = []

    sample_solutions.groupby("problem_id").apply(
        lambda x: all_pairs.extend(
            [
                (
                    {
                        "problem": row1["problem"],
                        "solution_1": row1["solution"],
                        "solution_2": row2["solution"],
                        "model_id": row1["model_id"],
                        "problem_id": f"{row1['problem_id']}-part{row2['model_id'].replace('/','_')}",
                        "solutions": row1["solutions"],
                    }
                    if deterministic_random_order(
                        row1["model_id"], row2["model_id"], row1["problem_id"]
                    )
                    else {
                        "problem": row1["problem"],
                        "solution_1": row2["solution"],
                        "solution_2": row1["solution"],
                        "model_id": row2["model_id"],
                        "problem_id": f"{row2['problem_id']}-part{row1['model_id'].replace('/','_')}",
                        "solutions": row1["solutions"],
                    }
                )
                for i, row1 in x.iterrows()
                for j, row2 in x.iterrows()
                if i < j
                and row1["model_id"] in args.target_models
                and row2["model_id"] in args.target_models
            ]
        )
    )

    print(f"Created {len(all_pairs)} pairwise comparisons.")

    with open(f"data/postprocess/{args.project}/pairwise_solutions.json", "w") as f:
        json.dump(all_pairs, f, indent=4)


if __name__ == "__main__":
    main()
