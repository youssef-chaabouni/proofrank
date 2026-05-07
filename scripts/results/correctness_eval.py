import argparse
import itertools
from pathlib import Path
from llmteach.result_utils import parse_outputs
from llmteach.parser import find_last_boxed_content

TARGET_MODELS = {
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
}


def correctness_word(setting):
    if setting == "completeness_checker":
        return "incomplete"
    else:
        return "incorrect"


def correctness_eval(outputs, setting):
    if isinstance(outputs, str):
        return [correctness_word(setting) not in outputs.lower()]
    elif isinstance(outputs, list):
        return [correctness_word(setting) not in output.lower() for output in outputs]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate correctness of correctness scheme."
    )
    parser.add_argument(
        "--components",
        nargs="+",
        help="List of components to the correctness scheme.",
        default=["answer_checker", "completeness_checker"],
    )
    parser.add_argument(
        "--gt_folder",
        type=str,
        help="Folder containing the ground truth outputs.",
        default="correctness_checker",
    )

    args = parser.parse_args()

    gt_outputs = parse_outputs(
        "outputs", "configs", args.gt_folder, target_models=TARGET_MODELS
    )
    is_correct = {
        (row["solver"], row["problem_id"]): sum(
            correctness_eval(row["outputs"], "correctness_checker")
        )
        > 0.5
        for i, row in gt_outputs.iterrows()
        if row["judge"] == "gpt-54"
    }

    predictions = {}
    for component in args.components:
        component_outputs = parse_outputs(
            "outputs", "configs", component, target_models=TARGET_MODELS
        )
        for i, row in component_outputs.iterrows():
            key = (row["solver"], row["problem_id"])
            if key not in predictions:
                predictions[key] = {}
            predictions[key][component] = (
                sum(correctness_eval(row["outputs"], component)) > 0.5
            )

    # Evaluate the accuracy of each component and each combination of components
    all_combinations = []
    for r in range(1, len(args.components) + 1):
        all_combinations.extend(itertools.combinations(args.components, r))

    # Evaluate the accuracy of each component and each combination of components
    results = {}
    for combo in all_combinations:
        combo_name = " + ".join(combo)
        correct_matches = 0
        total_evaluated = 0

        for key, gt_val in is_correct.items():
            if key not in predictions:
                continue

            # Ensure all required components in this combination were successfully parsed
            if not all(comp in predictions[key] for comp in combo):
                continue

            # For a combination, we assume a logical AND:
            # Usually, an output is fully correct if it is *both* correct AND complete.
            combo_prediction = all(predictions[key][comp] for comp in combo)

            if combo_prediction == gt_val:
                correct_matches += 1
            elif len(combo) == 2:
                breakpoint()
            total_evaluated += 1

        if total_evaluated > 0:
            accuracy = correct_matches / total_evaluated
        else:
            accuracy = 0.0

        results[combo_name] = {
            "accuracy": accuracy,
            "correct": correct_matches,
            "total": total_evaluated,
        }

    # Output the results in a formatted table
    print(f"{'Component Combination':<50} | {'Accuracy':<10} | {'Matched/Total'}")
    print("-" * 80)
    for combo_name, metrics in sorted(
        results.items(), key=lambda x: x[1]["accuracy"], reverse=True
    ):
        acc = metrics["accuracy"]
        correct = metrics["correct"]
        total = metrics["total"]
        print(f"{combo_name:<50} | {acc:<10.4f} | {correct}/{total}")


if __name__ == "__main__":
    main()
