from .postprocess import fix_thinking
import json
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def process_problem(problem_path, judge_config, solver_config, solver_name=None):
    with open(problem_path, "r") as f:
        try:
            problem_data = json.load(f)
        except json.JSONDecodeError:
            breakpoint()

    solver_id = "/".join(problem_path.split("/")[-3:-1])

    return {
        "judge": judge_config["human_readable_id"],
        "solver": (
            solver_config["human_readable_id"] if solver_name is None else solver_name
        ),
        "solver_id": solver_id,
        "problem": problem_data["problem"],
        "problem_id": problem_data["problem_id"],
        "original_problem_id": problem_data["problem_id"].split("-")[0],
        "solution": fix_thinking(problem_data.get("solution", "")),
        "outputs": (
            fix_thinking(problem_data["outputs"])
            if isinstance(problem_data["outputs"], str)
            else [fix_thinking(p) for p in problem_data["outputs"]]
        ),
        "competition": problem_data["problem_id"].split("_")[0],
        "true_grade": problem_data.get("majority_vote", None),
        "cost": problem_data.get("solution_cost", 0),
        "extra_metadata": problem_data,
    }


def parse_outputs(output_folder, configs_folder, setting_name, target_models=[], judge_name="openai/oss-120b"):
    output_dir = os.path.join(output_folder, setting_name.replace(".yaml", ""))
    results = []

    output_path = Path(output_dir)
    config_path = Path(configs_folder)

    files = list(output_path.glob("*/*/*/*/*.json"))

    for problem_path in tqdm(files, desc=f"Loading {setting_name}"):
        parts = problem_path.parts
        if len(parts) < 5:
            continue

        judge_model = parts[-4]

        if judge_model != judge_name:
            continue

        solver_api = parts[-3]
        solver_model = parts[-2]

        current_solver_id = f"{solver_api}/{solver_model}"

        judge_config = {"human_readable_id": judge_model}
        solver_config = {"human_readable_id": solver_model}
        problem_data = process_problem(
            str(problem_path), judge_config, solver_config, solver_name=solver_model
        )
        

        problem_data["is_valid"] = current_solver_id in target_models

        solution = problem_data["solution"]
        problem_data["is_valid"] = problem_data["is_valid"] and (
            isinstance(solution, list)
            and len(solution[-1]["content"]) > 2
            or isinstance(solution, dict)
            and len(solution["content"]) > 2
            or isinstance(solution, str)
            and len(solution) > 2
        )
        results.append(problem_data)
    return pd.DataFrame(results)
