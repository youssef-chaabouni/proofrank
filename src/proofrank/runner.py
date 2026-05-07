import os
import json
import random
import re
import glob

from copy import deepcopy
from .api import APIQuery
from .executor import QueryExecutor
from .postprocess import fix_thinking
from .results_processors import ResultsProcessor, DefaultProcessor
from loguru import logger


random.seed(0)


def run(
    model_config,
    setting_config,
    config_path,
    skip_existing=False,
    output_folder="outputs",
    reparse=False,
    results_preprocessor: ResultsProcessor = DefaultProcessor,
):
    model = model_config["model"]
    n = model_config["n"]
    api = model_config["api"]
    n_retries = setting_config.get("n_retries", 10)

    max_tokens = model_config.get("max_tokens", setting_config["default_max_tokens"])
    temperature = model_config.get("temperature", setting_config["default_temperature"])
    kwargs = model_config.copy()
    del kwargs["model"]
    del kwargs["n"]
    del kwargs["api"]
    del kwargs["human_readable_id"]
    if "date" in kwargs:
        del kwargs["date"]
    kwargs["max_tokens"] = max_tokens
    kwargs["temperature"] = temperature

    logger.info(f"New run, model: {model}, competition: {setting_config['name']}")

    multi_input_keys = setting_config.get("multi_input_placeholders", [])
    include_human = setting_config.get("include_human", False)
    compile_all = setting_config.get("compile_all_parts", False)

    key_labels = {
        k: " ".join(word.capitalize() for word in k.split("_"))
        for k in multi_input_keys
    }

    if setting_config.get("data_path").endswith(".json"):
        with open(setting_config["data_path"], "r") as f:
            raw_problems = json.load(f)

        problems = []

        if not multi_input_keys:
            problems = raw_problems
        elif compile_all:
            unique_pids = set(
                p.get("problem_id").split("-part")[0]
                for p in raw_problems 
                if p.get("problem_id")
            )
            for pid in unique_pids:
                problem = None
                solution_ids = []
                aggregations = {}
                for key in multi_input_keys:
                    aggregations[key] = []

                for p in raw_problems:
                    if p.get("problem_id").split("-part")[0] == pid:
                        if problem is None:
                            problem = deepcopy(p)
                        solution_ids.append(
                            (p.get("model_id", "human/human"), p["problem_id"])
                        )
                        for key in multi_input_keys:
                            if key in p:
                                label = key_labels[key]
                                aggregations[key].append(
                                    f"{label} {len(aggregations[key]) + 1}\n{fix_thinking(p[key])}"
                                )

                if len(aggregations) > 0:
                    new_p = deepcopy(problem)
                    for key, parts in aggregations.items():
                        new_p[key] = "\n\n".join(parts)
                    new_p["compiled_solutions_ids"] = solution_ids
                    problems.append(new_p)

        else:
            id_map = {
                (p.get("problem_id"), p.get("model_id", "human/human")): p
                for p in raw_problems
                if p.get("problem_id")
            }

            part_suffix_pattern = re.compile(r"^(.*)-part(\d+)$")

            for p in raw_problems:
                p_id = p.get("problem_id")
                p_model = p.get("model_id", "human/human")

                if not p_id or part_suffix_pattern.match(p_id):
                    continue

                aggregations = {}
                for key in multi_input_keys:
                    if key in p:
                        label = key_labels[key]
                        aggregations[key] = [f"{label} 1\n{fix_thinking(p[key])}"]

                display_idx = 1
                lookup_idx = 1
                while True:
                    target_part_id = f"{p_id}-part{lookup_idx}"

                    if (target_part_id, p_model) in id_map:
                        part_obj = id_map[(target_part_id, p_model)]
                        for key in multi_input_keys:
                            if key in aggregations and key in part_obj:
                                label = key_labels[key]
                                aggregations[key].append(
                                    f"{label} {display_idx + 1}\n{fix_thinking(part_obj[key])}"
                                )
                        lookup_idx += 1
                        display_idx += 1
                    else:
                        break

                if include_human and "human" not in p_id:
                    human_model_id = "human/human"

                    if (p_id, human_model_id) in id_map:
                        h_base_obj = id_map[(p_id, human_model_id)]

                        for key in multi_input_keys:
                            if key in aggregations and key in h_base_obj:
                                label = key_labels[key]
                                aggregations[key].append(
                                    f"{label} {display_idx + 1}\n{fix_thinking(h_base_obj[key])}"
                                )

                        display_idx += 1

                        h_lookup_idx = 1
                        while True:
                            h_target_id = f"{p_id}-part{h_lookup_idx}"
                            if (h_target_id, human_model_id) in id_map:
                                h_part_obj = id_map[(h_target_id, human_model_id)]
                                for key in multi_input_keys:
                                    if key in aggregations and key in h_part_obj:
                                        label = key_labels[key]
                                        aggregations[key].append(
                                            f"{label} {display_idx + 1}\n{fix_thinking(h_part_obj[key])}"
                                        )
                                h_lookup_idx += 1
                                display_idx += 1
                            else:
                                break

                new_p = p.copy()
                for key, parts in aggregations.items():
                    new_p[key] = "\n\n".join(parts)

                problems.append(new_p)

        if setting_config.get("n_solutions", -1) != -1:
            random.shuffle(problems)
            problems = problems[: setting_config.get("n_solutions", -1)]
    else:
        all_files = glob.glob(
            f"{setting_config['data_path']}/**/*.json", recursive=True
        )

        part_pattern = re.compile(r"-part\d+\.json$")
        base_files = [f for f in all_files if not part_pattern.search(f)]

        problems = []
        for base_file in base_files:
            try:
                with open(base_file, "r") as f:
                    problem_data = json.load(f)

                if multi_input_keys:

                    aggregations = {}
                    for key in multi_input_keys:
                        if key in problem_data:
                            label = key_labels[key]
                            content = problem_data[key]
                            aggregations[key] = [f"{label} 1\n{content}"]

                    base_path_root = os.path.splitext(base_file)[0]
                    display_idx = 1
                    lookup_idx = 1

                    while True:
                        part_filename = f"{base_path_root}-part{lookup_idx}.json"
                        if not os.path.isfile(part_filename):
                            break
                        try:
                            with open(part_filename, "r") as pf:
                                part_data = json.load(pf)

                            for key in multi_input_keys:
                                if key in aggregations and key in part_data:
                                    label = key_labels[key]
                                    content = part_data[key]
                                    aggregations[key].append(
                                        f"{label} {display_idx + 1}\n{content}"
                                    )
                        except Exception as e:
                            logger.error(
                                f"Error reading part file {part_filename}: {e}"
                            )

                        lookup_idx += 1
                        display_idx += 1

                    if include_human and "human" not in problem_data["model_id"]:
                        current_model_id = problem_data.get("model_id", "human/human")
                        if current_model_id:
                            human_base_file = base_file.replace(
                                current_model_id, "human/human"
                            )

                            if os.path.exists(human_base_file):
                                try:
                                    with open(human_base_file, "r") as hf:
                                        h_data = json.load(hf)

                                    for key in multi_input_keys:
                                        if key in aggregations and key in h_data:
                                            label = key_labels[key]
                                            aggregations[key].append(
                                                f"{label} {display_idx + 1}\n{h_data[key]}"
                                            )

                                    display_idx += 1

                                    h_base_path_root = os.path.splitext(
                                        human_base_file
                                    )[0]
                                    h_lookup_idx = 1

                                    while True:
                                        h_part_filename = f"{h_base_path_root}-part{h_lookup_idx}.json"
                                        if not os.path.isfile(h_part_filename):
                                            break
                                        try:
                                            with open(h_part_filename, "r") as hpf:
                                                h_part_data = json.load(hpf)

                                            for key in multi_input_keys:
                                                if (
                                                    key in aggregations
                                                    and key in h_part_data
                                                ):
                                                    label = key_labels[key]
                                                    aggregations[key].append(
                                                        f"{label} {display_idx + 1}\n{h_part_data[key]}"
                                                    )
                                        except Exception as e:
                                            logger.error(
                                                f"Error reading human part file {h_part_filename}: {e}"
                                            )

                                        h_lookup_idx += 1
                                        display_idx += 1

                                except Exception as e:
                                    logger.warning(
                                        f"Could not load human file {human_base_file}: {e}"
                                    )

                    for key, parts in aggregations.items():
                        problem_data[key] = "\n\n".join(parts)

                problems.append(problem_data)
            except Exception as e:
                logger.error(f"Error loading base problem file {base_file}: {e}")

    valid_solvers = set(setting_config.get("solver_models", []))

    output_dir = os.path.join(output_folder, setting_config["name"])
    os.makedirs(output_dir, exist_ok=True)

    processor = results_preprocessor()

    all_grades_per_problem = {i: [] for i in range(len(problems))}
    detailed_costs_per_problem = {i: [] for i in range(len(problems))}
    failed_attempts_per_problem = {i: 0 for i in range(len(problems))}

    # Initial Pass: Check existing files
    for i, problem in enumerate(problems):
        try:
            problem_id = problem["problem_id"].replace("/", "_")
        except:
            # Fallback if problem_id is missing/broken, though validation above usually catches this
            continue

        model_id = problem.get("model_id", "human/human")

        if not compile_all and len(valid_solvers) > 0 and model_id not in valid_solvers:
            continue

        output_file = os.path.join(
            output_dir, config_path, model_id, f"{problem_id}.json"
        )

        if skip_existing and os.path.exists(output_file):
            try:
                existing_results = json.load(open(output_file))
                existing_messages = existing_results["messages"]

                if "detailed_costs" in existing_results:
                    detailed_costs = existing_results["detailed_costs"]
                else:
                    cost = existing_results["cost"]
                    detailed_costs = [
                        {
                            "cost": cost["cost"] if i == 0 else 0,
                            "input_tokens": cost["input_tokens"] if i == 0 else 0,
                            "output_tokens": cost["output_tokens"] if i == 0 else 0,
                        }
                        for i in range(len(existing_results))
                    ]

                detailed_costs_per_problem[i] = detailed_costs
                all_grades_per_problem[i] = existing_messages
                logger.info(
                    f"Skipping problem: {problem_id} ({len(existing_results)} times)"
                )

                if len(existing_messages) == n or reparse:
                    processor.process_results(
                        model_config=model_config,
                        config_path=config_path,
                        problem=problem,
                        output_dir=output_dir,
                        existing_results=None,
                        all_grades_per_problem=existing_messages,
                        detailed_costs_per_problem=detailed_costs,
                        model=model,
                        **setting_config,
                    )
            except Exception as e:
                logger.error(f"Error loading existing file {output_file}: {e}")

    # Initialize API
    if api != "baseline":
        api = APIQuery(model=model, api=api, **kwargs)

    prompt_template = setting_config["prompt"]

    executor = QueryExecutor(
        querier=api,
        system_prompt=setting_config.get("system_prompt", None),
        prompt_template=prompt_template,
    )

    # Retry Loop
    while True:
        batch_prompts = []
        batch_idx_to_problem_idx = {}

        # Construct Batch
        for i, problem in enumerate(problems):
            problem_id = problem.get("problem_id", str(i))
            model_id = problem.get("model_id", "human/human")

            # Skip invalid solvers
            if (
                not compile_all
                and len(valid_solvers) > 0
                and model_id not in valid_solvers
            ):
                continue

            # Calculate how many generated so far vs how many needed
            current_count = len(all_grades_per_problem[i])
            needed = n - current_count

            # If we still need solutions
            if needed > 0:
                # Check retry limit
                if failed_attempts_per_problem[i] >= n_retries:
                    logger.warning(
                        f"Max retries ({n_retries}) reached for problem {problem_id}. Stopping with {current_count}/{n} solutions."
                    )
                    continue

                # Add to batch
                for _ in range(needed):
                    batch_idx_to_problem_idx[len(batch_prompts)] = i
                    batch_prompts.append(problem)

        if len(batch_prompts) == 0:
            logger.info("All problems completed or max retries reached.")
            break

        logger.info(f"Running batch with {len(batch_prompts)} queries...")

        # Execute Batch
        for idx, messages, detailed_cost in executor.execute(batch_prompts):
            problem_idx = batch_idx_to_problem_idx[idx]
            problem = problems[problem_idx]

            # Validate the generation specific to the processor requirements
            last_content = messages[-1]["content"]
            if isinstance(last_content, list):
                last_content = last_content[-1]["content"]

            is_valid = processor.validate_single_generation(last_content)

            if is_valid:
                all_grades_per_problem[problem_idx].append(messages)
                detailed_costs_per_problem[problem_idx].append(detailed_cost)

                # Check completion and save
                if len(all_grades_per_problem[problem_idx]) == n:
                    processor.process_results(
                        model_config=model_config,
                        config_path=config_path,
                        problem=problem,
                        output_dir=output_dir,
                        existing_results=None,
                        all_grades_per_problem=all_grades_per_problem[problem_idx],
                        detailed_costs_per_problem=detailed_costs_per_problem[
                            problem_idx
                        ],
                        model=model,
                        **setting_config,
                    )
            else:
                failed_attempts_per_problem[problem_idx] += 1
                logger.warning(
                    f"Result rejected by processor for problem {problem.get('problem_id')}. "
                    f"Retrying... ({failed_attempts_per_problem[problem_idx]}/{n_retries})"
                )

    api.free_model()
