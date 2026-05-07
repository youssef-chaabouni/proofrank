from loguru import logger
from collections import Counter
import os
import json
import sympy
from .parser import WarningType, find_last_boxed_content
from .dag import SolutionGraphParser
from .diversity import DiversityAnalysisParser
from .dag_classifier import GraphMetricsParser
from .technique_diversity import TechniqueDiversityParser
from .reformat import CleanedMathProblemParser
from .core_idea import CoreIdeaAnnotationParser
from .postprocess import fix_thinking
import numpy as np


class ResultsProcessor:

    def __init__(self):
        pass

    def validate_single_generation(self, content) -> bool:
        """
        Validates a single generation output.
        Returns True if valid, False if it should be retried.
        """
        return True

    def process_results(self, **results):
        """Process results method to be implemented in subclasses. The method should handle saving or processing results as needed."""
        raise NotImplementedError(
            "ResultsProcessor must be instantiated in a separate class."
        )


class DefaultProcessor(ResultsProcessor):
    def __init__(self):
        super().__init__()

    def parse_params(self, **results):
        return (
            results["model_config"],
            results["config_path"],
            results["problem"],
            results["output_dir"],
            results["all_grades_per_problem"],
            results["detailed_costs_per_problem"],
            results["model"],
        )

    def log_processing_start(self):
        logger.info("DefaultProcessor: No processing applied to results.")

    def process_results(self, **results):
        self.log_processing_start()

        (
            model_config,
            config_path,
            problem,
            output_dir,
            messages_problem,
            costs_problem,
            model,
        ) = self.parse_params(**results)

        problem_id = problem["problem_id"].replace("/", "_")
        model_id = problem["model_id"]
        output_file = os.path.join(
            output_dir, config_path, model_id, f"{problem_id}.json"
        )
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        for i in range(len(costs_problem)):
            costs_problem[i]["cost"] = (
                model_config["read_cost"] * costs_problem[i]["input_tokens"]
                + model_config["write_cost"] * costs_problem[i]["output_tokens"]
            )
            costs_problem[i]["cost"] /= 10**6
        cost = {
            "cost": sum([d["cost"] for d in costs_problem]),
            "input_tokens": sum([d["input_tokens"] for d in costs_problem]),
            "output_tokens": sum([d["output_tokens"] for d in costs_problem]),
        }

        outputs = []

        for msg in messages_problem:
            if isinstance(msg[-1]["content"], str):
                outputs.append(msg[-1]["content"])
            elif isinstance(msg[-1]["content"], list):
                outputs.append(msg[-1]["content"][-1]["content"])
            else:
                outputs.append(str(msg[-1]["content"]))

        if len(outputs) == 1:
            outputs = outputs[0]

        with open(output_file, "w") as f:
            json.dump(
                {
                    "problem": problem.get("problem", problem.get("statement", "")),
                    "problem_id": problem_id,
                    "solutions": problem.get("solutions", []),
                    "full_solution": problem.get("full_solution", ""),
                    "thinking": problem.get("thinking", ""),
                    "solution": problem.get("solution", ""),
                    "model_id": model_id,
                    "cost": cost,
                    "messages": messages_problem,
                    "outputs": outputs,
                    "detailed_costs": costs_problem,
                },
                f,
                indent=4,
            )


class JSONParsingProcessor(DefaultProcessor):
    def __init__(self):
        super().__init__()

    def log_processing_start(self):
        raise NotImplementedError(
            "JSONParsingProcessor must be instantiated in a separate class."
        )

    def validate_single_generation(self, content) -> bool:
        text = content
        if isinstance(content, list):
            text = content[-1]["content"]
        elif isinstance(content, dict):
            text = content.get("content", str(content))

        text = fix_thinking(text)
        success = self.parser.parse(text)

        if not success:
            logger.warning(
                f"DAG Parsing failed. Warnings: {self.parser.parsing_warnings}"
            )
            return False

        if len(self.parser.parsing_warnings) > 0:
            logger.warning(
                f"DAG Parsing succeeded with warnings: {self.parser.parsing_warnings}"
            )
            return False

        return True


class DAGProcessor(JSONParsingProcessor):
    def __init__(self):
        super().__init__()
        self.parser = SolutionGraphParser()

    def log_processing_start(self):
        logger.info("DAGProcessor: Saving results.")


class GraphMetricsProcessor(JSONParsingProcessor):
    def __init__(self):
        super().__init__()
        self.parser = GraphMetricsParser()

    def log_processing_start(self):
        logger.info("GraphMetricsProcessor: Saving results.")


class ReformattingProcessor(JSONParsingProcessor):
    def __init__(self):
        super().__init__()
        self.parser = CleanedMathProblemParser()

    def log_processing_start(self):
        logger.info("ReformattingProcessor: Saving results.")


class CoreIdeaProcessor(JSONParsingProcessor):
    def __init__(self):
        super().__init__()
        self.parser = CoreIdeaAnnotationParser()

    def log_processing_start(self):
        logger.info("CoreIdeaProcessor: Saving results.")


class DiversityClusteringProcessor(JSONParsingProcessor):
    def __init__(self):
        super().__init__()
        self.parser = DiversityAnalysisParser()

    def log_processing_start(self):
        logger.info("DiversityClusteringProcessor: Saving results.")

    def process_results(self, **results):
        self.log_processing_start()

        (
            model_config,
            config_path,
            problem,
            output_dir,
            messages_problem,
            costs_problem,
            model,
        ) = self.parse_params(**results)

        problem_id = problem["problem_id"].replace("/", "_")
        model_id = problem["model_id"]
        output_file = os.path.join(
            output_dir, config_path, model_id, f"{problem_id}.json"
        )
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        for i in range(len(costs_problem)):
            costs_problem[i]["cost"] = (
                model_config["read_cost"] * costs_problem[i]["input_tokens"]
                + model_config["write_cost"] * costs_problem[i]["output_tokens"]
            )
            costs_problem[i]["cost"] /= 10**6
        cost = {
            "cost": sum([d["cost"] for d in costs_problem]),
            "input_tokens": sum([d["input_tokens"] for d in costs_problem]),
            "output_tokens": sum([d["output_tokens"] for d in costs_problem]),
        }

        outputs = []

        for msg in messages_problem:
            if isinstance(msg[-1]["content"], str):
                outputs.append(msg[-1]["content"])
            elif isinstance(msg[-1]["content"], list):
                outputs.append(msg[-1]["content"][-1]["content"])
            else:
                outputs.append(str(msg[-1]["content"]))

        if len(outputs) == 1:
            outputs = outputs[0]

        with open(output_file, "w") as f:
            json.dump(
                {
                    "problem": problem.get("problem", problem.get("statement", "")),
                    "problem_id": problem_id,
                    "solutions": problem.get("solutions", []),
                    "full_solution": problem.get("full_solution", ""),
                    "thinking": problem.get("thinking", ""),
                    "solution": problem.get("solution", ""),
                    "compiled_solutions_ids": problem.get("compiled_solutions_ids", []),
                    "model_id": model_id,
                    "cost": cost,
                    "messages": messages_problem,
                    "outputs": outputs,
                    "detailed_costs": costs_problem,
                },
                f,
                indent=4,
            )


class TechniqueDiversityProcessor(JSONParsingProcessor):
    def __init__(self):
        super().__init__()
        self.parser = TechniqueDiversityParser()

    def log_processing_start(self):
        logger.info("TechniqueDiversityProcessor: Saving results.")


class JudgeProcessor(ResultsProcessor):
    def __init__(self):
        super().__init__()

    def parse_params(self, **results):
        return (
            results["model_config"],
            results["config_path"],
            results["problem"],
            results["output_dir"],
            results["all_grades_per_problem"],
            results["detailed_costs_per_problem"],
            results["model"],
        )

    def process_results(self, **results):
        (
            model_config,
            config_path,
            problem,
            output_dir,
            messages_problem,
            costs_problem,
            executor_id,
        ) = self.parse_params(**results)

        problem_id = problem["problem_id"].replace("/", "_")
        model_id = problem.get("model_id", "human/human")
        output_file = os.path.join(
            output_dir, config_path, model_id, f"{problem_id}.json"
        )
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        n = len(messages_problem)
        judgements = []
        warnings = []
        corrects = []
        for j in range(n):
            model_answer = messages_problem[j][-1]["content"]
            if isinstance(model_answer, list):
                model_answer = model_answer[-1]["content"]
            model_judgement, warning = find_last_boxed_content(model_answer)

            if len(model_answer) == 0:
                logger.warning(f"Empty message in problem: {problem_id}, idx: {j}")
                warning = WarningType.MAJOR
            judgements.append(model_judgement)
            warnings.append(warning.value)
        try:
            logger.info(
                f"Finished problem {problem_id}, solved by {model_id} - judgements: {judgements}, #Correct: {sum(corrects)}"
            )
        except:
            pass

        counts = Counter(judgements)
        majority_vote = counts.most_common(1)[0][0]

        for i in range(len(costs_problem)):
            costs_problem[i]["cost"] = (
                model_config["read_cost"] * costs_problem[i]["input_tokens"]
                + model_config["write_cost"] * costs_problem[i]["output_tokens"]
            )
            costs_problem[i]["cost"] /= 10**6
        cost = {
            "cost": sum([d["cost"] for d in costs_problem]),
            "input_tokens": sum([d["input_tokens"] for d in costs_problem]),
            "output_tokens": sum([d["output_tokens"] for d in costs_problem]),
        }

        outputs = []

        for msg in messages_problem:
            if isinstance(msg[-1]["content"], str):
                outputs.append(msg[-1]["content"])
            elif isinstance(msg[-1]["content"], list):
                outputs.append(msg[-1]["content"][-1]["content"])
            else:
                outputs.append(str(msg[-1]["content"]))

        if len(outputs) == 1:
            outputs = outputs[0]

        with open(output_file, "w") as f:
            json.dump(
                {
                    "problem": problem["problem"],
                    "problem_id": problem_id,
                    "solutions": problem.get("solutions", []),
                    "full_solution": problem.get("full_solution", ""),
                    "thinking": problem.get("thinking", ""),
                    "solution": problem.get("solution", ""),
                    "grading_scheme": problem.get("grading_scheme", None),
                    "model_id": model_id,
                    "cost": cost,
                    "judgements": judgements,
                    "majority_vote": majority_vote,
                    "solution_cost": problem.get("cost", {}).get("cost", 0),
                    "detailed_costs": costs_problem,
                    "warnings": warnings,
                    "messages": messages_problem,
                    "outputs": outputs,
                },
                f,
                indent=4,
            )


def convert_answer(answer):
    try:
        if type(answer) == sympy.Integer:
            return int(answer)
        else:
            return str(answer)
    except:
        return "None"
