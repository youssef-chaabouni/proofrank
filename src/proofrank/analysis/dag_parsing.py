from ..postprocess import fix_thinking
import json
from loguru import logger


def analyse_single(base_path, file_path, parser):

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_json = json.load(f)

        if "outputs" not in raw_json:
            return

        sample_json = raw_json["outputs"]
        sample_json = fix_thinking(sample_json)
        problem_statement = raw_json.get(
            "problem", "Problem statement not found in source JSON."
        )

        success = parser.parse(sample_json)

        if not success:
            logger.warning(
                f"Parsing failed for file: {file_path}: {parser.parsing_warnings}"
            )
            return

        leaves = parser.get_leaves()
        conclusion_node = parser.get_conclusion_node()
        depth = parser.compute_depth()
        width = parser.compute_width()

        result_obj = {
            "problem_id": raw_json.get("problem_id"),
            "problem": problem_statement,
            "model_id": raw_json.get("model_id"),
            "solution": raw_json.get("solution"),
            "thinking": raw_json.get("thinking"),
            "full_solution": raw_json.get("full_solution"),
            "solutions": raw_json.get("solutions", []),
            "steps_stringified": parser.get_stringified_steps(),
            "edges": list(parser.graph.edges()),
            "leaves": leaves,
            "conclusion_node": conclusion_node,
            "depth": depth,
            "is_cyclic": parser.check_cycles()[0],
            "warnings": parser.parsing_warnings,
            "width": width,
            "graph": parser.get_graph_json(),
        }

        return result_obj

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
