import argparse
import os
from llmteach.runner import run
from llmteach.results_processors import (
    ResultsProcessor,
    DefaultProcessor,
    JudgeProcessor,
    DAGProcessor,
    DiversityClusteringProcessor,
    GraphMetricsProcessor,
    TechniqueDiversityProcessor,
    ReformattingProcessor,
    CoreIdeaProcessor,
)
import yaml
import sys
from loguru import logger

sys.path.append("../")

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=1)
parser.add_argument("--checker_configs", type=str, nargs="+", required=True)
parser.add_argument("--processor_config", type=str, required=True)
parser.add_argument("--skip-existing", action="store_true")
parser.add_argument("--output-folder", type=str, default="outputs")
parser.add_argument("--configs-folder", type=str, default="configs/")
parser.add_argument("--reparse", action="store_true")

if __name__ == "__main__":
    args = parser.parse_args()
    if args.n % 2 != 1:
        logger.warning(
            "Even number of samples ({args.n}) may cause conflicts in majority voting."
        )
    processor_config_path = (
        args.processor_config + ".yaml"
        if not args.processor_config.endswith(".yaml")
        else args.processor_config
    )
    with open(f"{args.configs_folder}/processors/{processor_config_path}", "r") as f:
        processor_config = yaml.safe_load(f)

    for config_path in args.checker_configs:
        config_path = (
            config_path + ".yaml" if not config_path.endswith(".yaml") else config_path
        )
        with open(f"{args.configs_folder}/models/{config_path}", "r") as f:
            judge_config = yaml.safe_load(f)

        judge_config["n"] = judge_config.get("n", args.n)
        logger.info(f"Running config: {config_path}")
        if processor_config.get("task", None) == "judge":
            processor = JudgeProcessor
        elif processor_config.get("task", None) == "dag":
            processor = DAGProcessor
        elif processor_config.get("task", None) == "diversity":
            processor = DiversityClusteringProcessor
        elif processor_config.get("task", None) == "dag_classifier":
            processor = GraphMetricsProcessor
        elif processor_config.get("task", None) == "technique":
            processor = TechniqueDiversityProcessor
        elif processor_config.get("task", None) == "reformat":
            processor = ReformattingProcessor
        elif processor_config.get("task", None) == "core_ideas":
            processor = CoreIdeaProcessor
        else:
            processor = DefaultProcessor
        run(
            judge_config,
            processor_config,
            config_path.replace(".yaml", ""),
            skip_existing=args.skip_existing,
            output_folder=args.output_folder,
            reparse=args.reparse,
            results_preprocessor=processor,
        )
