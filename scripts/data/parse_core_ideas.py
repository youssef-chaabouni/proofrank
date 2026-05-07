from proofrank.core_idea import CoreIdeaAnnotationParser
from proofrank.postprocess import fix_thinking
from pathlib import Path
import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description="Parse core idea annotations.")
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Input file path"
    )
    parser.add_argument(
        "--processor", "-p", type=str, required=True, help="Core idea processor config"
    )
    parser.add_argument(
        "--judge_model",
        "-j",
        type=str,
        default="openai/oss-120b",
        help="Core idea extractor model",
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True, help="Output file path"
    )
    args = parser.parse_args()

    parser = CoreIdeaAnnotationParser()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    for entry in data:
        if os.path.exists(
            f"outputs/{args.processor}/{args.judge_model}/{entry['model_id']}/{entry['problem_id']}.json"
        ):
            with open(
                f"outputs/{args.processor}/{args.judge_model}/{entry['model_id']}/{entry['problem_id']}.json",
                "r",
                encoding="utf-8",
            ) as f:
                annotation = json.load(f)
                if isinstance(annotation["outputs"], list):
                    output = annotation["outputs"][0]
                else:
                    output = annotation["outputs"]
                output = fix_thinking(output)
                parser.parse(output)
                if parser.parsing_warnings:
                    print(
                        f"Warnings for {entry['problem_id']} (model: {entry['model_id']}):"
                    )
                    for warning in parser.parsing_warnings:
                        print(f"  - {warning}")
                    continue
                entry["core_idea"] = parser.core_idea if parser.core_idea is not None else "Empty"
                entry["solution_summary"] = parser.solution_summary if parser.solution_summary is not None else entry["solution"]
                try:
                    entry["family"] = parser.family if parser.family is not None else "Empty"
                    entry["backbone"] = parser.backbone if parser.backbone is not None else "Empty"
                    entry["closing_engine"] = parser.closing_engine if parser.closing_engine is not None else "Empty"
                except:
                    pass
        else:
            entry["solution_summary"] = entry["solution"] if len(entry["solution"]) > 0 else "No solution provided."

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
