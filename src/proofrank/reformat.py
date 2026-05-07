import json
import re
from typing import Dict, List


class CleanedMathProblemParser:
    def __init__(self):
        self.problem: str = ""
        self.solution: str = ""
        self.parsing_warnings: List[str] = []
        self.raw_json_data: Dict = {}
        self.human_readable_text: str = ""

    def _reset(self):
        self.problem = ""
        self.solution = ""
        self.parsing_warnings = []
        self.raw_json_data = {}
        self.human_readable_text = ""

    def parse(self, llm_output: str) -> bool:
        self._reset()

        try:
            json_match = re.search(r"```json\s*(.*?)```", llm_output, re.DOTALL)

            if json_match:
                json_str = json_match.group(1).strip()
                self.human_readable_text = llm_output.replace(
                    json_match.group(0), ""
                ).strip()
            else:
                start = llm_output.find("{")
                end = llm_output.rfind("}")
                if start != -1 and end != -1:
                    json_str = llm_output[start : end + 1]
                    self.human_readable_text = (
                        llm_output[:start] + llm_output[end + 1 :]
                    ).strip()
                else:
                    self.parsing_warnings.append(
                        "No JSON block detected in the output."
                    )
                    return False

            data = json.loads(json_str)
            self.raw_json_data = data

            self.problem = data.get("problem", "")
            self.solution = data.get("solution", "")
            self.answer = data.get("answer", "")

            self._validate_consistency()

            if not self.problem or not self.solution:
                return False

            return True

        except json.JSONDecodeError as e:
            self.parsing_warnings.append(f"JSON Syntax Error: {str(e)}")
            return False
        except Exception as e:
            self.parsing_warnings.append(f"Critical Parser Error: {str(e)}")
            return False

    def _validate_consistency(self):
        if "problem" not in self.raw_json_data:
            self.parsing_warnings.append("Missing key 'problem' in JSON.")
        if "solution" not in self.raw_json_data:
            self.parsing_warnings.append("Missing key 'solution' in JSON.")
        if "answer" not in self.raw_json_data:
            self.parsing_warnings.append("Missing key 'answer' in JSON.")

        if not self.problem:
            self.parsing_warnings.append("The problem field is empty.")
        if not self.solution:
            self.parsing_warnings.append("The solution field is empty.")
        if not self.answer:
            self.parsing_warnings.append("The answer field is empty.")

        if self.solution and r"\boxed" not in self.solution:
            self.parsing_warnings.append(
                "The solution does not appear to contain the final answer in \\boxed{...} or \\boxed{} format."
            )

    def get_formatted_report(self) -> str:
        lines = []
        lines.append("## Cleaned Math Problem Result")

        if self.parsing_warnings:
            lines.append("### Parser Warnings")
            for w in self.parsing_warnings:
                lines.append(f"- {w}")
            lines.append("")

        if self.problem:
            lines.append("### Cleaned Problem:")
            lines.append(self.problem)
        else:
            lines.append("> *Cleaned problem is missing.*")

        lines.append("")

        if self.solution:
            lines.append("### Cleaned Solution:")
            lines.append(self.solution)
        else:
            lines.append("> *Cleaned solution is missing.*")

        if self.human_readable_text:
            lines.append("\n### Additional Model Text:")
            lines.append(f"_{self.human_readable_text}_")

        return "\n".join(lines)
