import json
import re
from typing import Dict, List, Optional, Any, Set


class TechniqueDiversityParser:
    """
    Parses determining if a proof solution is correct and uses a specific mathematical technique.
    Handles the standard JSON output format: verdict, reason, and comments.
    """

    # Allowed enumeration values for validation
    VALID_VERDICTS = {"correct", "incorrect", "warning"}
    VALID_REASONS = {"correctness", "technique", "both", "unknown"}

    def __init__(self):
        self.parsed_data: Dict[str, Any] = {}
        self.parsing_warnings: List[str] = []
        self.parsed_successfully = False

    def parse(self, json_output: str) -> bool:
        """
        Parses the raw string output from the LLM.
        Returns True if JSON structure was valid and required keys were found.
        """
        self._reset()

        try:
            cleaned_json = re.sub(
                r"^```(json)?\s*", "", json_output.strip(), flags=re.IGNORECASE
            )
            cleaned_json = re.sub(r"\s*```$", "", cleaned_json)


            cleaned_json = re.sub(r",\s*}", "}", cleaned_json)
            cleaned_json = re.sub(r",\s*]", "]", cleaned_json)

            data = json.loads(cleaned_json)

            self.parsed_data = self._validate_fields(data)

            self.parsed_successfully = True
            return True

        except json.JSONDecodeError as e:
            self.parsing_warnings.append(
                f"JSON Decode Error: {e.msg} at line {e.lineno}"
            )
            return False
        except Exception as e:
            self.parsing_warnings.append(f"Critical Parse Error: {str(e)}")
            return False

    def _reset(self):
        self.parsed_data = {}
        self.parsing_warnings = []
        self.parsed_successfully = False

    def _validate_fields(self, data: dict) -> dict:
        """Validates and normalizes the three specific keys required by the prompt."""

        # --- Verdict ---
        raw_verdict = str(data.get("verdict", "")).lower().strip()
        if raw_verdict not in self.VALID_VERDICTS:
            self.parsing_warnings.append(
                f"Invalid verdict '{raw_verdict}'. Expected one of {self.VALID_VERDICTS}. Defaulting to 'incorrect'."
            )
            verdict = "incorrect"
        else:
            verdict = raw_verdict


        raw_reason = str(data.get("reason", "")).lower().strip()

        if verdict == "correct":
            reason = "correctness"
        elif verdict == "warning":
            reason = "unknown"
        else:
            if raw_reason not in self.VALID_REASONS:
                self.parsing_warnings.append(
                    f"Invalid reason '{raw_reason}'. Expected one of {self.VALID_REASONS}. Defaulting to 'unknown'."
                )
                reason = "unknown"
            else:
                reason = raw_reason

        comments = str(data.get("comments", ""))
        if not comments:
            self.parsing_warnings.append("Field 'comments' was empty.")

        return {"verdict": verdict, "reason": reason, "comments": comments}

    @property
    def verdict(self) -> str:
        """Returns the raw verdict (correct, incorrect, warning)."""
        return self.parsed_data.get("verdict", "incorrect")

    @property
    def reason(self) -> str:
        """Returns the raw reason category."""
        return self.parsed_data.get("reason", "unknown")

    @property
    def comments(self) -> str:
        """Returns the explanation text."""
        return self.parsed_data.get("comments", "")

    def is_solution_correct(self) -> bool:
        """
        Returns True ONLY if the logic of the proof was deemed correct.
        Note: A proof can be logically correct but fail the technique check.
        """
        if not self.parsed_successfully:
            return False

        if self.verdict == "correct":
            return True

        if self.verdict == "incorrect" and self.reason == "technique":
            return True

        return False

    def is_technique_followed(self) -> bool:
        """
        Returns True ONLY if the specific technique was used correctly.
        """
        if not self.parsed_successfully:
            return False

        if self.verdict == "correct":
            return True

        return False

    def is_valid_technique_request(self) -> bool:
        """
        Returns False if the judge determined the requested technique does not exist
        (i.e., verdict is 'warning'). This helps filter out bad prompts or
        hallucinated technique names.
        """
        return self.verdict != "warning"

    def get_failure_classification(self) -> Optional[str]:
        """
        Returns a high-level classification of why the proof failed.
        Returns None if the proof passed.
        """
        if not self.parsed_successfully:
            return "parse_error"

        if self.verdict == "correct":
            return None

        if self.verdict == "warning":
            return "invalid_technique_specification"

        if self.reason == "correctness":
            return "math_error"
        elif self.reason == "technique":
            return "wrong_technique"
        elif self.reason == "both":
            return "math_and_technique_error"
        else:
            return "ambiguous_failure"
