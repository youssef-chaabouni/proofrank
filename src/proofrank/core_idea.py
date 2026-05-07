import json
import re
from typing import Any, Dict, List, Optional


class CoreIdeaAnnotationParser:
    """
    Parser and validator for the core-idea annotation schema.
    """

    REQUIRED_KEYS = {
        "noCoreIdea",
        "solutionSummary",
        "coreIdea",
        "supportingIdeas",
        "family",
        "backbone",
        "closingEngine",
        "evidenceQuotes",
        "confidence",
    }

    def __init__(self):
        self.no_core_idea: bool = False
        self.solution_summary: Optional[str] = None
        self.core_idea: Optional[str] = None
        self.supporting_ideas: List[str] = []
        self.family: Optional[str] = None
        self.backbone: Optional[str] = None
        self.closing_engine: Optional[str] = None
        self.evidence_quotes: List[str] = []
        self.confidence: Optional[float] = None

        self.parsing_warnings: List[str] = []
        self.raw_json_data: Any = None
        self.non_json_text: str = ""

    def _reset(self):
        self.no_core_idea = False
        self.solution_summary = None
        self.core_idea = None
        self.supporting_ideas = []
        self.family = None
        self.backbone = None
        self.closing_engine = None
        self.evidence_quotes = []
        self.confidence = None

        self.parsing_warnings = []
        self.raw_json_data = None
        self.non_json_text = ""

    def parse(self, llm_output: str) -> bool:
        self._reset()

        try:
            json_str, extra_text = self._extract_json(llm_output)
            self.non_json_text = extra_text.strip()

            if self.non_json_text:
                self.parsing_warnings.append(
                    "Output contains non-JSON text outside the JSON payload."
                )

            duplicate_keys: List[str] = []

            def object_pairs_hook(pairs):
                obj = {}
                for key, value in pairs:
                    if key in obj:
                        duplicate_keys.append(key)
                    obj[key] = value
                return obj

            data = json.loads(json_str, object_pairs_hook=object_pairs_hook)
            self.raw_json_data = data

            if duplicate_keys:
                unique_dupes = sorted(set(duplicate_keys))
                self.parsing_warnings.append(
                    f"Duplicate JSON key(s) detected: {unique_dupes}."
                )

            if not isinstance(data, dict):
                self.parsing_warnings.append(
                    f"Top-level JSON value must be an object, got {type(data).__name__}."
                )
                return True

            self.no_core_idea = data.get("noCoreIdea", False)
            self.solution_summary = data.get("solutionSummary")
            self.core_idea = data.get("coreIdea")
            self.supporting_ideas = data.get("supportingIdeas", [])
            self.family = data.get("family")
            self.backbone = data.get("backbone")
            self.closing_engine = data.get("closingEngine")
            self.evidence_quotes = data.get("evidenceQuotes", [])
            self.confidence = data.get("confidence")

            self._validate()
            return True

        except ValueError as e:
            self.parsing_warnings.append(str(e))
            return False
        except json.JSONDecodeError as e:
            self.parsing_warnings.append(f"JSON syntax error: {str(e)}")
            return False
        except Exception as e:
            self.parsing_warnings.append(f"Critical parser error: {str(e)}")
            return False

    def _extract_json(self, llm_output: str) -> (str, str):
        fenced_blocks = list(re.finditer(r"```json\s*(.*?)```", llm_output, re.DOTALL))

        if len(fenced_blocks) > 1:
            self.parsing_warnings.append(
                "Multiple ```json``` blocks detected; using the first one."
            )

        if fenced_blocks:
            match = fenced_blocks[0]
            json_str = match.group(1).strip()
            extra_text = (
                llm_output[: match.start()] + llm_output[match.end() :]
            ).strip()
            return json_str, extra_text

        start = llm_output.find("{")
        end = llm_output.rfind("}")

        if start == -1 or end == -1 or end < start:
            raise ValueError("No JSON object detected in output.")

        json_str = llm_output[start : end + 1]
        extra_text = (llm_output[:start] + llm_output[end + 1 :]).strip()
        return json_str, extra_text

    def _validate(self):
        if not isinstance(self.raw_json_data, dict):
            return

        data = self.raw_json_data

        missing = self.REQUIRED_KEYS - set(data.keys())
        unexpected = set(data.keys()) - self.REQUIRED_KEYS

        for key in sorted(missing):
            self.parsing_warnings.append(f"Missing required key: '{key}'.")

        for key in sorted(unexpected):
            self.parsing_warnings.append(f"Unexpected key present: '{key}'.")

        self._validate_bool_field(data, "noCoreIdea")
        self._validate_nullable_string_field(data, "solutionSummary")
        self._validate_nullable_string_field(data, "coreIdea")
        self._validate_string_list_field(data, "supportingIdeas")
        self._validate_nullable_string_field(data, "family")
        self._validate_nullable_string_field(data, "backbone")
        self._validate_nullable_string_field(data, "closingEngine")
        self._validate_string_list_field(data, "evidenceQuotes")
        self._validate_confidence_field(data, "confidence")

        self._validate_cross_field_consistency(data)

    def _validate_bool_field(self, data: Dict[str, Any], key: str):
        if key not in data:
            return
        if not isinstance(data[key], bool):
            self.parsing_warnings.append(
                f"'{key}' must be a boolean, got {type(data[key]).__name__}."
            )

    def _validate_nullable_string_field(self, data: Dict[str, Any], key: str):
        if key not in data:
            return

        value = data[key]
        if value is not None and not isinstance(value, str):
            self.parsing_warnings.append(
                f"'{key}' must be a string or null, got {type(value).__name__}."
            )
            return

        if isinstance(value, str) and value.strip() == "":
            self.parsing_warnings.append(f"'{key}' must not be an empty string.")

    def _validate_string_list_field(self, data: Dict[str, Any], key: str):
        if key not in data:
            return

        value = data[key]
        if not isinstance(value, list):
            self.parsing_warnings.append(
                f"'{key}' must be a list, got {type(value).__name__}."
            )
            return

        for i, item in enumerate(value):
            if not isinstance(item, str):
                self.parsing_warnings.append(
                    f"'{key}[{i}]' must be a string, got {type(item).__name__}."
                )
            elif item.strip() == "":
                self.parsing_warnings.append(
                    f"'{key}[{i}]' must not be an empty string."
                )

    def _validate_confidence_field(self, data: Dict[str, Any], key: str):
        if key not in data:
            return

        value = data[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            self.parsing_warnings.append(
                f"'{key}' must be a number, got {type(value).__name__}."
            )
            return

        if not (0.0 <= float(value) <= 1.0):
            self.parsing_warnings.append(
                f"'{key}' must be between 0 and 1, got {value}."
            )

    def _validate_cross_field_consistency(self, data: Dict[str, Any]):
        no_core = data.get("noCoreIdea")
        core_idea = data.get("coreIdea")

        if isinstance(no_core, bool):
            if no_core is True and core_idea is not None:
                self.parsing_warnings.append(
                    "Consistency error: 'noCoreIdea' is true but 'coreIdea' is not null."
                )
            elif no_core is False and core_idea is None:
                self.parsing_warnings.append(
                    "Consistency error: 'noCoreIdea' is false but 'coreIdea' is null."
                )

    def is_valid(self) -> bool:
        return len(self.parsing_warnings) == 0

    def get_formatted_report(self) -> str:
        lines = [
            f"**Has Core Idea:** {'No' if self.no_core_idea else 'Yes'} | "
            f"**Confidence:** {self.confidence if self.confidence is not None else 'N/A'}",
        ]

        if self.core_idea is not None:
            lines.append(f"\n### Core Idea\n> {self.core_idea}")

        if self.solution_summary is not None:
            lines.append(f"\n### Solution Summary\n{self.solution_summary}")

        if self.family is not None:
            lines.append(f"\n### Family\n{self.family}")

        if self.backbone is not None:
            lines.append(f"\n### Backbone\n{self.backbone}")

        if self.closing_engine is not None:
            lines.append(f"\n### Closing Engine\n{self.closing_engine}")

        if self.supporting_ideas:
            lines.append("\n### Supporting Ideas")
            for idea in self.supporting_ideas:
                lines.append(f"- {idea}")

        if self.evidence_quotes:
            lines.append("\n### Evidence Quotes")
            for quote in self.evidence_quotes:
                lines.append(f"- {quote}")

        if self.non_json_text:
            lines.append("\n### Extra Non-JSON Text")
            lines.append(self.non_json_text)

        if self.parsing_warnings:
            lines.append("\n### Validation Warnings")
            for warning in self.parsing_warnings:
                lines.append(f"- {warning}")

        return "\n".join(lines)
