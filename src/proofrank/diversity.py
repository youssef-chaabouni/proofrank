import json
import re
from typing import Dict, List, Optional, Any, Tuple
from .utils import prepare_json


class DiversityAnalysisParser:
    def __init__(self):
        self.declared_N = 0
        self.declared_K = 0
        self.diversity_score = 0.0

        self.clusters: List[Dict[str, Any]] = []
        self.fingerprints: Dict[str, Dict[str, Any]] = {}
        self.raw_warnings: List[str] = []

        self.parsing_warnings: List[str] = []
        self.raw_json_data: Dict = {}
        self.human_readable_summary: str = ""

    def _reset(self):
        self.declared_N = 0
        self.declared_K = 0
        self.diversity_score = 0.0
        self.clusters = []
        self.fingerprints = {}
        self.raw_warnings = []
        self.parsing_warnings = []
        self.raw_json_data = {}
        self.human_readable_summary = ""

    def parse(self, llm_output: str) -> bool:
        """
        Parses the specific output format:
        1. JSON block
        2. Human readable summary (before or after)
        """
        self._reset()

        try:
            json_match = re.search(r"```json\s*(.*?)```", llm_output, re.DOTALL)

            if json_match:
                json_str = json_match.group(1)
                self.human_readable_summary = llm_output.replace(
                    json_match.group(0), ""
                ).strip()
            else:
                start = llm_output.find("{")
                end = llm_output.rfind("}")
                if start != -1 and end != -1:
                    json_str = llm_output[start : end + 1]
                    self.human_readable_summary = (
                        llm_output[:start] + llm_output[end + 1 :]
                    ).strip()
                else:
                    self.parsing_warnings.append("No JSON block detected in output.")
                    return False

            cleaned_json = prepare_json(json_str)
            data = json.loads(cleaned_json)
            self.raw_json_data = data

            self.declared_N = data.get("N", 0)
            self.declared_K = data.get("K", 0)
            self.diversity_score = data.get("diversity_score_D", 0.0)
            self.raw_warnings = data.get("warnings", [])

            raw_fingerprints = data.get("proof_fingerprints", [])
            for fp in raw_fingerprints:
                pid = str(fp.get("proof_id", "UNKNOWN"))

                if "primary_approach" not in fp:
                    self.parsing_warnings.append(
                        f"Proof {pid} missing 'primary_approach'."
                    )

                self.fingerprints[pid] = fp

            self.clusters = data.get("clusters", [])

            self._validate_consistency()

            return True

        except json.JSONDecodeError as e:
            self.parsing_warnings.append(f"JSON Syntax Error: {str(e)}")
            return False
        except Exception as e:
            self.parsing_warnings.append(f"Critical Parser Error: {str(e)}")
            return False

    def _validate_consistency(self):
        """
        Checks internal consistency of the analyzed math proofs.
        Example: Do the number of fingerprints match N? Do cluster members exist?
        """
        actual_n = len(self.fingerprints)
        if actual_n != self.declared_N:
            self.parsing_warnings.append(
                f"Consistency Error: Declared N={self.declared_N}, but found {actual_n} proof fingerprints."
            )

        actual_k = len(self.clusters)
        if actual_k != self.declared_K:
            self.parsing_warnings.append(
                f"Consistency Error: Declared K={self.declared_K}, but found {actual_k} clusters."
            )

        all_clustered_ids = set()
        for i, cluster in enumerate(self.clusters):
            members = cluster.get("members", [])
            cluster_id = cluster.get("cluster_id", f"C{i}")

            if not members:
                self.parsing_warnings.append(f"Cluster '{cluster_id}' is empty.")

            for m in members:
                m_str = str(m)
                if m_str not in self.fingerprints:
                    self.parsing_warnings.append(
                        f"Cluster '{cluster_id}' contains Proof ID '{m_str}' which has no analysis fingerprint."
                    )
                all_clustered_ids.add(m_str)

        fingerprint_ids = set(self.fingerprints.keys())
        orphans = fingerprint_ids - all_clustered_ids
        if orphans:
            self.parsing_warnings.append(
                f"Orphaned Proofs (analyzed but not in any cluster): {orphans}"
            )

    def get_formatted_report(self) -> str:
        """Returns a formatted Markdown summary of the analysis."""
        lines = []
        lines.append(f"## Diversity Analysis Report")
        lines.append(
            f"**Score (D):** {self.diversity_score:.2f} | **Clusters (K):** {len(self.clusters)} | **Proofs (N):** {len(self.fingerprints)}"
        )

        if self.declared_N != len(self.fingerprints) or self.declared_K != len(
            self.clusters
        ):
            lines.append(
                f"> ⚠️ **Stats Mismatch Warning**: Declared stats do not match parsed data items."
            )

        lines.append("\n### Clusters")
        for c in self.clusters:
            name = c.get("cluster_name", "Unnamed")
            method = c.get("defining_approach", "Unknown")
            members = c.get("members", [])
            lines.append(f"- **{name}** ({method})")
            lines.append(f"  - Members: {members}")
            if "defining_features" in c:
                feats = ", ".join(c["defining_features"])
                lines.append(f"  - Key Features: {feats}")

        lines.append("\n### Proof Fingerprints")
        sorted_ids = sorted(
            self.fingerprints.keys(), key=lambda x: int(x) if x.isdigit() else x
        )

        for pid in sorted_ids:
            fp = self.fingerprints[pid]
            lines.append(f"#### Proof {pid}")
            lines.append(f"- **Primary:** {fp.get('primary_approach', 'N/A')}")
            lines.append(
                f"- **Techniques:** {', '.join(fp.get('secondary_techniques', []))}"
            )
            lines.append(f"- **Pivot Step:** {fp.get('key_pivot_step', 'N/A')}")
            if fp.get("evidence_quotes"):
                lines.append(f"- *Evidence:* \"{fp.get('evidence_quotes')[0]}\"")

        if self.parsing_warnings:
            lines.append("\n### Parser Warnings")
            for w in self.parsing_warnings:
                lines.append(f"- ⚠️ {w}")

        if self.raw_warnings:
            lines.append("\n### LLM-Generated Warnings")
            for w in self.raw_warnings:
                lines.append(f"- {w}")

        return "\n".join(lines)
