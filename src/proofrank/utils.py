import re
import sympy
from sympy.parsing.latex import parse_latex
import os
import json
import glob


def latex2sympy_fixed(latex: str):
    # if _integer is present, replace it with _{integer} for any integer
    latex = re.sub(r"_([0-9]+)", r"_{\1}", latex)
    latex_parsed = parse_latex(latex)
    # replace constants like pi and e with their numerical value
    known_constants = {"pi": sympy.pi, "e": sympy.E}

    # Replace any symbol in expr that is in our known_constants dictionary.
    expr = latex_parsed.xreplace(
        {
            s: known_constants[s.name]
            for s in latex_parsed.free_symbols
            if s.name in known_constants
        }
    )
    return expr


def get_latex_array(mat: list[list[str]]) -> str:
    n_cols = len(mat[0])
    cs = "{" + "c" * n_cols + "}"
    return (
        "\\begin{array}"
        + cs
        + "\n"
        + "\n".join([" & ".join(row) + " \\\\" for row in mat])
        + "\n\\end{array}"
    )


def convert_to_int(obj):
    if isinstance(obj, str):
        return int(obj)
    if isinstance(obj, list):
        return [convert_to_int(item) for item in obj]
    raise ValueError(f"Cannot convert {type(obj)} to int")


def load_judged_data(dir, regexes=None):
    all_data = {}

    for folder in os.listdir(dir):
        all_data[folder] = {}
        glob_files = glob.glob(os.path.join(dir, folder, "**/*.json"), recursive=True)

        if regexes is not None:
            glob_files = [
                file
                for file in glob_files
                if any(
                    re.match(regex, file.replace(f"{dir}/{folder}/", ""))
                    for regex in regexes
                )
            ]

        for file in glob_files:
            file_name = file.replace(f"{dir}/{folder}/", "")
            data_file = []
            raw_data = json.load(open(file, "r"))
            for attempt in raw_data["attempts"]:
                if (
                    "grading" not in attempt
                    or attempt["grading"] is None
                    or attempt["grading"]["score"] is None
                ):
                    continue
                grading = attempt["grading"]
                data_file.append(
                    {
                        "model": attempt["model_id"],
                        "metadata": raw_data.get("metadata", {}),
                        "score": grading["score"],
                        "uncertain": grading["uncertain"],
                        "no_feedback": grading["no_feedback"],
                        "long": grading.get("long", False),
                        "grading": grading,
                        "solution": attempt["solution"],
                        "problem_issue": raw_data.get("issue") is not None,
                    }
                )
            all_data[folder][file_name] = data_file

    return all_data


def identify_overlapping_judgments(all_data):
    overlapping_scores = []
    judges_list = list(all_data.keys())
    for i, judge in enumerate(judges_list):
        for j in range(i + 1, len(judges_list)):
            judge2 = judges_list[j]
            for file_name in all_data[judge]:
                if file_name in all_data[judge2]:
                    for k, attempt in enumerate(all_data[judge][file_name]):
                        for k2, attempt2 in enumerate(all_data[judge2][file_name]):
                            if (
                                attempt["model"] != attempt2["model"]
                                or attempt["solution"] != attempt2["solution"]
                            ):
                                continue
                            uncertain = attempt2["uncertain"] or attempt2["uncertain"]
                            if attempt2["no_feedback"] or attempt["no_feedback"]:
                                continue
                            if attempt2["long"] or attempt["long"]:
                                continue
                            is_equal = attempt["score"] == attempt2["score"]
                            overlapping_scores.append(
                                {
                                    "judge": judge,
                                    "judge2": judge2,
                                    "file_name": file_name,
                                    "run": k,
                                    "run2": k2,
                                    "uncertain": uncertain,
                                    "is_equal": is_equal,
                                }
                            )

    return overlapping_scores


def extract_solution(text: str, only_numeric=False) -> str:
    start_tag = r"\boxed{"
    start = text.rfind(start_tag)
    if start == -1:
        return "none" if not only_numeric else "0"

    i = start + len(start_tag)
    depth = 1
    while i < len(text) and depth:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1

    if depth:  # never balanced → abort
        return "none" if not only_numeric else "0"

    sol = text[start + len(start_tag) : i - 1]  # slice inside the braces

    while True:
        new_sol = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", sol)
        if new_sol == sol:  # nothing left to unwrap
            break
        sol = new_sol

    sol = re.sub(r"\s+", "", sol).lower()
    if only_numeric:  # only keep numeric and "."
        sol = re.sub(r"[^0-9.]", "", sol)
        if not sol:
            sol = "0"
    return sol


def fix_invalid_escapes(s: str) -> str:
    """
    Fix invalid escapes inside JSON string values while preserving
    structural whitespace (newlines between keys, array elements, etc.)
    """
    result = []
    i = 0
    n = len(s)

    while i < n:
        # Look for the start of a JSON string
        if s[i] == '"':
            result.append('"')
            i += 1
            string_chars = []

            # Process characters inside the string
            while i < n:
                if s[i] == "\\" and i + 1 < n:
                    next_char = s[i + 1]
                    # Valid JSON escape sequences
                    if next_char in '"\\/bfnrtu':
                        string_chars.append(s[i : i + 2])
                        i += 2
                    else:
                        # Invalid escape - double the backslash
                        string_chars.append("\\\\")
                        i += 1
                elif s[i] == '"':
                    # End of string
                    break
                elif s[i] == "\n":
                    string_chars.append("\\n")
                    i += 1
                elif s[i] == "\r":
                    string_chars.append("\\r")
                    i += 1
                elif s[i] == "\t":
                    string_chars.append("\\t")
                    i += 1
                elif ord(s[i]) < 32:
                    # Other control characters - escape as unicode
                    string_chars.append(f"\\u{ord(s[i]):04x}")
                    i += 1
                else:
                    string_chars.append(s[i])
                    i += 1

            result.append("".join(string_chars))

            if i < n and s[i] == '"':
                result.append('"')
                i += 1
        else:
            # Outside of string - keep as-is (structural whitespace is fine)
            result.append(s[i])
            i += 1

    return "".join(result)


def fix_edges_tuples(s: str) -> str:
    """
    Convert tuple syntax (...) to list syntax [...] in the "edges" array.
    Handles formats like:
      "edges": [("1", "2"), ("3", "4")]
      "edges": [(1, 2), (3, 4)]
    Converts to:
      "edges": [["1", "2"], ["3", "4"]]
      "edges": [[1, 2], [3, 4]]
    """
    # Pattern to match the "edges" key and its array value
    # This captures everything between "edges": [ and the closing ]
    pattern = r'("edges"\s*:\s*\[)(.*?)(\])'

    def replace_tuples_with_lists(match):
        prefix = match.group(1)  # "edges": [
        content = match.group(2)  # the array contents
        suffix = match.group(3)  # ]

        # Replace ( with [ and ) with ] within the edges content
        # But only for tuple-like structures, not inside strings
        fixed_content = []
        i = 0
        n = len(content)

        while i < n:
            if content[i] == '"':
                # Inside a string - copy until closing quote
                fixed_content.append(content[i])
                i += 1
                while i < n:
                    if content[i] == "\\" and i + 1 < n:
                        fixed_content.append(content[i : i + 2])
                        i += 2
                    elif content[i] == '"':
                        fixed_content.append(content[i])
                        i += 1
                        break
                    else:
                        fixed_content.append(content[i])
                        i += 1
            elif content[i] == "(":
                fixed_content.append("[")
                i += 1
            elif content[i] == ")":
                fixed_content.append("]")
                i += 1
            else:
                fixed_content.append(content[i])
                i += 1

        return prefix + "".join(fixed_content) + suffix

    # Use re.DOTALL to match across newlines
    return re.sub(pattern, replace_tuples_with_lists, s, flags=re.DOTALL)


def fix_trailing_commas(s: str) -> str:
    """
    Remove illegal trailing commas before closing brackets ] or braces }.
    Handles cases like:
      {"a": 1, "b": 2,}  ->  {"a": 1, "b": 2}
      [1, 2, 3,]         ->  [1, 2, 3]
      {"items": [1, 2,],} -> {"items": [1, 2]}
    Only removes commas outside of string values.
    """
    result = []
    i = 0
    n = len(s)

    while i < n:
        if s[i] == '"':
            # Inside a string - copy until closing quote (handle escapes)
            result.append(s[i])
            i += 1
            while i < n:
                if s[i] == "\\" and i + 1 < n:
                    result.append(s[i : i + 2])
                    i += 2
                elif s[i] == '"':
                    result.append(s[i])
                    i += 1
                    break
                else:
                    result.append(s[i])
                    i += 1
        elif s[i] == ",":
            # Check if this comma is followed only by whitespace and then ] or }
            j = i + 1
            while j < n and s[j] in " \t\n\r":
                j += 1

            if j < n and s[j] in "]}":
                # This is a trailing comma - skip it
                i += 1
            else:
                # Valid comma - keep it
                result.append(s[i])
                i += 1
        else:
            result.append(s[i])
            i += 1

    return "".join(result)


def prepare_json(s: str) -> str:
    """
    Apply all JSON fixes before parsing.
    """
    s = fix_edges_tuples(s)
    s = fix_trailing_commas(s)
    s = fix_invalid_escapes(s)
    return s
