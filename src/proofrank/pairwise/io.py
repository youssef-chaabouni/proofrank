import csv
import json
import math
from tqdm import tqdm


def load_json(path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def iter_json_files(root, desc = "Loading"):
    if not root.exists():
        return
    for file_path in tqdm(sorted(root.rglob("*.json")), desc=desc, leave=False):
        raw = load_json(file_path)
        if raw is not None:
            yield file_path, raw


def print_markdown_table(rows, headers):
    def fmt(value):
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.3f}" if math.isfinite(value) else ""
        return str(value)

    table = [headers] + [[fmt(r.get(h)) for h in headers] for r in rows]
    widths = [max(len(r[i]) for r in table) for i in range(len(headers))]

    def render(items):
        return "| " + " | ".join(
            item.ljust(widths[i]) for i, item in enumerate(items)
        ) + " |"

    print(render(table[0]))
    print("| " + " | ".join("-" * w for w in widths) + " |")
    for row in table[1:]:
        print(render(row))


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)