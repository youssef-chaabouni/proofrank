from .io import iter_json_files
from .pairwise_parsing import (
    extract_per_judge_votes,
    majority_label,
    parse_pairwise_problem_field,
    vote_to_winner,
)


def load_pairwise_records(
    outputs_root,
    project,
    judges,
    target_models = None,
    record_builder = None,
):
    records = []

    for judge in judges:
        base = outputs_root / project / judge.replace("_", "/", 1)
        for file_path, raw in iter_json_files(base,  desc=f"Loading {project} ({judge})"):
            model_a = raw.get("model_id")
            parsed = parse_pairwise_problem_field(raw.get("problem_id", ""))
            if not model_a or parsed is None:
                breakpoint()
                continue
            problem, model_b = parsed

            if target_models is not None and (
                model_a not in target_models or model_b not in target_models
            ):
                continue

            base_record = {
                "judge": judge,
                "project": project,
                "problem": problem,
                "model_a": model_a,
                "model_b": model_b,
                "source_file": str(file_path),
            }

            if record_builder is not None:
                extras = record_builder(raw, base_record) or {}
                base_record.update(extras)
            else:
                base_record.update(_default_extras(raw))

            records.append(base_record)

    return records


def _default_extras(raw):
    votes = extract_per_judge_votes(raw)
    maj = majority_label(votes)
    if maj is None and raw.get("majority_vote") is not None:
        try:
            mv = int(raw["majority_vote"])
            if mv in (0, 1):
                maj = mv
        except Exception:
            pass
    return {"votes": votes, "n_votes": len(votes), "majority": maj}


def ground_truth_record_builder(raw, base):
    return {
        "winner": vote_to_winner(raw.get("majority_vote")),
        "majority_vote": raw.get("majority_vote"),
        "judgements": raw.get("judgements"),
        "outputs": raw.get("outputs"),
    }