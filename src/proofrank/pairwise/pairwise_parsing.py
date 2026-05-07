from proofrank.parser import find_last_boxed_content


def parse_pairwise_problem_field(raw_problem_id):
    if "-part" not in raw_problem_id:
        if "sol" in raw_problem_id:
        
            return raw_problem_id.split("_sol")[0], "unknown_model"
        else:
            return None
    try:
        actual_pid, encoded = raw_problem_id.rsplit("-part", 1)
        return actual_pid, encoded.replace("_", "/", 1)
    except Exception:
        return None


def vote_to_winner(majority_vote):
    vote = str(majority_vote)
    if vote == "1":
        return "model_a"
    if vote == "2":
        return "model_b"
    return "tie"


def extract_boxed_label(text):
    if not isinstance(text, str):
        return None
    result = find_last_boxed_content(text, list_answer=False)

    if result is None:
        return None
    if isinstance(result, tuple):
        content = result[0]
    else:
        content = result
    if content is None:
        return None

    stripped = content.strip().strip("$").strip()
    for token in ("0", "1"):
        if token in stripped:
            if stripped == token:
                return int(token)

    for ch in stripped:
        if ch in ("0", "1"):
            return int(ch)
    return None


def extract_per_judge_votes(record):
    votes = []

    outputs = record.get("outputs")
    if isinstance(outputs, list):
        for out in outputs:
            text = out if isinstance(out, str) else out.get("text") if isinstance(out, dict) else None
            label = extract_boxed_label(text) if text is not None else None
            if label is not None:
                votes.append(label)
        if votes:
            return votes

    judgements = record.get("judgements")
    if isinstance(judgements, list):
        for j in judgements:
            if isinstance(j, (int, float)) and int(j) in (0, 1):
                votes.append(int(j))
            elif isinstance(j, str):
                label = extract_boxed_label(j)
                if label is not None:
                    votes.append(label)
            elif isinstance(j, dict):
                for key in ("vote", "label", "answer"):
                    if key in j and j[key] is not None:
                        try:
                            v = int(j[key])
                            if v in (0, 1):
                                votes.append(v)
                                break
                        except Exception:
                            pass
    return votes


def majority_label(votes):
    if not votes:
        return None
    ones = sum(votes)
    zeros = len(votes) - ones
    if ones > zeros:
        return 1
    if zeros > ones:
        return 0
    return None 