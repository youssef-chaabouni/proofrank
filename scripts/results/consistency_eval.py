import argparse
import os
import json
from llmteach.analysis.verbosity import calculate_length
from llmteach.postprocess import fix_thinking
from collections import defaultdict
from llmteach.parser import find_last_boxed_content
from llmteach.result_utils import parse_outputs
from pathlib import Path
from tqdm import tqdm

def process_outputs(project, data):
    if "verbosity_rephrase" in project:
        sol_len_tuple = calculate_length(data['solution'])
        sol_spacy_count = sol_len_tuple[3]
        rephrase_stats = [calculate_length(fix_thinking(o)) for o in data['outputs']]
        avg_rephrase_spacy = sum(s[3] for s in rephrase_stats) / len(rephrase_stats) if rephrase_stats else 0
        ratio = sol_spacy_count / avg_rephrase_spacy if avg_rephrase_spacy > 0 else 0
        return ratio
    

TARGET_MODELS = {
    'deepseek/deepseek_v32_think',
    'gemini/gemini-3-flash',
    'gemini/gemini-31-pro',
    'openai/gpt-54',
    'stepfun/3.5-flash',
    'glm/glm-5',
    'xai/grok-41-fast-reasoning',
    'moonshot/k25',
    'qwen/qwen35_397b_a17b_high',
    "openai/oss-120b",
}

JUDGE_NAME = "oss-120b"


def load_correctness():
    eval_settings = ["answer_checker", "completeness_checker"]
    dfs = {}
    for setting in eval_settings:

        if not os.path.exists(os.path.join('outputs', setting)):
            continue

        df = parse_outputs('outputs', 'configs/', setting, target_models=TARGET_MODELS, judge_name=JUDGE_NAME)


        dfs[setting] = df
    
    correctness_map = {}
    if "answer_checker" in dfs and not dfs["answer_checker"].empty:
        for i, row in dfs["answer_checker"].iterrows():
            is_answer_correct = False
            if "judgements" in row:
                corrects = ["incorrect" not in j.lower() for j in row["judgements"]]
                is_answer_correct = sum(corrects) / len(corrects) >= 0.5
            elif isinstance(row["outputs"], list):
                try:
                    corrects = [
                        "incorrect" not in find_last_boxed_content(j)[0].lower()
                        for j in row["outputs"]
                        if find_last_boxed_content(j)[0] is not None
                    ]
                    is_answer_correct = sum(corrects) / len(corrects) >= 0.5
                except:
                    # breakpoint()
                    pass
            else:
                is_answer_correct = "incorrect" not in row.get("true_grade", False).lower()
            correctness_map[(row["solver_id"], row["problem_id"])] = is_answer_correct
    
    if "completeness_checker" in dfs and not dfs["completeness_checker"].empty:
        for i, row in dfs["completeness_checker"].iterrows():
            is_complete = False
            if "judgements" in row:
                corrects = ["incomplete" not in j.lower() for j in row["judgements"]]
                is_complete = sum(corrects) / len(corrects) >= 0.5
            elif isinstance(row["outputs"], list):
                try:
                    corrects = [
                        "incomplete" not in find_last_boxed_content(j)[0].lower()
                        for j in row["outputs"]
                        if find_last_boxed_content(j)[0] is not None
                    ]
                    is_complete = sum(corrects) / len(corrects) >= 0.5
                except:
                    # breakpoint()
                    pass
            else:
                is_complete = "incomplete" not in row.get("true_grade", False).lower()
            correctness_map[(row["solver_id"], row["problem_id"])] = correctness_map.get((row["solver_id"], row["problem_id"]), False) and is_complete
    
    return correctness_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("--processor1", type=str, required=True)
    parser.add_argument("--processor2", type=str, required=True)

    args = parser.parse_args()
    diffs = defaultdict(list)
    results_1 = defaultdict(lambda: defaultdict(float))
    results_2 = defaultdict(lambda: defaultdict(float))
    correctness_map = load_correctness()

    path1 = f"outputs/{args.project}/{args.processor1}"
    for file in tqdm(Path(path1).rglob("**.json")):
        with open(file, "r") as f:
            data = json.load(f)
            if isinstance(data['outputs'], str):
                data['outputs'] = [data['outputs']]
        if (data['model_id'], data['problem_id']) not in correctness_map:
            continue
        if not os.path.exists(str(file).replace(args.processor1, args.processor2, 1)):
            continue
        with open(str(file).replace(args.processor1, args.processor2), "r") as f:
            data2 = json.load(f)
            if isinstance(data2['outputs'], str):
                data2['outputs'] = [data2['outputs']]
        outs1 = process_outputs(args.project, data)
        outs2 = process_outputs(args.project, data2)
        diffs[data['model_id']].append((outs1 - outs2))
        results_1[data['problem_id']][data['model_id']] = outs1
        results_2[data['problem_id']][data['model_id']] = outs2

    # Measure pairwise 
    pairwise_agreements = defaultdict(list)
    winrates_1 = defaultdict(list)
    winrates_2 = defaultdict(list)
    pairwise_winrates_1 = defaultdict(list)
    pairwise_winrates_2 = defaultdict(list)
    for problem_id in results_1:
        for i, model_id_1 in enumerate(results_1[problem_id]):
            for model_id_2 in list(results_1[problem_id].keys())[i+1:]:
                pairwise_agreements[model_id_1].append((results_1[problem_id][model_id_1] > results_1[problem_id][model_id_2]) == (results_2[problem_id][model_id_1] > results_2[problem_id][model_id_2]))
                pairwise_agreements[model_id_2].append((results_1[problem_id][model_id_1] > results_1[problem_id][model_id_2]) == (results_2[problem_id][model_id_1] > results_2[problem_id][model_id_2]))
                winrates_1[model_id_1].append(results_1[problem_id][model_id_1] > results_1[problem_id][model_id_2])
                winrates_2[model_id_1].append(results_2[problem_id][model_id_1] > results_2[problem_id][model_id_2])
                winrates_1[model_id_2].append(results_1[problem_id][model_id_2] > results_1[problem_id][model_id_1])
                winrates_2[model_id_2].append(results_2[problem_id][model_id_2] > results_2[problem_id][model_id_1])
                alphabetical_model_1 = model_id_1 if model_id_1 < model_id_2 else model_id_2
                alphabetical_model_2 = model_id_2 if model_id_1 < model_id_2 else model_id_1
                pairwise_winrates_1[(alphabetical_model_1, alphabetical_model_2)].append(results_1[problem_id][model_id_1] > results_1[problem_id][model_id_2])
                pairwise_winrates_2[(alphabetical_model_1, alphabetical_model_2)].append(results_2[problem_id][model_id_1] > results_2[problem_id][model_id_2])   
    for model_id, agreements in pairwise_agreements.items():
        agreement_rate = sum(agreements) / len(agreements) if agreements else 0
        print(f"Model: {model_id}, Pairwise Agreement Rate: {agreement_rate:.2f}")
    winrate_diffs = []
    # Print winrates
    for model_id in winrates_1.keys():
        wins = winrates_1[model_id]
        winrate_1 = sum(wins) / len(wins) if wins else 0
        print(f"Model: {model_id}, Winrate in Processor 1: {winrate_1:.2f}")
        wins = winrates_2[model_id]
        winrate_2 = sum(wins) / len(wins) if wins else 0
        print(f"Model: {model_id}, Winrate in Processor 2: {winrate_2:.2f}")
        winrate_diffs.append(abs(winrate_1 - winrate_2))
    print(f"Average Winrate Diff: {sum(winrate_diffs)/len(winrate_diffs) if winrate_diffs else 0:.7f}")
    for model_id1, model_id2 in pairwise_winrates_1.keys():
        wins = pairwise_winrates_1[(model_id1, model_id2)]
        winrate = sum(wins) / len(wins) if wins else 0
        print(f"Model Pair: ({model_id1} vs {model_id2}), Winrate in Processor 1: {winrate:.2f}")
        wins = pairwise_winrates_2[(model_id1, model_id2)]
        winrate = sum(wins) / len(wins) if wins else 0
        print(f"Model Pair: ({model_id1} vs {model_id2}), Winrate in Processor 2: {winrate:.2f}")
    
    for model_id, diff_list in diffs.items():
        print(f"Model: {model_id}, Diff: {sum(diff_list)/len(diff_list)}")
    for model_id, diff_list in diffs.items():
        print(f"Model: {model_id}, Diff: {sum(diff_list)/len(diff_list)}")
    if diffs:
        print(f"Overall Diff: {sum([sum(diff_list) for diff_list in diffs.values()])/sum([len(diff_list) for diff_list in diffs.values()])}")

if __name__ == "__main__":
    main()