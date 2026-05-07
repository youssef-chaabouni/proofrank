import json
from collections import defaultdict
def main():
    with open('data/raw/technique_adaptivity/initial_sample.json', 'r') as f:
        data = json.load(f)
    cluster_counts_per_problem = defaultdict(int)
    truncated_data = []
    for entry in data:
        problem_id = entry['problem_id']
        cluster_counts_per_problem[problem_id.split('-tech')[0]] += 1
        if cluster_counts_per_problem[problem_id.split('-tech')[0]] <= 4:
            truncated_data.append(entry)

    with open('data/raw/technique_adaptivity/sample.json', 'w') as f:
        json.dump(truncated_data, f, indent=4)

    print(f"Original dataset size: {len(data)}")
    print(f"Truncated dataset size: {len(truncated_data)}")

    pids_per_count = defaultdict(list)
    for entry in data:
        base_id = entry['problem_id'].split('-tech')[0]
        count = cluster_counts_per_problem[base_id]
        pids_per_count[count].append(base_id)
    
    print("\nDistribution of unique human-only techniques per problem:")
    for count in sorted(pids_per_count.keys()):
        print(f"{count} techniques: {len(pids_per_count[count])//count} problems")
        
if __name__ == "__main__":
    main()