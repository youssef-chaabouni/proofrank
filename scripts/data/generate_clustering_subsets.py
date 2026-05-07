import argparse
import json
import random
import itertools
import copy
from collections import defaultdict
import math


def reindex_multiple_solutions(samples):
    """
    Reassigns problem_ids by appending '-partX' to duplicate problem IDs for the same model.
    """
    id_counter = defaultdict(lambda: defaultdict(int))
    for sample in samples:
        sample["original_problem_id"] = sample["problem_id"]
        if id_counter[sample["model_id"]][sample["original_problem_id"]] > 0:
            sample[
                "problem_id"
            ] += f"-part{id_counter[sample['model_id']][sample['original_problem_id']]}"
        id_counter[sample["model_id"]][sample["original_problem_id"]] += 1


def main():
    parser = argparse.ArgumentParser(
        description="Group problems and sample subsets of solutions."
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Path to input JSON file."
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True, help="Path to output JSON file."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Maximum number of distinct subsets to sample per problem.",
    )
    parser.add_argument(
        "--min_subset",
        type=int,
        default=2,
        help="Minimum number of solutions in a subset.",
    )
    parser.add_argument(
        "--max_subset",
        type=int,
        default=None,
        help="Maximum number of solutions in a subset. Defaults to all.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for deterministic sampling."
    )

    args = parser.parse_args()
    random.seed(args.seed)

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    grouped_data = defaultdict(list)
    for entry in data:
        base_id = entry["problem_id"].split("-part")[0]
        grouped_data[base_id].append(entry)

    new_samples = []

    for base_id, entries in grouped_data.items():
        # Collect unique full entries based on the top-level "solution" string
        unique_entries = []
        seen_solution_texts = set()

        for entry in entries:
            sol_text = entry.get("solution")
            if sol_text and sol_text not in seen_solution_texts:
                seen_solution_texts.add(sol_text)
                unique_entries.append(entry)

        num_unique = len(unique_entries)
        if num_unique <= 1:
            print(
                f"Skipping {base_id} due to insufficient unique solutions (found {num_unique})."
            )
            continue

        # Determine subset size limits safely
        max_size = args.max_subset if args.max_subset is not None else num_unique
        max_size = min(max_size, num_unique)
        min_size = min(args.min_subset, max_size)

        # Efficient Sampling Approach
        sizes = list(range(min_size, max_size + 1))
        weights = [math.comb(num_unique, r) for r in sizes]
        total_combs = sum(weights)

        sampled_indices = set()

        if total_combs <= args.n:
            # Safe to generate all since total combinations are extremely small
            for r in sizes:
                for combo in itertools.combinations(range(num_unique), r):
                    sampled_indices.add(combo)
        else:
            # Rejection sampling using combinatorial weights mapping to standard uniform distribution over sets
            attempts = 0
            max_attempts = args.n * 10
            while len(sampled_indices) < args.n and attempts < max_attempts:
                r = random.choices(sizes, weights=weights)[0]
                # Sample random unique indices of size r, storing them as a sorted tuple to ensure distinct subset representation
                combo = tuple(sorted(random.sample(range(num_unique), r)))
                sampled_indices.add(combo)
                attempts += 1

        sampled_subsets = [
            [unique_entries[i] for i in idx_tuple] for idx_tuple in sampled_indices
        ]

        # Create flattened entries for every combined subset
        for subset_idx, subset in enumerate(sampled_subsets):
            subset_base_id = f"{base_id}_subset{subset_idx}"

            for orig_entry in subset:
                # Copy the ENTIRE entry, not just the string
                new_entry = copy.deepcopy(orig_entry)
                new_entry["problem_id"] = subset_base_id

                # Strip out 'solutions' if it was present as an artifact
                if "solutions" in new_entry:
                    del new_entry["solutions"]

                new_samples.append(new_entry)

    # Reindex using the provided function logic
    reindex_multiple_solutions(new_samples)

    # Save data
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(new_samples, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
