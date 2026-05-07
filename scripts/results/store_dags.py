from argparse import ArgumentParser
from pathlib import Path
from proofrank.dag import SolutionGraphParser
from proofrank.analysis.dag_parsing import analyse_single
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import os


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("base_path", type=str, help="Base directory path")
    args = parser.parse_args()
    base_path = Path(args.base_path)
    json_files = base_path.rglob("*.json")

    graph_parser = SolutionGraphParser()
    analysis_results = {}

    for file_path in tqdm(json_files):
        analysis_results[str(file_path)] = analyse_single(
            base_path, file_path, graph_parser
        )

    print(f"\nProcessed {len(analysis_results)} files.")

    depths = [
        item["depth"]
        for item in analysis_results.values()
        if item and item["depth"] >= 0
    ]
    n_nodes = [
        len(item["edges"])
        for item in analysis_results.values()
        if item and item["depth"] >= 0
    ]

    if depths:
        plt.figure(figsize=(10, 6))
        plt.hist(
            n_nodes, bins=range(0, max(n_nodes) + 2), edgecolor="black", align="left"
        )
        plt.hist(
            depths, bins=range(0, max(depths) + 2), edgecolor="black", align="left"
        )
        plt.title("Histogram of Solution Depths")
        plt.xlabel("Depth (Longest Path Length)")
        plt.ylabel("Frequency")
        plt.grid(axis="y", alpha=0.75)

        plot_filename = "depth_histogram.png"
        plt.savefig(plot_filename)
        print(f"\nHistogram saved to {plot_filename}")

    os.makedirs("data/postprocess/solution_graphs/", exist_ok=True)
    with open("data/postprocess/solution_graphs/test_samples.json", "w") as f:
        json.dump(
            [value for _, value in analysis_results.items() if value is not None],
            f,
            indent=4,
        )
