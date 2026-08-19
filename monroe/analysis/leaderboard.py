"""Leaderboard comparison against published baselines."""

import argparse
import json
from pathlib import Path
from typing import Any

from monroe.eval.polaris import metric_direction, polaris_benchmarks

# Load data from JSON assets
_ASSETS_DIR = Path(__file__).parent.parent / "assets"

with open(_ASSETS_DIR / "leaderboard_tdc.json") as f:
    leaderboard_tdc: dict[str, dict[str, dict]] = json.load(f)

with open(_ASSETS_DIR / "leaderboard_polaris.json") as f:
    leaderboard_polaris: dict[str, dict[str, dict]] = json.load(f)


def compute_mean_rank(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Compute mean rank across Polaris/TDC benchmarks using leaderboard data.

    Args:
        results: Dict mapping seed -> {task_name: {metric: score}}

    Returns:
        Dict with mean_rank, per-task ranks, and aggregated scores
    """
    # Combine leaderboards
    combined_leaderboard = {}
    for task, methods in leaderboard_tdc.items():
        combined_leaderboard[task] = {m: d["mean"] for m, d in methods.items()}

    for task, methods in leaderboard_polaris.items():
        if task not in combined_leaderboard:
            combined_leaderboard[task] = {}
        combined_leaderboard[task].update({m: d["mean"] for m, d in methods.items()})

    # Aggregate scores across seeds
    task_scores = {}
    for seed, tasks in results.items():
        if seed.startswith("_"):  # skip metadata keys like "_summary"
            continue
        for task, metrics in tasks.items():
            if task not in polaris_benchmarks:
                continue

            metric_name = polaris_benchmarks[task]
            if metric_name not in metrics:
                continue

            val = metrics[metric_name]
            if task not in task_scores:
                task_scores[task] = []
            task_scores[task].append(val)

    # Calculate mean scores and rankings
    all_ranks = []
    task_ranks = {}
    task_means = {}

    for task in polaris_benchmarks.keys():
        if task not in task_scores or task not in combined_leaderboard:
            continue

        values = task_scores[task]
        my_mean = sum(values) / len(values)
        task_means[task] = my_mean

        # Collect competitor scores
        competitor_scores = list(combined_leaderboard[task].values())
        competitor_scores.append(my_mean)

        # Determine sorting order using metric_direction
        metric = polaris_benchmarks[task]
        higher_is_better = metric_direction.get(metric, True)

        # Sort scores
        competitor_scores.sort(reverse=higher_is_better)

        # Find rank (1-based)
        rank = competitor_scores.index(my_mean) + 1
        all_ranks.append(rank)
        task_ranks[task] = rank

    mean_rank = sum(all_ranks) / len(all_ranks) if all_ranks else float("nan")

    return {
        "mean_rank": mean_rank,
        "n_tasks_ranked": len(all_ranks),
        "task_ranks": task_ranks,
        "task_scores": task_means,
    }


def run_leaderboard(results: str | dict):
    if isinstance(results, str):
        with open(results, "r") as f:
            results = json.load(f)

    rank_info = compute_mean_rank(results)

    if rank_info["task_ranks"]:
        print(f"{'Task Name':<45} | {'Score':<8} | {'Rank':<6}")

        for task in polaris_benchmarks.keys():
            if task not in rank_info["task_ranks"]:
                continue
            score = rank_info["task_scores"][task]
            rank = rank_info["task_ranks"][task]
            print(f"{task:<45} | {score:.4f}   | {rank:<6}")

        print("-" * 70)
        print(f"Mean Rank: {rank_info['mean_rank']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, required=True)
    args = parser.parse_args()

    run_leaderboard(args.json)
