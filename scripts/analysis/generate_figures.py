#!/usr/bin/env python3
"""Generate publication figures from benchmark results.

Usage:
    python scripts/analysis/generate_figures.py --fdr --scale
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from monroe.analysis.loader import (
    enrich_moleculeace_cliff_delta,
    load_benchmark_results,
)
from monroe.analysis.plots import (
    plot_combined_summary,
    plot_moleculeace_cliff_delta,
)
from monroe.analysis.stats import compute_win_stats
from monroe.analysis.style import sort_methods
from monroe.eval.polaris import metric_direction, polaris_benchmarks


def extract_moleculeace_cliff_data(
    all_results: dict[str, dict],
) -> tuple[dict[str, dict[str, dict[str, float]]], list[str]]:
    """Extract per-method per-task cliff delta and RMSE statistics.

    Args:
        all_results: Dict mapping method_name -> {seed -> {task -> {metric -> value}}}
                     (must already be enriched via enrich_moleculeace_cliff_delta)

    Returns:
        (data, sorted_tasks) where:
          data[method][task] = {delta_mean, delta_std, rmse_mean, rmse_std}
          sorted_tasks is ordered by best overall RMSE (ascending)
    """
    raw: dict[str, dict[str, dict[str, list[float]]]] = {}
    all_tasks: set[str] = set()

    for method, seeds in all_results.items():
        raw[method] = {}
        for seed, tasks in seeds.items():
            if seed.startswith("_"):
                continue
            for task, metrics in tasks.items():
                cliff = metrics.get("cliff_test_rmse")
                overall = metrics.get("overall_test_rmse")
                if cliff is not None and overall is not None:
                    if task not in raw[method]:
                        raw[method][task] = {"deltas": [], "rmses": []}
                    raw[method][task]["deltas"].append(cliff - overall)
                    raw[method][task]["rmses"].append(overall)
                    all_tasks.add(task)

    # Compute mean/std
    data: dict[str, dict[str, dict[str, float]]] = {}
    for method, tasks in raw.items():
        data[method] = {}
        for task, vals in tasks.items():
            n = len(vals["deltas"])
            data[method][task] = {
                "delta_mean": float(np.mean(vals["deltas"])),
                "delta_std": float(np.std(vals["deltas"], ddof=1)) if n > 1 else 0.0,
                "rmse_mean": float(np.mean(vals["rmses"])),
                "rmse_std": float(np.std(vals["rmses"], ddof=1)) if n > 1 else 0.0,
            }

    # Sort tasks by best (minimum) mean overall RMSE across methods
    task_best_rmse = {}
    for task in all_tasks:
        best = float("inf")
        for method in data:
            if task in data[method]:
                best = min(best, data[method][task]["rmse_mean"])
        task_best_rmse[task] = best

    sorted_tasks = sorted(all_tasks, key=lambda t: task_best_rmse[t])

    return data, sorted_tasks


def generate_moleculeace_cliff_figure(
    all_results: dict[str, dict],
    output_path: Path,
) -> None:
    """Generate MoleculeACE cliff delta error bar figure."""
    data, sorted_tasks = extract_moleculeace_cliff_data(all_results)
    methods = sort_methods(list(data.keys()))

    fig = plot_moleculeace_cliff_delta(
        methods=methods,
        tasks=sorted_tasks,
        data=data,
    )

    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate publication figures")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing method results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures"),
        help="Output directory for figures",
    )
    parser.add_argument(
        "--benchmark",
        nargs="+",
        default=["polaris", "moleculeace"],
        help="Benchmarks to process",
    )
    parser.add_argument(
        "--no-scale", action="store_false", dest="scale",
        help="Disable MAE scaling",
    )
    parser.add_argument(
        "--no-fdr", action="store_false", dest="fdr",
        help="Disable Benjamini-Hochberg FDR correction",
    )
    parser.set_defaults(scale=True, fdr=True)
    parser.add_argument(
        "--mace-metric",
        default="rmse",
        choices=["delta", "rmse"],
        help="MoleculeACE metric for win rate: delta (cliff-overall) or rmse (overall)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load MAE scales if requested
    scales = None
    if args.scale:
        scales_path = Path(__file__).resolve().parent.parent.parent / "monroe" / "assets" / "polaris_scales.json"
        with open(scales_path) as f:
            scales = json.load(f)

    benchmark_stats = {}
    moleculeace_results = None

    for benchmark in args.benchmark:
        print(f"\nProcessing {benchmark}...")

        # Load results
        all_results = load_benchmark_results(args.results_dir, benchmark)
        if not all_results:
            print(f"  No results found for {benchmark}, skipping")
            continue

        print(f"  Found {len(all_results)} methods: {list(all_results.keys())}")

        # Get tasks and metrics
        if benchmark == "moleculeace":
            # Enrich results with cliff_delta_rmse (cliff - overall)
            all_results = enrich_moleculeace_cliff_delta(all_results)
            moleculeace_results = all_results
            # Derive task list from loaded results
            task_set: set[str] = set()
            for method_results in all_results.values():
                for seed, seed_data in method_results.items():
                    if seed.startswith("_"):
                        continue
                    task_set.update(seed_data.keys())
            tasks = sorted(task_set)
            if args.mace_metric == "rmse":
                metric_per_task = {t: "overall_test_rmse" for t in tasks}
            else:
                metric_per_task = {t: "cliff_delta_rmse" for t in tasks}
        else:
            tasks = list(polaris_benchmarks.keys())
            metric_per_task = polaris_benchmarks

        # Compute stats (needed for combined figure)
        stats = compute_win_stats(
            all_results,
            tasks=tasks,
            metric_per_task=metric_per_task,
            higher_better=metric_direction,
            scales=scales if benchmark == "polaris" else None,
            use_fdr=args.fdr,
            equal_var=True,   # Tukey HSD, as the paper reports
        )
        benchmark_stats[benchmark] = stats

        print(f"  Valid tasks: {stats['n_tasks']}")

    # Generate combined figure
    if "polaris" in benchmark_stats and "moleculeace" in benchmark_stats:
        fig = plot_combined_summary(
            polaris_stats=benchmark_stats["polaris"],
            moleculeace_stats=benchmark_stats["moleculeace"],
            output_path=str(args.output_dir / "combined_summary.pdf"),
        )
        plt.close(fig)

    # Generate MoleculeACE cliff delta figure
    if moleculeace_results is not None:
        generate_moleculeace_cliff_figure(
            moleculeace_results,
            args.output_dir / "moleculeace_cliff_delta.pdf",
        )


if __name__ == "__main__":
    main()
