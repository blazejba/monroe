"""Results loading and aggregation utilities."""

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_results(path: str | Path) -> dict[str, Any]:
    """Load results from a JSON file.

    Args:
        path: Path to JSON results file

    Returns:
        Parsed JSON as dictionary
    """
    with open(path, "r") as f:
        return json.load(f)


def aggregate_seeds(
    results: dict[str, dict],
    tasks: list[str] | None = None,
) -> dict[str, dict[str, dict]]:
    """Aggregate results across seeds, computing mean and std.

    Args:
        results: Dict mapping seed -> task -> metrics
        tasks: Optional list of tasks to include (default: all)

    Returns:
        Dict mapping task -> metric -> {mean, std, values}

    Example:
        >>> results = {
        ...     "seed_0": {"task1": {"mae": 0.5}},
        ...     "seed_1": {"task1": {"mae": 0.6}},
        ... }
        >>> agg = aggregate_seeds(results)
        >>> agg["task1"]["mae"]
        {'mean': 0.55, 'std': 0.05, 'values': [0.5, 0.6]}
    """
    # Collect values per task/metric
    collected: dict[str, dict[str, list]] = {}

    for seed, seed_results in results.items():
        if seed.startswith("_"):  # Skip metadata keys
            continue

        for task, metrics in seed_results.items():
            if tasks is not None and task not in tasks:
                continue

            if task not in collected:
                collected[task] = {}

            for metric, value in metrics.items():
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    continue

                if metric not in collected[task]:
                    collected[task][metric] = []
                collected[task][metric].append(value)

    # Compute statistics
    aggregated = {}
    for task, metrics in collected.items():
        aggregated[task] = {}
        for metric, values in metrics.items():
            if len(values) == 0:
                continue
            aggregated[task][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "values": values,
            }

    return aggregated


def get_task_values(
    results: dict[str, dict],
    task: str,
    metric: str,
) -> list[float]:
    """Extract values for a specific task/metric across all seeds.

    Args:
        results: Dict mapping seed -> task -> metrics
        task: Task name
        metric: Metric name

    Returns:
        List of values across seeds
    """
    values = []
    for seed, seed_results in results.items():
        if seed.startswith("_"):
            continue
        if task in seed_results and metric in seed_results[task]:
            val = seed_results[task][metric]
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                values.append(val)
    return values


def enrich_moleculeace_cliff_delta(
    all_results: dict[str, dict],
) -> dict[str, dict]:
    """Add cliff_delta_rmse (cliff_test_rmse - overall_test_rmse) to MoleculeACE results.

    This derived metric captures how much worse a model is on activity cliffs
    compared to the overall test set. Lower is better.

    Mutates and returns all_results.
    """
    for method, seeds in all_results.items():
        for seed, tasks in seeds.items():
            if seed.startswith("_"):
                continue
            for task, metrics in tasks.items():
                cliff = metrics.get("cliff_test_rmse")
                overall = metrics.get("overall_test_rmse")
                if cliff is not None and overall is not None:
                    metrics["cliff_delta_rmse"] = cliff - overall
    return all_results


def load_benchmark_results(results_dir: Path, benchmark: str) -> dict[str, dict]:
    """Load all method results for a benchmark.

    Supports two directory layouts:

    Layout 1 (benchmark-first):
        results/<benchmark>/<method>.json

    Layout 2 (method-first, legacy):
        results/<method>/<benchmark>_results.json  (or results.json)

    Args:
        results_dir: Directory containing benchmark results
        benchmark: Benchmark name (e.g., 'polaris', 'moleculeace')

    Returns:
        Dict mapping method_name -> results
    """
    all_results = {}

    # Layout 1: results/<benchmark>/<method>.json
    benchmark_dir = results_dir / benchmark
    if benchmark_dir.is_dir():
        for json_file in sorted(benchmark_dir.glob("*.json")):
            # A leading underscore marks a result that is not part of the method
            # comparison (e.g. the ensemble), so figures and tables skip it.
            if json_file.stem.startswith("_"):
                continue
            all_results[json_file.stem] = load_results(json_file)

    # Layout 2: results/<method>/<benchmark>_results.json
    if not all_results:
        for method_dir in results_dir.iterdir():
            if not method_dir.is_dir():
                continue
            results_file = method_dir / f"{benchmark}_results.json"
            if not results_file.exists():
                results_file = method_dir / "results.json"
            if results_file.exists():
                all_results[method_dir.name] = load_results(results_file)

    return all_results


def load_multiple_methods(
    paths: dict[str, str | Path],
) -> dict[str, dict]:
    """Load results for multiple methods.

    Args:
        paths: Dict mapping method_name -> path to results JSON

    Returns:
        Dict mapping method_name -> results
    """
    return {name: load_results(path) for name, path in paths.items()}


def extract_for_comparison(
    all_results: dict[str, dict],
    tasks: list[str],
    metric_per_task: dict[str, str],
) -> dict[str, dict[str, list[float]]]:
    """Extract values for comparison across methods.

    Args:
        all_results: Dict mapping method_name -> seed_results
        tasks: List of tasks to extract
        metric_per_task: Dict mapping task -> metric name

    Returns:
        Dict mapping method_name -> task -> list of values
    """
    extracted = {}

    for method_name, results in all_results.items():
        extracted[method_name] = {}
        for task in tasks:
            metric = metric_per_task.get(task)
            if metric is None:
                continue
            values = get_task_values(results, task, metric)
            if values:
                extracted[method_name][task] = values

    return extracted
