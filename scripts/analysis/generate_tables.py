#!/usr/bin/env python3
"""Generate results tables from benchmark results.

Usage:
    python scripts/analysis/generate_tables.py --leaderboard --fdr --scale
"""

import argparse
import json
from pathlib import Path

import numpy as np

from monroe.analysis.leaderboard import leaderboard_polaris, leaderboard_tdc
from monroe.analysis.loader import (
    aggregate_seeds,
    enrich_moleculeace_cliff_delta,
    load_benchmark_results,
)
from monroe.analysis.stats import compute_win_stats
from monroe.analysis.tables import CombinedSummaryTable, LeaderboardTable, ResultsTable
from monroe.eval.polaris import metric_direction, polaris_benchmarks


def _deep_merge_leaderboards(
    *leaderboards: dict[str, dict[str, dict]],
) -> dict[str, dict[str, dict]]:
    """Deep merge leaderboards at the method level.

    Later sources override earlier ones for the same (task, method) pair.
    """
    merged: dict[str, dict[str, dict]] = {}
    for lb in leaderboards:
        for task, methods in lb.items():
            if task not in merged:
                merged[task] = {}
            merged[task].update(methods)
    return merged


def _transpose_leaderboard(
    raw: dict[str, dict[str, dict]],
    valid_tasks: list[str],
    exclude_methods: set[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Transpose leaderboard from task-first to method-first.

    Input:  {task: {method: {mean, std}}}
    Output: {method: {task: score}}

    Methods whose lowercase name is in exclude_methods are skipped.
    """
    exclude = {m.lower() for m in (exclude_methods or set())}
    out: dict[str, dict[str, float]] = {}
    for task, methods in raw.items():
        if task not in valid_tasks:
            continue
        for method, data in methods.items():
            if method.lower() in exclude:
                continue
            if method not in out:
                out[method] = {}
            out[method][task] = data["mean"]
    return out


def _compute_delta(results: dict[str, dict]) -> dict[str, dict]:
    """Compute delta (cliff_rmse - noncliff_rmse) per seed/task.

    Takes raw results {seed: {task: {metrics}}} and returns the same
    structure with a single 'delta' metric per task.
    """
    delta_results: dict[str, dict] = {}
    for seed, tasks in results.items():
        if str(seed).startswith("_"):
            continue
        delta_results[seed] = {}
        for task, metrics in tasks.items():
            cliff = metrics.get("cliff_test_rmse")
            noncliff = metrics.get("noncliff_test_rmse")
            if cliff is not None and noncliff is not None:
                delta_results[seed][task] = {"delta": cliff - noncliff}
    return delta_results


def _build_values(
    all_results: dict[str, dict],
    tasks: list[str],
    metric_per_task: dict[str, str],
) -> tuple[
    list[str],
    dict[tuple[str, str], tuple[float, float]],
    dict[str, dict[str, tuple[float, float]]],
]:
    """Aggregate seeds and extract (mean, std) for the given metric per task.

    Returns (methods, values, our_methods).
    """
    methods = list(all_results.keys())
    values: dict[tuple[str, str], tuple[float, float]] = {}
    our_methods: dict[str, dict[str, tuple[float, float]]] = {}

    for method, results in all_results.items():
        agg = aggregate_seeds(results)
        our_methods[method] = {}
        for task in tasks:
            if task not in agg:
                continue
            metric = metric_per_task.get(task)
            if metric and metric in agg[task]:
                stats = agg[task][metric]
                pair = (stats["mean"], stats.get("std", 0.0))
                values[(method, task)] = pair
                our_methods[method][task] = pair

    return methods, values, our_methods


def _compute_win_pct(
    our_methods: dict[str, dict[str, tuple[float, float]]],
    valid_tasks: list[str],
    higher_better: dict[str, bool],
    competitor_scores: dict[str, dict[str, float]] | None = None,
) -> dict[str, float]:
    """Compute win percentage for each method across tasks.

    If competitor_scores is provided, a method wins a task when it beats
    the best competitor score (leaderboard Win %).  Otherwise a method wins
    when it has the best value among our_methods (head-to-head Win %).
    """
    win_counts = {m: 0 for m in our_methods}
    for task in valid_tasks:
        hb = higher_better.get(task, True)

        if competitor_scores is not None:
            # Leaderboard Win %: beat the best competitor
            comp_vals = [
                competitor_scores[c][task]
                for c in competitor_scores
                if task in competitor_scores[c]
            ]
            if not comp_vals:
                # No competitors → all methods with a score win
                for method in our_methods:
                    if task in our_methods[method]:
                        win_counts[method] += 1
                continue
            best_comp = max(comp_vals) if hb else min(comp_vals)
            tol = abs(best_comp) * 0.001
            for method in our_methods:
                if task not in our_methods[method]:
                    continue
                mean, _ = our_methods[method][task]
                if hb and mean >= best_comp - tol:
                    win_counts[method] += 1
                elif not hb and mean <= best_comp + tol:
                    win_counts[method] += 1
        else:
            # Head-to-head Win %: best among our_methods
            scores = []
            for method in our_methods:
                if task in our_methods[method]:
                    mean, _ = our_methods[method][task]
                    scores.append((method, mean))
            if not scores:
                continue
            best_val = max(v for _, v in scores) if hb else min(v for _, v in scores)
            tol = abs(best_val) * 0.001
            for method, val in scores:
                if abs(val - best_val) <= tol:
                    win_counts[method] += 1

    n = len(valid_tasks)
    return {m: 100.0 * c / n for m, c in win_counts.items()} if n else {}


def _compute_leaderboard_rank(
    our_methods: dict[str, dict[str, tuple[float, float]]],
    competitor_scores: dict[str, dict[str, float]],
    valid_tasks: list[str],
    higher_better: dict[str, bool],
) -> dict[str, float]:
    """Compute average leaderboard rank for each of our methods."""
    method_ranks: dict[str, list[int]] = {m: [] for m in our_methods}
    for task in valid_tasks:
        scores = []
        for method in our_methods:
            if task in our_methods[method]:
                mean, _ = our_methods[method][task]
                scores.append((method, mean))
        for method in competitor_scores:
            if task in competitor_scores[method]:
                scores.append((method, competitor_scores[method][task]))
        if not scores:
            continue
        hb = higher_better.get(task, True)
        scores.sort(key=lambda x: x[1], reverse=hb)
        for rank, (method, _) in enumerate(scores, 1):
            if method in method_ranks:
                method_ranks[method].append(rank)
    return {m: float(np.mean(r)) for m, r in method_ranks.items() if r}


def _compute_mean_metric(
    our_methods: dict[str, dict[str, tuple[float, float]]],
    valid_tasks: list[str],
) -> dict[str, tuple[float, float]]:
    """Compute mean-of-means and mean-of-stds across tasks for each method."""
    result = {}
    for method in our_methods:
        means, stds = [], []
        for task in valid_tasks:
            if task in our_methods[method]:
                m, s = our_methods[method][task]
                means.append(m)
                stds.append(s)
        if means:
            result[method] = (float(np.mean(means)), float(np.mean(stds)))
    return result


def main():
    parser = argparse.ArgumentParser(description="Generate results tables")
    parser.add_argument("--results-dir", type=Path, default="results/")
    parser.add_argument("--output-dir", type=Path, default="results/tables/")
    parser.add_argument("--benchmark", nargs="+", default=["polaris", "moleculeace"])
    parser.add_argument("--leaderboard", action="store_true", default=True)
    parser.add_argument("--no-leaderboard", action="store_false", dest="leaderboard")
    parser.add_argument(
        "--no-scale", action="store_false", dest="scale",
        help="Disable MAE scaling",
    )
    parser.add_argument(
        "--no-fdr", action="store_false", dest="fdr",
        help="Disable Benjamini-Hochberg FDR correction",
    )
    parser.add_argument(
        "--ablation", action="store_true", default=False,
        help="Generate ablation table from results/ablation/",
    )
    parser.set_defaults(scale=True, fdr=True)
    # --mace-metric removed: combined table now shows both RMSE and delta win %
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load MAE scales if requested
    scales = None
    if args.scale:
        scales_path = (
            Path(__file__).resolve().parent.parent.parent
            / "monroe" / "assets" / "polaris_scales.json"
        )
        with open(scales_path) as f:
            scales = json.load(f)

    # Storage for combined table
    polaris_info: dict | None = None
    mace_info: dict | None = None
    # Store raw results per benchmark for compute_win_stats
    benchmark_all_results: dict[str, dict] = {}
    benchmark_tasks: dict[str, list[str]] = {}
    benchmark_metric_per_task: dict[str, dict[str, str]] = {}

    for benchmark in args.benchmark:
        all_results = load_benchmark_results(args.results_dir, benchmark)
        if not all_results:
            print(f"No results for {benchmark}, skipping")
            continue

        # Enrich moleculeace with cliff_delta_rmse (cliff - overall)
        if benchmark == "moleculeace":
            enrich_moleculeace_cliff_delta(all_results)

        print(f"{benchmark}: {list(all_results.keys())}")

        # ── Determine tasks and primary metric per task ──────────────
        if benchmark == "polaris":
            # All 28 tasks (polaris/ + tdcommons/)
            tasks = list(polaris_benchmarks.keys())
            metric_per_task = dict(polaris_benchmarks)
        elif benchmark == "tdc":
            tasks = [t for t in polaris_benchmarks if t.startswith("tdcommons/")]
            metric_per_task = {t: polaris_benchmarks[t] for t in tasks}
        elif benchmark == "moleculeace":
            # Discover tasks from results; primary metric is overall_test_rmse
            all_tasks: set[str] = set()
            for results in all_results.values():
                for seed_key, seed_results in results.items():
                    if str(seed_key).startswith("_"):
                        continue
                    all_tasks.update(seed_results.keys())
            tasks = sorted(all_tasks)
            metric_per_task = {t: "overall_test_rmse" for t in tasks}
        else:
            print(f"  Unknown benchmark {benchmark}, skipping")
            continue

        # ── Aggregate and build values ───────────────────────────────
        methods, values, our_methods = _build_values(
            all_results, tasks, metric_per_task,
        )
        valid_tasks = [t for t in tasks if any((m, t) in values for m in methods)]
        if not valid_tasks:
            print(f"  No valid tasks for {benchmark}")
            continue

        higher_better = {
            t: metric_direction.get(metric_per_task.get(t, ""), True)
            for t in valid_tasks
        }

        # ── Results table ────────────────────────────────────────────
        table = ResultsTable(
            methods=methods, tasks=valid_tasks, values=values,
            higher_better=higher_better, title=f"{benchmark.title()} results",
        )
        if benchmark == "moleculeace":
            out = args.output_dir / "moleculeace_overall.md"
        else:
            out = args.output_dir / f"{benchmark}_results.md"
        out.write_text(str(table))
        print(f"  Saved {out}")

        # ── Store data for combined table ────────────────────────────
        if benchmark == "polaris":
            polaris_info = {
                "our_methods": our_methods,
                "valid_tasks": valid_tasks,
                "higher_better": higher_better,
            }
        elif benchmark == "moleculeace":
            delta_metric_co = {t: "cliff_delta_rmse" for t in tasks}
            _, _, delta_methods = _build_values(all_results, tasks, delta_metric_co)
            delta_tasks = [
                t for t in tasks
                if any(t in delta_methods.get(m, {}) for m in methods)
            ]
            mace_info = {
                "our_methods": our_methods,
                "valid_tasks": valid_tasks,
                "delta_methods": delta_methods,
                "delta_tasks": delta_tasks,
            }

        # ── MoleculeACE delta table (cliff_rmse - noncliff_rmse) ─────
        if benchmark == "moleculeace":
            delta_metric = {t: "delta" for t in tasks}
            delta_data = {
                method: _compute_delta(all_results[method])
                for method in all_results
            }
            _, delta_values, _ = _build_values(delta_data, tasks, delta_metric)
            delta_valid = [t for t in tasks if any((m, t) in delta_values for m in methods)]
            if delta_valid:
                # Delta: positive means cliff is harder; lower delta is better
                delta_hb = {t: False for t in delta_valid}
                delta_table = ResultsTable(
                    methods=methods, tasks=delta_valid, values=delta_values,
                    higher_better=delta_hb,
                    title="MoleculeACE Delta (Cliff RMSE - NonCliff RMSE)",
                )
                out = args.output_dir / "moleculeace_delta.md"
                out.write_text(str(delta_table))
                print(f"  Saved {out}")

        # ── Store raw data for statistical win % ──────────────────────
        benchmark_all_results[benchmark] = all_results
        benchmark_tasks[benchmark] = tasks
        benchmark_metric_per_task[benchmark] = metric_per_task

        # ── Leaderboard table ────────────────────────────────────────
        if args.leaderboard and benchmark in ("polaris", "tdc"):
            # Deep merge: TDC first, polaris overrides per-method
            if benchmark == "polaris":
                raw_lb = _deep_merge_leaderboards(leaderboard_tdc, leaderboard_polaris)
            else:
                raw_lb = _deep_merge_leaderboards(leaderboard_tdc)

            # Exclude leaderboard entries for methods we have our own results for
            exclude = set(our_methods.keys())
            competitor_scores = _transpose_leaderboard(raw_lb, valid_tasks, exclude)

            if competitor_scores:
                lb = LeaderboardTable(
                    tasks=valid_tasks, our_methods=our_methods,
                    competitor_scores=competitor_scores,
                    metric_per_task={t: metric_per_task.get(t, "") for t in valid_tasks},
                    higher_better=higher_better,
                    title=f"{benchmark.title()} Leaderboard",
                )
                out = args.output_dir / f"{benchmark}_leaderboard.md"
                out.write_text(str(lb))
                print(f"  Saved {out}")


    # ── Combined summary table ────────────────────────────────────
    if polaris_info is not None and mace_info is not None:
        all_methods = sorted(
            set(polaris_info["our_methods"]) | set(mace_info["our_methods"])
        )

        # Compute leaderboard rank from raw leaderboard data
        raw_lb = _deep_merge_leaderboards(leaderboard_tdc, leaderboard_polaris)
        exclude = set(polaris_info["our_methods"].keys())
        competitor_scores = _transpose_leaderboard(
            raw_lb, polaris_info["valid_tasks"], exclude,
        )
        polaris_rank = _compute_leaderboard_rank(
            polaris_info["our_methods"],
            competitor_scores,
            polaris_info["valid_tasks"],
            polaris_info["higher_better"],
        )

        # Compute statistical win % (top-rank %) via Games-Howell + global FDR
        polaris_stats = compute_win_stats(
            benchmark_all_results["polaris"],
            tasks=benchmark_tasks["polaris"],
            metric_per_task=benchmark_metric_per_task["polaris"],
            higher_better=metric_direction,
            scales=scales,
            use_fdr=args.fdr,
            equal_var=True,   # Tukey HSD, as the paper reports
        )
        polaris_win = {}
        for m in all_methods:
            part = polaris_stats["participation"].get(m, 0)
            top = polaris_stats["top_rank"].get(m, 0)
            polaris_win[m] = 100.0 * top / part if part > 0 else 0.0

        mace_rmse_metric = {t: "overall_test_rmse" for t in benchmark_tasks.get("moleculeace", [])}
        mace_rmse_stats = compute_win_stats(
            benchmark_all_results["moleculeace"],
            tasks=benchmark_tasks.get("moleculeace", []),
            metric_per_task=mace_rmse_metric,
            higher_better=metric_direction,
            use_fdr=args.fdr,
            equal_var=True,   # Tukey HSD, as the paper reports
        )
        mace_rmse_win = {}
        for m in all_methods:
            part = mace_rmse_stats["participation"].get(m, 0)
            top = mace_rmse_stats["top_rank"].get(m, 0)
            mace_rmse_win[m] = 100.0 * top / part if part > 0 else 0.0

        mace_delta_metric = {t: "cliff_delta_rmse" for t in benchmark_tasks.get("moleculeace", [])}
        mace_delta_stats = compute_win_stats(
            benchmark_all_results["moleculeace"],
            tasks=benchmark_tasks.get("moleculeace", []),
            metric_per_task=mace_delta_metric,
            higher_better=metric_direction,
            use_fdr=args.fdr,
            equal_var=True,   # Tukey HSD, as the paper reports
        )
        mace_delta_win = {}
        for m in all_methods:
            part = mace_delta_stats["participation"].get(m, 0)
            top = mace_delta_stats["top_rank"].get(m, 0)
            mace_delta_win[m] = 100.0 * top / part if part > 0 else 0.0

        mace_rmse = _compute_mean_metric(
            mace_info["our_methods"],
            mace_info["valid_tasks"],
        )
        mace_delta = _compute_mean_metric(
            mace_info["delta_methods"],
            mace_info["delta_tasks"],
        )

        combined = CombinedSummaryTable(
            methods=all_methods,
            polaris_win_pct=polaris_win,
            polaris_rank=polaris_rank,
            mace_rmse_win_pct=mace_rmse_win,
            mace_delta_win_pct=mace_delta_win,
            mace_rmse=mace_rmse,
            mace_delta=mace_delta,
            title="Combined results",
        )
        out = args.output_dir / "combined_results.md"
        out.write_text(str(combined))
        print(f"Saved {out}")

    # ── Ablation table ─────────────────────────────────────────────
    if args.ablation:
        _generate_ablation_table(args)


def _generate_ablation_table(args: argparse.Namespace) -> None:
    """Generate grouped ablation table matching the paper format.

    Groups: MTL balancing, Head-to-trunk ratio, Conformer denoising,
    Stereochemistry, Decorrelation, States.
    """
    from monroe.analysis.leaderboard import leaderboard_polaris, leaderboard_tdc

    ablation_dir = args.results_dir / "ablation"
    if not ablation_dir.is_dir():
        print("No results/ablation/ directory found, skipping ablation table")
        return

    # Load ablation results
    abl_polaris = load_benchmark_results(ablation_dir, "polaris")
    abl_mace = load_benchmark_results(ablation_dir, "moleculeace")

    # Include Monroe from main results as baseline
    main_polaris = load_benchmark_results(args.results_dir, "polaris")
    main_mace = load_benchmark_results(args.results_dir, "moleculeace")
    if "monroe" in main_polaris:
        abl_polaris["monroe"] = main_polaris["monroe"]
    if "monroe" in main_mace:
        abl_mace["monroe"] = main_mace["monroe"]

    enrich_moleculeace_cliff_delta(abl_mace)

    polaris_tasks = list(polaris_benchmarks.keys())
    polaris_metric_per_task = dict(polaris_benchmarks)

    mace_task_set: set[str] = set()
    for results in abl_mace.values():
        for seed, seed_data in results.items():
            if str(seed).startswith("_"):
                continue
            mace_task_set.update(seed_data.keys())
    mace_tasks = sorted(mace_task_set)

    # Compute leaderboard for win %
    raw_lb = _deep_merge_leaderboards(leaderboard_tdc, leaderboard_polaris)
    higher_better = {
        t: metric_direction.get(polaris_benchmarks.get(t, ""), True)
        for t in polaris_tasks
    }

    # ── Per-config leaderboard rank ───────────────────────────────
    # Competitor pool: non-Monroe main methods + external leaderboard
    non_monroe_main = {m: main_polaris[m] for m in main_polaris if m != "monroe"}
    _, _, non_monroe_our = _build_values(non_monroe_main, polaris_tasks, polaris_metric_per_task)
    non_monroe_scores = {
        m: {t: mean for t, (mean, _) in td.items()}
        for m, td in non_monroe_our.items()
    }
    external_scores = _transpose_leaderboard(
        raw_lb, polaris_tasks, exclude_methods=set(main_polaris.keys()),
    )
    competitor_pool = {**external_scores, **non_monroe_scores}

    # Load MAE scales
    scales = None
    if args.scale:
        scales_path = Path(__file__).resolve().parent.parent.parent / "monroe" / "assets" / "polaris_scales.json"
        with open(scales_path) as f:
            scales = json.load(f)

    def _compute_config_metrics(configs_in_group: list[str]) -> dict[str, dict]:
        """Compute Polaris win %, MACE RMSE, MACE Δ for a group of configs."""
        # Build combined results for the group (for pairwise win %)
        group_polaris = {c: abl_polaris[c] for c in configs_in_group if c in abl_polaris}

        # Polaris win % (within group)
        pol_stats = compute_win_stats(
            group_polaris, tasks=polaris_tasks,
            metric_per_task=polaris_metric_per_task,
            higher_better=metric_direction,
            scales=scales, use_fdr=args.fdr,
        )
        pol_win = {}
        for m in configs_in_group:
            part = pol_stats["participation"].get(m, 0)
            top = pol_stats["top_rank"].get(m, 0)
            pol_win[m] = 100.0 * top / part if part > 0 else 0.0

        # MACE RMSE and Δ
        mace_rmse_metric = {t: "overall_test_rmse" for t in mace_tasks}
        mace_delta_metric = {t: "cliff_delta_rmse" for t in mace_tasks}

        result = {}
        for config in configs_in_group:
            entry = {"pol_win": pol_win.get(config, 0.0)}

            if config in abl_mace:
                _, _, rmse_our = _build_values({config: abl_mace[config]}, mace_tasks, mace_rmse_metric)
                rmse_mean = _compute_mean_metric(rmse_our, mace_tasks)
                entry["mace_rmse"] = rmse_mean.get(config, (float("nan"), 0.0))

                _, _, delta_our = _build_values({config: abl_mace[config]}, mace_tasks, mace_delta_metric)
                delta_mean = _compute_mean_metric(delta_our, mace_tasks)
                entry["mace_delta"] = delta_mean.get(config, (float("nan"), 0.0))
            else:
                entry["mace_rmse"] = (float("nan"), 0.0)
                entry["mace_delta"] = (float("nan"), 0.0)

            result[config] = entry
        return result

    # Define groups (matching paper structure)
    groups = [
        ("Multi-task balancing", [
            ("EW", "EW"),
            ("UW", "UW"),
            ("DWA", "DWA"),
            ("RLW", "RLW"),
            ("**STCH**", "monroe"),
        ]),
        ("Head-to-trunk ratio", [
            ("Linear probing", "linear"),
            ("**Perceptron**", "monroe"),
            ("MLP", "MLP"),
        ]),
        ("Conformer denoising", [
            ("×", "no_denoise"),
            ("**✓**", "monroe"),
        ]),
        ("Stereochemistry augmentation", [
            ("×", "no_stereo"),
            ("**✓**", "monroe"),
        ]),
        ("Decorrelation loss", [
            ("×", "no_decorr"),
            ("**✓**", "monroe"),
        ]),
        ("Ground-state only (S0)", [
            ("S0 only", "S0only"),
            ("**All states**", "monroe"),
        ]),
    ]

    # Compute per-config leaderboard rank (each config ranked independently)
    all_cfg_keys = {k for _, entries in groups for _, k in entries}
    config_ranks: dict[str, float | None] = {}
    for cfg in all_cfg_keys:
        if cfg not in abl_polaris:
            config_ranks[cfg] = None
            continue
        _, _, cfg_our = _build_values(
            {cfg: abl_polaris[cfg]}, polaris_tasks, polaris_metric_per_task,
        )
        if cfg not in cfg_our:
            config_ranks[cfg] = None
            continue
        ranks = _compute_leaderboard_rank(cfg_our, competitor_pool, polaris_tasks, higher_better)
        config_ranks[cfg] = ranks.get(cfg)

    # Build markdown
    lines = ["### Ablation Study", ""]
    lines.append("| Variant | Polaris Win % | Polaris Rank ↓ | MACE RMSE ↓ | MACE Δ ↓ |")
    lines.append("|---|---|---|---|---|")

    for group_name, entries in groups:
        lines.append(f"| *{group_name}* | | | | |")
        config_keys = [key for _, key in entries]
        metrics = _compute_config_metrics(config_keys)

        # Find best in group for bolding
        pol_wins = [metrics[k]["pol_win"] for k in config_keys if k in metrics]
        rmse_vals = [metrics[k]["mace_rmse"][0] for k in config_keys
                     if k in metrics and not np.isnan(metrics[k]["mace_rmse"][0])]
        delta_vals = [metrics[k]["mace_delta"][0] for k in config_keys
                      if k in metrics and not np.isnan(metrics[k]["mace_delta"][0])]

        best_pw = max(pol_wins) if pol_wins else None
        rank_vals = [config_ranks[k] for _, k in entries if config_ranks.get(k) is not None]
        best_rank = min(rank_vals) if rank_vals else None
        best_rmse = min(rmse_vals) if rmse_vals else None
        best_delta = min(delta_vals) if delta_vals else None

        for display_name, config_key in entries:
            if config_key not in metrics:
                lines.append(f"| {display_name} | -- | -- | -- | -- |")
                continue
            m = metrics[config_key]

            pw = m["pol_win"]
            pw_s = f"{pw:.1f}"
            if best_pw is not None and abs(pw - best_pw) < 0.05:
                pw_s = f"**{pw_s}**"

            rank = config_ranks.get(config_key)
            rank_s = f"{rank:.2f}" if rank is not None else "--"
            if best_rank is not None and rank is not None and abs(rank - best_rank) < 0.005:
                rank_s = f"**{rank_s}**"

            rmse_mean, rmse_std = m["mace_rmse"]
            if not np.isnan(rmse_mean):
                rmse_s = f"{rmse_mean:.3f} ± {rmse_std:.3f}"
                if best_rmse is not None and abs(rmse_mean - best_rmse) < 0.0005:
                    rmse_s = f"**{rmse_s}**"
            else:
                rmse_s = "--"

            delta_mean, delta_std = m["mace_delta"]
            if not np.isnan(delta_mean):
                delta_s = f"{delta_mean:.3f} ± {delta_std:.3f}"
                if best_delta is not None and abs(delta_mean - best_delta) < 0.0005:
                    delta_s = f"**{delta_s}**"
            else:
                delta_s = "--"

            lines.append(f"| {display_name} | {pw_s} | {rank_s} | {rmse_s} | {delta_s} |")

    out = args.output_dir / "ablation_results.md"
    out.write_text("\n".join(lines))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
