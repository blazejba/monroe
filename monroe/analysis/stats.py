"""Statistical testing utilities for method comparison.

Provides:
- Benjamini-Hochberg FDR correction
- Games-Howell pairwise comparison (unequal variance Tukey HSD)
- Welch's t-test for two-sample comparison
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import ttest_ind, tukey_hsd


def benjamini_hochberg_fdr(
    p_values: Sequence[float],
    alpha: float = 0.05,
) -> list[bool]:
    """Apply Benjamini-Hochberg FDR correction to p-values.

    Args:
        p_values: List of p-values from multiple comparisons
        alpha: Significance level (default 0.05)

    Returns:
        List of booleans indicating which hypotheses are rejected (significant)
    """
    n = len(p_values)
    if n == 0:
        return []

    p_array = np.array(p_values)
    sorted_indices = np.argsort(p_array)
    sorted_p = p_array[sorted_indices]

    # BH thresholds: (rank / n) * alpha
    thresholds = (np.arange(n) + 1) / n * alpha
    rejected = sorted_p <= thresholds

    # Find largest k where p_(k) <= threshold_(k)
    is_significant = [False] * n
    if np.any(rejected):
        max_k = np.where(rejected)[0].max()
        for idx in range(max_k + 1):
            is_significant[sorted_indices[idx]] = True

    return is_significant


def bonferroni_correction(
    p_values: Sequence[float],
    alpha: float = 0.05,
) -> list[bool]:
    """Apply Bonferroni correction to p-values.

    Args:
        p_values: List of p-values from multiple comparisons
        alpha: Significance level (default 0.05)

    Returns:
        List of booleans indicating which hypotheses are rejected (significant)
    """
    n = len(p_values)
    if n == 0:
        return []
    return [float(p) * n < alpha for p in p_values]


@dataclass
class GamesHowellResult:
    """Result of Games-Howell pairwise comparison."""

    groups: list[str]
    means: list[float]
    p_values: np.ndarray  # (n_groups, n_groups) matrix
    significant: np.ndarray  # (n_groups, n_groups) boolean matrix after FDR

    def get_winners(self, higher_better: bool = True) -> list[int]:
        """Get indices of groups statistically indistinguishable from best.

        Args:
            higher_better: If True, higher mean is better

        Returns:
            List of group indices that are "winners" (tied for best)
        """
        best_idx = int(np.argmax(self.means) if higher_better else np.argmin(self.means))
        winners = [best_idx]

        for i in range(len(self.groups)):
            if i == best_idx:
                continue
            # If not significantly different from best, it's also a winner
            if not self.significant[best_idx, i]:
                winners.append(i)

        return sorted(winners)


def games_howell_test(
    *groups: Sequence[float],
    group_names: list[str] | None = None,
    alpha: float = 0.05,
    use_fdr: bool = True,
    equal_var: bool = False,
) -> GamesHowellResult:
    """Perform Games-Howell pairwise comparison (Tukey HSD with unequal variance).

    Args:
        *groups: Variable number of groups, each a sequence of values
        group_names: Optional names for each group
        alpha: Significance level
        use_fdr: Whether to apply Benjamini-Hochberg FDR correction

    Returns:
        GamesHowellResult with p-values and significance matrix
    """
    n_groups = len(groups)
    if group_names is None:
        group_names = [f"Group_{i}" for i in range(n_groups)]

    # equal_var=False: Games-Howell (unequal variance), True: Tukey HSD
    result = tukey_hsd(*groups, equal_var=equal_var)
    p_values = result.pvalue
    means = [np.mean(g) for g in groups]

    # Determine significance
    if use_fdr:
        # Collect all unique pairwise p-values
        all_p = []
        indices = []
        for i in range(n_groups):
            for j in range(i + 1, n_groups):
                all_p.append(p_values[i, j])
                indices.append((i, j))

        is_sig = benjamini_hochberg_fdr(all_p, alpha)

        # Build significance matrix
        significant = np.zeros((n_groups, n_groups), dtype=bool)
        for idx, (i, j) in enumerate(indices):
            significant[i, j] = is_sig[idx]
            significant[j, i] = is_sig[idx]
    else:
        significant = p_values < alpha

    return GamesHowellResult(
        groups=group_names,
        means=means,
        p_values=p_values,
        significant=significant,
    )


def compute_win_stats(
    all_results: dict[str, dict],
    tasks: list[str],
    metric_per_task: dict[str, str],
    higher_better: dict[str, bool],
    scales: dict[str, float] | None = None,
    use_fdr: bool = False,
    correction: str | None = None,
    equal_var: bool = False,
) -> dict:
    """Compute win rate statistics using pairwise comparison with optional correction.

    A method is "top rank" on a task if it is either the best or statistically
    indistinguishable from the best.  Pairwise wins are counted when the
    difference is statistically significant; otherwise a tie is recorded for
    both directions.

    Args:
        all_results: Dict mapping method_name -> {seed -> {task -> {metric -> value}}}
        tasks: List of task names to evaluate
        metric_per_task: Dict mapping task -> primary metric name
        higher_better: Dict mapping metric name -> whether higher is better
        scales: Optional dict mapping task -> MAE scale for 1 - mae/scale transform
        use_fdr: Whether to apply global BH FDR (ignored if correction is set)
        correction: "bh" (Benjamini-Hochberg), "bonferroni", or "none".
            Overrides use_fdr when set. Default None falls back to use_fdr.
        equal_var: False = Games-Howell (unequal variance), True = Tukey HSD

    Returns:
        Dict with keys: participation, top_rank, pairwise_wins, pairwise_ties, n_tasks
    """
    from monroe.analysis.loader import aggregate_seeds

    if correction is None:
        correction = "bh" if use_fdr else "none"

    methods = list(all_results.keys())
    participation = {m: 0 for m in methods}
    top_rank = {m: 0 for m in methods}
    pairwise_wins: dict[tuple[str, str], int] = {}
    pairwise_ties: dict[tuple[str, str], int] = {}

    valid_tasks: list[str] = []
    # Collect per-task results (always raw p-values, no per-task FDR)
    task_results: list[dict] = []

    for task in tasks:
        metric = metric_per_task.get(task)
        if metric is None:
            continue

        # Collect per-seed values for this task
        task_arrays: dict[str, list[float]] = {}
        for method, results in all_results.items():
            agg = aggregate_seeds(results, tasks=[task])
            if task in agg and metric in agg[task]:
                values = agg[task][metric]["values"]
                if len(values) > 1:
                    task_arrays[method] = values
                    participation[method] += 1

        if len(task_arrays) < 2:
            continue

        valid_tasks.append(task)
        hb = higher_better.get(metric, True)

        # Apply MAE scaling: 1 - (mae / scale) -> higher is better
        if scales and metric in ("mean_absolute_error", "mae"):
            scale = scales.get(task)
            if scale is not None:
                task_arrays = {
                    m: [1.0 - (x / scale) for x in vals]
                    for m, vals in task_arrays.items()
                }
                hb = True

        # Run pairwise test (always raw p-values, correction applied globally below)
        task_methods = list(task_arrays.keys())
        try:
            result = games_howell_test(
                *[task_arrays[m] for m in task_methods],
                group_names=task_methods,
                use_fdr=False,
                equal_var=equal_var,
            )
        except ValueError:
            continue

        task_results.append({
            "task_methods": task_methods,
            "result": result,
            "hb": hb,
        })

    # Apply global correction across all tasks
    if correction in ("bh", "bonferroni") and task_results:
        # Collect all unique pairwise p-values across all tasks
        all_p_values: list[float] = []
        p_value_indices: list[tuple[int, int, int]] = []  # (task_idx, i, j)
        for t_idx, tr in enumerate(task_results):
            n = len(tr["task_methods"])
            for i in range(n):
                for j in range(i + 1, n):
                    all_p_values.append(tr["result"].p_values[i, j])
                    p_value_indices.append((t_idx, i, j))

        if correction == "bh":
            is_sig = benjamini_hochberg_fdr(all_p_values)
        else:
            is_sig = bonferroni_correction(all_p_values)

        # Rebuild significance matrices from global FDR results
        for t_idx, tr in enumerate(task_results):
            n = len(tr["task_methods"])
            sig = np.zeros((n, n), dtype=bool)
            tr["result"].significant = sig

        for k, (t_idx, i, j) in enumerate(p_value_indices):
            task_results[t_idx]["result"].significant[i, j] = is_sig[k]
            task_results[t_idx]["result"].significant[j, i] = is_sig[k]

    # Compute stats from (possibly FDR-corrected) significance
    for tr in task_results:
        task_methods = tr["task_methods"]
        result = tr["result"]
        hb = tr["hb"]

        # Top rank: methods statistically indistinguishable from the best
        for idx in result.get_winners(higher_better=hb):
            top_rank[task_methods[idx]] = top_rank.get(task_methods[idx], 0) + 1

        # Pairwise: sort by mean (best first), i < j means i is better
        sorted_indices = sorted(
            range(len(task_methods)),
            key=lambda i: result.means[i],
            reverse=hb,
        )
        for ii in range(len(sorted_indices)):
            for jj in range(ii + 1, len(sorted_indices)):
                i, j = sorted_indices[ii], sorted_indices[jj]
                m_i, m_j = task_methods[i], task_methods[j]
                if result.significant[i, j]:
                    pairwise_wins[(m_i, m_j)] = pairwise_wins.get((m_i, m_j), 0) + 1
                else:
                    pairwise_ties[(m_i, m_j)] = pairwise_ties.get((m_i, m_j), 0) + 1
                    pairwise_ties[(m_j, m_i)] = pairwise_ties.get((m_j, m_i), 0) + 1

    return {
        "participation": participation,
        "top_rank": top_rank,
        "pairwise_wins": pairwise_wins,
        "pairwise_ties": pairwise_ties,
        "n_tasks": len(valid_tasks),
    }


def welch_ttest(
    a: Sequence[float],
    b: Sequence[float],
) -> tuple[float, float]:
    """Perform Welch's t-test (two-sample t-test with unequal variance).

    Args:
        a: First sample
        b: Second sample

    Returns:
        Tuple of (t-statistic, p-value)
    """
    result = ttest_ind(a, b, equal_var=False)
    return float(result.statistic), float(result.pvalue)


@dataclass
class PairwiseComparison:
    """Result of pairwise method comparison across multiple tasks."""

    method1: str
    method2: str
    method1_wins: int
    method2_wins: int
    ties: int
    total_tasks: int
    task_results: dict  # task -> {winner, p_value, mean1, mean2, significant}

    def win_rate(self, method: str) -> float:
        """Get win rate for a method."""
        if method == self.method1:
            return self.method1_wins / self.total_tasks
        elif method == self.method2:
            return self.method2_wins / self.total_tasks
        raise ValueError(f"Unknown method: {method}")

    def summary(self) -> str:
        """Return a summary string."""
        lines = [
            f"Comparison: {self.method1} vs {self.method2}",
            f"  {self.method1} wins: {self.method1_wins}",
            f"  {self.method2} wins: {self.method2_wins}",
            f"  Ties: {self.ties}",
            f"  Total tasks: {self.total_tasks}",
        ]
        return "\n".join(lines)


def pairwise_comparison(
    data1: dict[str, list[float]],
    data2: dict[str, list[float]],
    method1_name: str,
    method2_name: str,
    higher_better: dict[str, bool] | bool = True,
    alpha: float = 0.05,
    use_fdr: bool = True,
) -> PairwiseComparison:
    """Compare two methods across multiple tasks using Welch's t-test.

    Args:
        data1: Dict mapping task -> list of values for method 1
        data2: Dict mapping task -> list of values for method 2
        method1_name: Display name for method 1
        method2_name: Display name for method 2
        higher_better: Whether higher values are better (bool or dict per task)
        alpha: Significance level
        use_fdr: Whether to apply FDR correction

    Returns:
        PairwiseComparison result
    """
    tasks = sorted(set(data1.keys()) & set(data2.keys()))

    # First pass: collect p-values
    task_info = []
    for task in tasks:
        vals1 = data1[task]
        vals2 = data2[task]

        if len(vals1) < 2 or len(vals2) < 2:
            continue

        mean1 = np.mean(vals1)
        mean2 = np.mean(vals2)
        _, p_val = welch_ttest(vals1, vals2)

        hb = higher_better if isinstance(higher_better, bool) else higher_better.get(task, True)

        task_info.append({
            "task": task,
            "mean1": mean1,
            "mean2": mean2,
            "p_value": p_val,
            "higher_better": hb,
        })

    # Apply FDR if requested
    p_values = [t["p_value"] for t in task_info]
    if use_fdr:
        is_sig = benjamini_hochberg_fdr(p_values, alpha)
    else:
        is_sig = [p < alpha for p in p_values]

    # Second pass: determine winners
    method1_wins = 0
    method2_wins = 0
    ties = 0
    task_results = {}

    for i, info in enumerate(task_info):
        task = info["task"]
        mean1 = info["mean1"]
        mean2 = info["mean2"]
        hb = info["higher_better"]
        sig = is_sig[i]

        if sig:
            if hb:
                winner = method1_name if mean1 > mean2 else method2_name
            else:
                winner = method1_name if mean1 < mean2 else method2_name

            if winner == method1_name:
                method1_wins += 1
            else:
                method2_wins += 1
        else:
            winner = "tie"
            ties += 1

        task_results[task] = {
            "winner": winner,
            "p_value": info["p_value"],
            "mean1": mean1,
            "mean2": mean2,
            "significant": sig,
        }

    return PairwiseComparison(
        method1=method1_name,
        method2=method2_name,
        method1_wins=method1_wins,
        method2_wins=method2_wins,
        ties=ties,
        total_tasks=len(task_info),
        task_results=task_results,
    )
