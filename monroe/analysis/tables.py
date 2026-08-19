"""Table generation utilities for results display.

Provides dataclasses for generating readable Markdown tables for:
- Results comparison across methods
- Ablation studies
- Leaderboard comparisons
"""

from dataclasses import dataclass, field

import numpy as np

from monroe.analysis.style import format_method_name, sort_methods


@dataclass
class ResultsTable:
    """Table for comparing method results across tasks.

    Displays mean +/- std for each method/task combination, with
    optional highlighting of best results.
    """

    methods: list[str]
    tasks: list[str]
    values: dict[tuple[str, str], tuple[float, float]]  # (method, task) -> (mean, std)
    higher_better: dict[str, bool] = field(default_factory=dict)
    title: str | None = None
    precision: int = 3

    def _get_best_per_task(self) -> dict[str, list[str]]:
        """Find best method(s) per task (allowing ties within tolerance)."""
        best = {}
        for task in self.tasks:
            task_vals = []
            for method in self.methods:
                if (method, task) in self.values:
                    mean, _ = self.values[(method, task)]
                    task_vals.append((method, mean))

            if not task_vals:
                best[task] = []
                continue

            hb = self.higher_better.get(task, True)
            if hb:
                best_val = max(v for _, v in task_vals)
            else:
                best_val = min(v for _, v in task_vals)

            tol = abs(best_val) * 0.001
            best[task] = [m for m, v in task_vals if abs(v - best_val) <= tol]

        return best

    def to_markdown(self, highlight_best: bool = True) -> str:
        """Generate Markdown table."""
        sorted_methods = sort_methods(self.methods)
        best_per_task = self._get_best_per_task() if highlight_best else {}

        lines = []
        if self.title:
            lines.append(f"### {self.title}")
            lines.append("")

        header = "| Task |"
        for method in sorted_methods:
            header += f" {format_method_name(method)} |"

        sep = "|---|"
        for _ in sorted_methods:
            sep += "---|"

        lines.extend([header, sep])

        for task in self.tasks:
            row = f"| {task} |"
            for method in sorted_methods:
                if (method, task) in self.values:
                    mean, std = self.values[(method, task)]
                    val_str = f"{mean:.{self.precision}f}"
                    if std > 0:
                        val_str += f" +/- {std:.{self.precision}f}"
                    if highlight_best and method in best_per_task.get(task, []):
                        val_str = f"**{val_str}**"
                    row += f" {val_str} |"
                else:
                    row += " -- |"
            lines.append(row)

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_markdown()


@dataclass
class AblationTable:
    """Table for ablation study results.

    Shows how different configurations affect performance.
    """

    configs: list[str]
    metrics: list[str]
    values: dict[tuple[str, str], tuple[float, float]]
    title: str | None = None
    baseline_config: str | None = None
    precision: int = 3

    def to_markdown(self, show_delta: bool = True) -> str:
        """Generate Markdown table."""
        lines = []
        if self.title:
            lines.append(f"### {self.title}")
            lines.append("")

        header = "| Configuration |"
        for metric in self.metrics:
            header += f" {metric} |"

        sep = "|---|"
        for _ in self.metrics:
            sep += "---|"

        lines.extend([header, sep])

        baseline_vals = {}
        if show_delta and self.baseline_config:
            for metric in self.metrics:
                if (self.baseline_config, metric) in self.values:
                    baseline_vals[metric] = self.values[(self.baseline_config, metric)][0]

        for config in self.configs:
            row = f"| {config} |"
            for metric in self.metrics:
                if (config, metric) in self.values:
                    mean, std = self.values[(config, metric)]
                    val_str = f"{mean:.{self.precision}f}"
                    if std > 0:
                        val_str += f" +/- {std:.{self.precision}f}"
                    if metric in baseline_vals and config != self.baseline_config:
                        delta = mean - baseline_vals[metric]
                        sign = "+" if delta >= 0 else ""
                        val_str += f" ({sign}{delta:.{self.precision}f})"
                    row += f" {val_str} |"
                else:
                    row += " -- |"
            lines.append(row)

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_markdown()


@dataclass
class LeaderboardTable:
    """Table for leaderboard comparison with competitors."""

    tasks: list[str]
    our_methods: dict[str, dict[str, tuple[float, float]]]
    competitor_scores: dict[str, dict[str, float]]
    metric_per_task: dict[str, str] = field(default_factory=dict)
    higher_better: dict[str, bool] = field(default_factory=dict)
    title: str | None = None
    precision: int = 3

    def _get_all_methods(self) -> list[str]:
        """Get all methods (ours + competitors), sorted with ours first."""
        our = sort_methods(list(self.our_methods.keys()))
        competitors = sorted(self.competitor_scores.keys())
        return our + competitors

    def _get_rank(self, task: str) -> dict[str, int]:
        """Get rank of each method for a task."""
        scores = []
        for method in self.our_methods:
            if task in self.our_methods[method]:
                mean, _ = self.our_methods[method][task]
                scores.append((method, mean))
        for method in self.competitor_scores:
            if task in self.competitor_scores[method]:
                scores.append((method, self.competitor_scores[method][task]))

        if not scores:
            return {}

        hb = self.higher_better.get(task, True)
        scores.sort(key=lambda x: x[1], reverse=hb)

        return {method: rank + 1 for rank, (method, _) in enumerate(scores)}

    def to_markdown(self, show_rank: bool = True) -> str:
        """Generate Markdown table."""
        lines = []
        if self.title:
            lines.append(f"### {self.title}")
            lines.append("")

        header = "| Method |"
        for task in self.tasks:
            metric = self.metric_per_task.get(task, "")
            if metric:
                header += f" {task} ({metric}) |"
            else:
                header += f" {task} |"
        if show_rank:
            header += " Avg Rank |"

        sep = "|---|"
        for _ in self.tasks:
            sep += "---|"
        if show_rank:
            sep += "---|"

        lines.extend([header, sep])

        # Our methods first
        for method in sort_methods(list(self.our_methods.keys())):
            row = f"| {format_method_name(method)} |"
            ranks = []
            for task in self.tasks:
                if task in self.our_methods[method]:
                    mean, std = self.our_methods[method][task]
                    val_str = f"{mean:.{self.precision}f}"
                    if std > 0:
                        val_str += f" +/- {std:.{self.precision}f}"
                    row += f" {val_str} |"
                    task_ranks = self._get_rank(task)
                    if method in task_ranks:
                        ranks.append(task_ranks[method])
                else:
                    row += " -- |"

            if show_rank and ranks:
                avg_rank = np.mean(ranks)
                row += f" {avg_rank:.1f} |"
            elif show_rank:
                row += " -- |"
            lines.append(row)

        lines.append("|---|" + "---|" * len(self.tasks) + ("---|" if show_rank else ""))

        # Competitor methods
        for method in sorted(self.competitor_scores.keys()):
            row = f"| {method} |"
            ranks = []
            for task in self.tasks:
                if task in self.competitor_scores[method]:
                    score = self.competitor_scores[method][task]
                    row += f" {score:.{self.precision}f} |"
                    task_ranks = self._get_rank(task)
                    if method in task_ranks:
                        ranks.append(task_ranks[method])
                else:
                    row += " -- |"

            if show_rank and ranks:
                avg_rank = np.mean(ranks)
                row += f" {avg_rank:.1f} |"
            elif show_rank:
                row += " -- |"
            lines.append(row)

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_markdown()


@dataclass
class CombinedSummaryTable:
    """Combined summary comparing methods across Polaris and MoleculeACE.

    Columns: Model | Polaris Win % | Polaris Rank | MACE RMSE Win % | MACE Δ Win % | RMSE | Δ
    """

    methods: list[str]
    polaris_win_pct: dict[str, float]
    polaris_rank: dict[str, float]
    mace_rmse_win_pct: dict[str, float]
    mace_delta_win_pct: dict[str, float]
    mace_rmse: dict[str, tuple[float, float]]  # (mean_of_means, mean_of_stds)
    mace_delta: dict[str, tuple[float, float]]  # (mean_of_means, mean_of_stds)
    title: str | None = None
    precision: int = 3

    def to_markdown(self) -> str:
        """Generate Markdown table."""
        sorted_methods = sort_methods(self.methods)
        lines = []
        if self.title:
            lines.append(f"### {self.title}")
            lines.append("")

        lines.append(
            "| Model | Polaris Win % | Polaris Rank ↓ "
            "| MACE RMSE Win % | MACE Δ Win % | RMSE ↓ | Δ ↓ |"
        )
        lines.append("|---|---|---|---|---|---|---|")

        pw_vals = list(self.polaris_win_pct.values())
        pr_vals = list(self.polaris_rank.values())
        mrw_vals = list(self.mace_rmse_win_pct.values())
        mdw_vals = list(self.mace_delta_win_pct.values())
        mr_vals = [v[0] for v in self.mace_rmse.values()]
        md_vals = [v[0] for v in self.mace_delta.values()]

        best_pw = max(pw_vals) if pw_vals else None
        best_pr = min(pr_vals) if pr_vals else None
        best_mrw = max(mrw_vals) if mrw_vals else None
        best_mdw = max(mdw_vals) if mdw_vals else None
        best_mr = min(mr_vals) if mr_vals else None
        best_md = min(md_vals) if md_vals else None

        def _bold(s: str, val: float | None, best: float | None, tol: float) -> str:
            if val is not None and best is not None and abs(val - best) <= tol:
                return f"**{s}**"
            return s

        p = self.precision
        for method in sorted_methods:
            display = format_method_name(method)

            pw = self.polaris_win_pct.get(method)
            pw_s = _bold(f"{pw:.1f}", pw, best_pw, 0.05) if pw is not None else "--"

            pr = self.polaris_rank.get(method)
            pr_s = _bold(f"{pr:.2f}", pr, best_pr, 0.005) if pr is not None else "--"

            mrw = self.mace_rmse_win_pct.get(method)
            mrw_s = _bold(f"{mrw:.1f}", mrw, best_mrw, 0.05) if mrw is not None else "--"

            mdw = self.mace_delta_win_pct.get(method)
            mdw_s = _bold(f"{mdw:.1f}", mdw, best_mdw, 0.05) if mdw is not None else "--"

            if method in self.mace_rmse:
                mean, std = self.mace_rmse[method]
                rs = f"{mean:.{p}f} +/- {std:.{p}f}"
                rs = _bold(rs, mean, best_mr, 10 ** (-p - 1))
            else:
                rs = "--"

            if method in self.mace_delta:
                mean, std = self.mace_delta[method]
                ds = f"{mean:.{p}f} +/- {std:.{p}f}"
                ds = _bold(ds, mean, best_md, 10 ** (-p - 1))
            else:
                ds = "--"

            lines.append(f"| {display} | {pw_s} | {pr_s} | {mrw_s} | {mdw_s} | {rs} | {ds} |")

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_markdown()


def build_results_table(
    method_results: dict[str, dict[str, dict[str, dict]]],
    tasks: list[str],
    metric_per_task: dict[str, str],
    higher_better: dict[str, bool] | None = None,
    title: str | None = None,
) -> ResultsTable:
    """Build a ResultsTable from aggregated method results.

    Args:
        method_results: Dict mapping method -> task -> metric -> {mean, std}
        tasks: List of tasks to include
        metric_per_task: Dict mapping task -> primary metric name
        higher_better: Dict mapping task -> whether higher is better
        title: Optional table title
    """
    methods = list(method_results.keys())
    values = {}

    for method, task_results in method_results.items():
        for task in tasks:
            if task not in task_results:
                continue
            metric = metric_per_task.get(task)
            if metric is None:
                continue
            if metric in task_results[task]:
                stats = task_results[task][metric]
                values[(method, task)] = (stats["mean"], stats.get("std", 0.0))

    return ResultsTable(
        methods=methods,
        tasks=tasks,
        values=values,
        higher_better=higher_better or {},
        title=title,
    )
