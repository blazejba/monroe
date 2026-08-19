"""Plotting utilities for method comparison and results visualization."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from monroe.analysis.style import (
    FONTSIZE,
    FONTSIZE_SMALL,
    MOLECULEACE_TASK_NAMES,
    ORANGE_GRADIENT,
    PURPLE_GRADIENT,
    format_method_name,
    get_method_color,
    sort_methods,
)

# Enable mathtext rendering for subscripts/superscripts
matplotlib.rcParams["mathtext.fontset"] = "dejavusans"
matplotlib.rcParams["mathtext.default"] = "regular"


def plot_moleculeace_cliff_delta(
    methods: list[str],
    tasks: list[str],
    data: dict[str, dict[str, dict[str, float]]],
    figsize_width: float = 4.5,
) -> plt.Figure:
    """Plot MoleculeACE cliff delta and overall RMSE as horizontal error bars.

    Two side-by-side panels:
      Left  -- delta RMSE (cliff_test_rmse - overall_test_rmse)
      Right -- overall RMSE

    Args:
        methods: Ordered list of method names (use sort_methods)
        tasks: Ordered list of task names (sorted by best overall RMSE)
        data: data[method][task] = {delta_mean, delta_std, rmse_mean, rmse_std}
        figsize_width: Figure width (height auto-scaled by number of tasks)

    Returns:
        Figure object
    """
    fs = FONTSIZE_SMALL  # base font size for this dense plot
    n_tasks = len(tasks)
    fig_height = max(6, n_tasks * 0.18)

    fig, (ax_delta, ax_rmse) = plt.subplots(
        1, 2,
        figsize=(figsize_width, fig_height),
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.08},
    )

    positions = np.arange(n_tasks)
    legend_handles = []

    for i, method in enumerate(methods):
        method_data = data.get(method, {})

        d_means, d_stds = [], []
        r_means, r_stds = [], []
        pos = []

        for j, task in enumerate(tasks):
            if task in method_data:
                td = method_data[task]
                d_means.append(td["delta_mean"])
                d_stds.append(td["delta_std"])
                r_means.append(td["rmse_mean"])
                r_stds.append(td["rmse_std"])
                pos.append(positions[j])

        if not pos:
            continue

        color = get_method_color(method)
        zorder = 20 - i

        ax_delta.errorbar(
            d_means, pos, xerr=d_stds,
            fmt="o", color=color, ecolor=color, elinewidth=1.6,
            capsize=0, markersize=5, alpha=0.8, zorder=zorder,
        )
        ax_rmse.errorbar(
            r_means, pos, xerr=r_stds,
            fmt="o", color=color, ecolor=color, elinewidth=1.6,
            capsize=0, markersize=5, alpha=0.8, zorder=zorder,
        )

        display_name = format_method_name(method)
        legend_handles.append(
            plt.Line2D(
                [0], [0], color=color, marker="o", linestyle="-",
                markersize=4, linewidth=1.5, label=display_name,
            )
        )

    # Zebra striping
    for j in range(n_tasks):
        if j % 2 == 1:
            for ax in [ax_delta, ax_rmse]:
                ax.axhspan(j - 0.5, j + 0.5, color="#F5F5F5", zorder=0, alpha=0.5)

    # Delta plot styling
    ax_delta.axvline(x=0, color="#333333", linestyle="--", linewidth=1.2, alpha=0.6, zorder=1)
    xlim_d = ax_delta.get_xlim()
    ax_delta.axvspan(xlim_d[0], 0, color="#6666FF", alpha=0.06, zorder=0)
    ax_delta.set_xlim(xlim_d)

    display_tasks = [MOLECULEACE_TASK_NAMES.get(t, t) for t in tasks]
    ax_delta.set_yticks(positions)
    ax_delta.set_yticklabels(display_tasks, fontsize=fs - 4.5, fontfamily="monospace")
    ax_delta.set_xlabel(r"$\Delta$ RMSE", fontsize=fs - 3)
    # RMSE plot styling
    ax_rmse.set_yticks(positions)
    ax_rmse.set_yticklabels([])
    ax_rmse.set_xlabel("RMSE", fontsize=fs - 3)
    ax_rmse.legend(
        handles=legend_handles, loc="lower left", fontsize=fs - 8.5,
        framealpha=0.9, edgecolor="gray", fancybox=True, ncol=1,
    )

    # Common styling
    for ax in [ax_delta, ax_rmse]:
        ax.set_facecolor("white")
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.tick_params(axis="both", labelsize=fs - 4.5)
        ax.grid(axis="x", linestyle="--", alpha=0.3, color="#999999")
        ax.invert_yaxis()

    plt.tight_layout()
    return fig


def plot_combined_summary(
    polaris_stats: dict,
    moleculeace_stats: dict,
    output_path: str | None = None,
    figsize: tuple[float, float] = (24, 6),
) -> plt.Figure:
    """Plot combined summary for both Polaris and MoleculeACE benchmarks.

    Creates a horizontal figure with:
    Bar(Polaris) | Heatmap(Polaris) | Heatmap(MoleculeACE) | Bar(MoleculeACE)

    Args:
        polaris_stats: Dict with keys 'participation', 'top_rank', 'pairwise_wins',
                       'pairwise_ties', 'n_tasks'
        moleculeace_stats: Same structure as polaris_stats
        output_path: Optional path to save figure
        figsize: Figure size

    Returns:
        Figure object
    """
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[0.3, 0.9, 0.3], wspace=0.3)

    ax_bar_polaris = fig.add_subplot(gs[0])
    ax_bar_moleculeace = fig.add_subplot(gs[2])

    gs_inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], wspace=0.0)
    ax_pair_polaris = fig.add_subplot(gs_inner[0])
    ax_pair_moleculeace = fig.add_subplot(gs_inner[1])

    # Helper to compute win rates
    def get_win_rates(stats):
        methods = sort_methods(list(stats["participation"].keys()))
        rates = []
        for m in methods:
            participation = stats["participation"].get(m, 0)
            top_rank = stats["top_rank"].get(m, 0)
            rate = (top_rank / participation) * 100 if participation > 0 else 0
            rates.append(rate)
        return methods, rates

    # Helper to build pairwise matrix
    def build_matrix(stats, methods):
        n = len(methods)
        matrix = np.zeros((n, n), dtype=int)
        for i, m_row in enumerate(methods):
            for j, m_col in enumerate(methods):
                if i != j:
                    wins = stats["pairwise_wins"].get((m_row, m_col), 0)
                    ties = stats["pairwise_ties"].get((m_row, m_col), 0)
                    matrix[i, j] = wins + ties
        return matrix

    # Polaris
    polaris_methods, polaris_rates = get_win_rates(polaris_stats)
    polaris_matrix = build_matrix(polaris_stats, polaris_methods)

    # MoleculeACE
    mace_methods, mace_rates = get_win_rates(moleculeace_stats)
    mace_matrix = build_matrix(moleculeace_stats, mace_methods)

    # Plot Polaris bar
    polaris_display = [format_method_name(m) for m in polaris_methods]
    x = np.arange(len(polaris_methods))
    bars_polaris = ax_bar_polaris.bar(x, polaris_rates, color="#6666FF")
    for bar, rate in zip(bars_polaris, polaris_rates):
        if bar.get_height() < 95:
            ax_bar_polaris.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{rate:.0f}%", ha="center", va="bottom", fontsize=FONTSIZE - 2, fontweight="bold",
            )
        else:
            ax_bar_polaris.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() - 3,
                f"{rate:.0f}%", ha="center", va="top", fontsize=FONTSIZE - 2, fontweight="bold", color="white",
            )
    ax_bar_polaris.set_ylabel("Win Rate (%)", fontsize=FONTSIZE)
    ax_bar_polaris.set_xticks(x)
    ax_bar_polaris.set_xticklabels(polaris_display, rotation=45, ha="right", fontsize=FONTSIZE)
    ax_bar_polaris.tick_params(axis="y", labelsize=FONTSIZE)
    ax_bar_polaris.set_ylim(0, 100)
    ax_bar_polaris.spines["top"].set_visible(False)
    ax_bar_polaris.spines["right"].set_visible(False)
    ax_bar_polaris.text(
        0.5,
        0.95,
        f"# tasks = {polaris_stats.get('n_tasks', 0)}",
        transform=ax_bar_polaris.transAxes,
        ha="center",
        va="top",
        fontsize=FONTSIZE - 2,
    )

    # Plot Polaris heatmap
    purple_cmap = LinearSegmentedColormap.from_list("purple", PURPLE_GRADIENT)
    ax_pair_polaris.imshow(polaris_matrix, cmap=purple_cmap)
    ax_pair_polaris.set_xticks(np.arange(len(polaris_methods)))
    ax_pair_polaris.set_yticks(np.arange(len(polaris_methods)))
    ax_pair_polaris.set_xticklabels(polaris_display, rotation=45, ha="right", fontsize=FONTSIZE)
    ax_pair_polaris.set_yticklabels(polaris_display, fontsize=FONTSIZE)
    for i in range(len(polaris_methods)):
        for j in range(len(polaris_methods)):
            if i != j:
                ax_pair_polaris.text(
                    j, i, str(polaris_matrix[i, j]),
                    ha="center", va="center", color="white", fontsize=FONTSIZE - 2, fontweight="bold"
                )

    # Plot MoleculeACE heatmap
    orange_cmap = LinearSegmentedColormap.from_list("orange", ORANGE_GRADIENT)
    mace_display = [format_method_name(m) for m in mace_methods]
    ax_pair_moleculeace.imshow(mace_matrix, cmap=orange_cmap)
    ax_pair_moleculeace.set_xticks(np.arange(len(mace_methods)))
    ax_pair_moleculeace.set_yticks([])
    ax_pair_moleculeace.set_xticklabels(mace_display, rotation=45, ha="right", fontsize=FONTSIZE)
    for i in range(len(mace_methods)):
        for j in range(len(mace_methods)):
            if i != j:
                ax_pair_moleculeace.text(
                    j, i, str(mace_matrix[i, j]),
                    ha="center", va="center", color="white", fontsize=FONTSIZE - 2, fontweight="bold"
                )

    # Plot MoleculeACE bar
    x = np.arange(len(mace_methods))
    bars_mace = ax_bar_moleculeace.bar(x, mace_rates, color="#FF9933")
    for bar, rate in zip(bars_mace, mace_rates):
        if bar.get_height() < 95:
            ax_bar_moleculeace.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{rate:.0f}%", ha="center", va="bottom", fontsize=FONTSIZE - 2, fontweight="bold",
            )
        else:
            ax_bar_moleculeace.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() - 3,
                f"{rate:.0f}%", ha="center", va="top", fontsize=FONTSIZE - 2, fontweight="bold", color="white",
            )
    ax_bar_moleculeace.set_ylabel("Win Rate (%)", fontsize=FONTSIZE)
    ax_bar_moleculeace.set_xticks(x)
    ax_bar_moleculeace.set_xticklabels(mace_display, rotation=45, ha="right", fontsize=FONTSIZE)
    ax_bar_moleculeace.tick_params(axis="y", labelsize=FONTSIZE)
    ax_bar_moleculeace.set_ylim(0, 100)
    ax_bar_moleculeace.spines["top"].set_visible(False)
    ax_bar_moleculeace.spines["right"].set_visible(False)
    ax_bar_moleculeace.text(
        0.5,
        0.95,
        f"# tasks = {moleculeace_stats.get('n_tasks', 0)}",
        transform=ax_bar_moleculeace.transAxes,
        ha="center",
        va="top",
        fontsize=FONTSIZE - 2,
    )

    # Add benchmark labels
    fig.text(0.243, 1.00, "Polaris", ha="center", va="top", fontsize=FONTSIZE + 2, fontweight="bold", color="#6666FF")
    fig.text(
        0.757, 1.00, "MoleculeACE",
        ha="center", va="top", fontsize=FONTSIZE + 2, fontweight="bold", color="#FF9933",
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig
