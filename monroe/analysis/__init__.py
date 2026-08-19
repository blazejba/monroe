"""Monroe analysis module for results visualization and statistical comparison."""

from monroe.analysis.loader import (
    aggregate_seeds,
    extract_for_comparison,
    load_multiple_methods,
    load_results,
)
from monroe.analysis.stats import (
    GamesHowellResult,
    PairwiseComparison,
    benjamini_hochberg_fdr,
    compute_win_stats,
    games_howell_test,
    pairwise_comparison,
    welch_ttest,
)
from monroe.analysis.style import (
    COLORS,
    METHOD_ORDER,
    format_method_name,
    sort_methods,
)
from monroe.analysis.tables import (
    AblationTable,
    CombinedSummaryTable,
    LeaderboardTable,
    ResultsTable,
    build_results_table,
)

__all__ = [
    # stats
    "benjamini_hochberg_fdr",
    "compute_win_stats",
    "games_howell_test",
    "welch_ttest",
    "pairwise_comparison",
    "GamesHowellResult",
    "PairwiseComparison",
    # loader
    "load_results",
    "aggregate_seeds",
    "load_multiple_methods",
    "extract_for_comparison",
    # style
    "COLORS",
    "METHOD_ORDER",
    "format_method_name",
    "sort_methods",
    # tables
    "ResultsTable",
    "AblationTable",
    "CombinedSummaryTable",
    "LeaderboardTable",
    "build_results_table",
]
