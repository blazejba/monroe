"""Styling constants and utilities for plots and tables."""

# Font sizes
FONTSIZE = 18
FONTSIZE_SMALL = 14
FONTSIZE_TITLE = 20

# Color palette
COLORS = {
    # Primary methods
    "monroe": "#6666FF",
    "minimol_pfn": "#BB3909",
    "minimol": "#FF9933",
    "chemeleon": "#7BB601",
    "chemeleon_pfn": "#4D7301",
    "molformer": "#666666",
    "gbm_ecfp_desc": "#8B5A2B",
    # Benchmark categories
    "polaris": "#6666FF",
    "moleculeace": "#FF9933",
    # Significance markers
    "significant": "#000000",
    "not_significant": "#FFFFFF",
    # General
    "best": "#6666FF",
    "tie": "#95A5A6",
    "worse": "#E74C3C",
}

# Purple gradient for heatmaps (Polaris)
PURPLE_GRADIENT = [
    "#B8B8FF",
    "#9999FF",
    "#7A7AFF",
    "#6666FF",
    "#5252CC",
    "#3D3D99",
    "#292966",
]

# Orange gradient for heatmaps (MoleculeACE)
ORANGE_GRADIENT = [
    "#FFCC99",
    "#FFB366",
    "#FF9933",
    "#E68A2E",
    "#CC7A29",
    "#B36B24",
    "#995C1F",
]

# Fixed method ordering (best methods first)
METHOD_ORDER = [
    "monroe",
    "minimol_pfn",
    "minimol_pt",
    "minimol",
    "minimol_native",
    "chemeleon_pfn",
    "chemeleon",
    "chemeleon_ft",
    "molformer",
    "molformer_ft",
    "gbm_ecfp_desc",
]

# Display name mapping
METHOD_DISPLAY_NAMES = {
    "monroe": "Monroe",
    "minimol": "MiniMol",
    "minimol_pfn": "MiniMol$_{PFN}$",
    "minimol_pt": "MiniMol$_{PFN}$",
    "minimol_native": "MiniMol",
    "chemeleon": "CheMeleon",
    "chemeleon_pfn": "CheMeleon$_{PFN}$",
    "chemeleon_ft": "CheMeleon",
    "molformer": "MolFormer",
    "molformer_ft": "MolFormer",
    "gbm_ecfp_desc": "GBM",
    "tabpfn_minimol_pt": "MiniMol$_{PFN}$",
    "tabpfn_minimol": "MiniMol$_{PFN}$",
    "native_minimol": "MiniMol",
    "tabpfn_chemeleon_ft": "CheMeleon$_{PFN}$",
    "native_chemeleon_ft": "CheMeleon",
    "tabpfn_molformer_ft": "MolFormer$_{PFN}$",
    "native_molformer_ft": "MolFormer",
}


# MoleculeACE task display names (CHEMBL ID → target short name)
MOLECULEACE_TASK_NAMES = {
    "CHEMBL1862_Ki": "ABL1",
    "CHEMBL1871_Ki": "AR",
    "CHEMBL2047_EC50": "FXR",
    "CHEMBL2147_Ki": "PIM1",
    "CHEMBL219_Ki": "D4R",
    "CHEMBL228_Ki": "SERT",
    "CHEMBL231_Ki": "HRH1",
    "CHEMBL233_Ki": "MOR",
    "CHEMBL234_Ki": "D3R",
    "CHEMBL235_EC50": r"PPAR$\gamma$",
    "CHEMBL236_Ki": "DOR",
    "CHEMBL237_EC50": "KOR (a)",
    "CHEMBL237_Ki": "KOR (i)",
    "CHEMBL238_Ki": "DAT",
    "CHEMBL239_EC50": r"PPAR$\alpha$",
    "CHEMBL244_Ki": "FX",
    "CHEMBL262_Ki": "GSK3b",
    "CHEMBL264_Ki": "HRH3",
    "CHEMBL287_Ki": "SOR",
    "CHEMBL2971_Ki": "JAK2",
    "CHEMBL3979_EC50": r"PPAR$\delta$",
    "CHEMBL4005_Ki": "p110a",
    "CHEMBL4203_Ki": "CLK4",
    "CHEMBL4616_EC50": "GHSR",
    "CHEMBL4792_Ki": "OX2R",
    "CHEMBL2034_Ki": "GR",
    "CHEMBL204_Ki": "F2",
    "CHEMBL214_Ki": "5-HT1A",
    "CHEMBL218_EC50": "CB1",
    "CHEMBL2835_Ki": "JAK1",
}


def format_method_name(name: str) -> str:
    """Format method name for display in plots/tables.

    Args:
        name: Raw method name (e.g., 'tabpfn_minimol_pt')

    Returns:
        Formatted display name (e.g., 'MiniMol$_{PFN}$')
    """
    name_lower = name.lower()

    # Check exact match first
    if name_lower in METHOD_DISPLAY_NAMES:
        return METHOD_DISPLAY_NAMES[name_lower]

    # Try to parse structured names
    has_pfn = name_lower.startswith("tabpfn_")
    has_native = name_lower.startswith("native_")

    # Extract base name
    base = name_lower
    if has_pfn:
        base = base[7:]  # Remove 'tabpfn_'
    elif has_native:
        base = base[7:]  # Remove 'native_'

    # Remove common suffixes
    for suffix in ["_pt", "_ft", "_pretrained", "_finetuned"]:
        base = base.replace(suffix, "")

    # Look up base name
    if base in METHOD_DISPLAY_NAMES:
        display = METHOD_DISPLAY_NAMES[base]
        if has_pfn and "$_{PFN}$" not in display:
            display = display + "$_{PFN}$"
        return display

    # Fallback: title case
    return name.replace("_", " ").title()


def get_method_color(name: str) -> str:
    """Get color for a method.

    Args:
        name: Method name

    Returns:
        Hex color string
    """
    name_lower = name.lower()

    # Check exact match
    if name_lower in COLORS:
        return COLORS[name_lower]

    # Check if it contains a known method
    for key in ["monroe", "minimol", "chemeleon", "molformer"]:
        if key in name_lower:
            return COLORS[key]

    # Default gray
    return "#95A5A6"


def sort_methods(methods: list[str]) -> list[str]:
    """Sort methods according to fixed ordering.

    Args:
        methods: List of method names

    Returns:
        Sorted list with known methods first in order, unknown at end
    """
    def get_priority(m: str) -> int:
        m_lower = m.lower()
        # Check exact match
        for i, ordered in enumerate(METHOD_ORDER):
            if ordered in m_lower:
                return i
        return len(METHOD_ORDER)  # Unknown methods go to end

    return sorted(methods, key=get_priority)
