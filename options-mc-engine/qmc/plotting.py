"""House matplotlib style — consistent, clean, publication-ish — with a fixed
colour per model so figures read the same everywhere (parallel to the sibling
project's ``qsa/plotting.py``)."""
from __future__ import annotations

import matplotlib.pyplot as plt

# Fixed colour per model / series.
CLR = {
    "MC": "#1b1b1b", "BS": "#1f77b4", "Black-Scholes": "#1f77b4",
    "Heston": "#2ca02c", "rBergomi": "#d62728", "Rough Bergomi": "#d62728",
    "Market": "#ff7f0e", "theory": "#8c8c8c",
}


def apply_style():
    """Set the house style. Call once per notebook."""
    plt.rcParams.update({
        "figure.figsize": (11, 4.5), "figure.dpi": 110, "savefig.dpi": 120,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
        "axes.titlelocation": "left", "legend.frameon": False, "legend.fontsize": 9,
        "font.size": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    })
