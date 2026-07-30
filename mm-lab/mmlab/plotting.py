"""One consistent visual language for every chart in the project — the same house
style as the sibling projects' ``qsa/plotting.py``. Import ``apply_style`` once at
the top of a notebook; use ``CLR`` for a fixed per-strategy colour.
"""
import matplotlib.pyplot as plt

# A fixed colour per strategy / series so it reads the same in every figure.
CLR = {
    "Naive": "#d62728", "AS": "#1f77b4", "AS-frozen": "#1f77b4",
    "mid": "#1b1b1b", "bid": "#2ca02c", "ask": "#d62728",
    "markout": "#9467bd", "fit": "#ff7f0e", "points": "#1b1b1b",
}


def apply_style():
    """Set the house matplotlib style. Call once per notebook."""
    plt.rcParams.update({
        "figure.figsize": (11, 4.5), "figure.dpi": 110, "savefig.dpi": 110,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
        "axes.titlelocation": "left", "legend.frameon": False, "legend.fontsize": 9,
        "font.size": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    })


def style_table(styler, caption):
    """House style for a pandas Styler: left-aligned caption above the table."""
    return (styler.set_caption(caption)
            .set_table_styles([{"selector": "caption", "props": [
                ("font-weight", "600"), ("font-size", "11pt"),
                ("text-align", "left"), ("padding-bottom", "6px"), ("color", "#222")]}]))
