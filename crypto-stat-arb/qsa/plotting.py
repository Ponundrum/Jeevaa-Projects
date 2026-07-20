"""One consistent visual language for every chart and table in the project.

Import ``apply_style`` once at the top of a notebook; use ``CLR`` for a fixed
per-strategy colour and ``style_table`` for a captioned, lightly-styled table.
"""
import matplotlib.pyplot as plt

# A fixed colour for each strategy so it reads the same in every figure.
CLR = {
    "Combined": "#1b1b1b", "Momentum": "#1f77b4", "Low-vol": "#2ca02c",
    "Carry": "#9467bd", "BTC": "#ff7f0e", "EW": "#8c8c8c",
    "reject": "#d62728", "keep": "#2ca02c",
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
