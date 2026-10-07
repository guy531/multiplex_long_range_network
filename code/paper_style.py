"""Common style of the figures of the paper: Times New Roman for the words, to match the text of the paper,
Computer Modern for the formulas, as in the equations of the paper, and larger axis labels when the label is a
formula."""
import matplotlib.pyplot as plt

MATH_LABEL = 12      # axis labels that are a formula only, such as $C$ or $P_\infty$
TEXT_LABEL = 9.5     # axis labels with words


def use():
    plt.rcParams.update({"font.size": 9, "font.family": "serif", "font.serif": ["Times New Roman"],
                         "mathtext.fontset": "cm", "axes.labelsize": TEXT_LABEL, "xtick.labelsize": 8.5,
                         "ytick.labelsize": 8.5, "axes.linewidth": 0.6, "lines.linewidth": 1.1,
                         "legend.frameon": False, "figure.dpi": 200, "savefig.bbox": "tight",
                         "xtick.direction": "in", "ytick.direction": "in",
                         "pdf.fonttype": 42})          # TrueType fonts in the PDF instead of Type 3


def panel(ax, text):
    """panel label, (a), (b), ..., above the axes and aligned with their left edge"""
    ax.text(0.0, 1.025, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=10)


def is_formula(s):
    return s.startswith("$") and s.endswith("$") and s.count("$") == 2


def label(ax, x=None, y=None):
    """axis labels; a label that is only a formula is set larger"""
    if x is not None:
        ax.set_xlabel(x, fontsize=MATH_LABEL if is_formula(x) else TEXT_LABEL)
    if y is not None:
        ax.set_ylabel(y, fontsize=MATH_LABEL if is_formula(y) else TEXT_LABEL)
