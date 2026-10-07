"""Fig. 1 of the paper: schematic of the model.  Two one-dimensional long-range layers, coupled node by node
with dependency links.  The layers are infinite: links that leave the drawn window are cut at its edge, and
dots mark the continuation.
The figure shows an example of a set of functional nodes (filled circles).  It is computed from the drawn
links with the cascade of the paper, so the picture is consistent with the definition.
Usage: python fig1_model.py          -> ../figures/fig_model_v11.pdf
       python fig1_model.py plain    -> ../figures/fig_model_plain.pdf  (all nodes drawn alike, no cascade)"""
import paths                                    # the result files are read and written in ../data
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Rectangle
import paper_style

OUT = paths.FIGURES
PLAIN = len(sys.argv) > 1 and sys.argv[1] == "plain"
paper_style.use()
BLUE, ORANGE, GREY, LIGHT = "#2a5d9f", "#d97b28", "#777777", "#c4c4c4"
n = 16
EDGE = 0.75                                   # links are drawn up to this distance beyond the first/last node
# links of the drawn nodes 0..15; pairs with an end outside 0..15 continue beyond the window
if PLAIN:
    LA = [(-1, 0), (0, 1), (1, 2), (3, 4), (4, 5), (5, 6), (6, 7), (8, 9), (9, 10), (11, 12), (12, 13), (13, 14),
          (14, 15), (15, 16), (2, 5), (7, 8), (1, 9), (10, 12), (6, 13), (-3, 3), (12, 19)]
    LB = [(-1, 0), (0, 1), (2, 3), (3, 4), (4, 5), (6, 7), (7, 8), (8, 9), (10, 11), (11, 12), (13, 14), (14, 15),
          (15, 16), (1, 3), (5, 7), (9, 11), (0, 6), (8, 13), (12, 15), (-2, 2), (14, 18)]
else:
    # In A, node 4 has no links and nodes 10 and 11 are linked
    # only to each other; in B, nodes 4 and 10 are linked only to each other, and node 11 has no links.
    LA = [(-1, 0), (0, 1), (1, 2), (2, 3), (5, 6), (6, 7), (7, 8), (8, 9), (10, 11), (12, 13), (13, 14),
          (14, 15), (15, 16), (3, 5), (1, 9), (6, 13), (-3, 3), (12, 19)]
    LB = [(-1, 0), (0, 1), (2, 3), (6, 7), (7, 8), (8, 9), (13, 14), (14, 15),
          (15, 16), (1, 3), (5, 7), (0, 6), (8, 13), (12, 15), (-2, 2), (14, 18), (4, 10)]


def largest_cluster(links, alive):
    """nodes of the largest connected cluster of the drawn nodes that are alive"""
    lab = {i: i for i in range(n) if alive[i]}
    for i, j in links:
        if i in lab and j in lab:
            a_, b_ = lab[i], lab[j]
            if a_ != b_:
                for k in lab:
                    if lab[k] == b_:
                        lab[k] = a_
    if not lab:
        return set()
    best = max(set(lab.values()), key=lambda c: sum(v == c for v in lab.values()))
    return {k for k, v in lab.items() if v == best}


alive = [True] * n
if not PLAIN:
    while True:                                # cascade of failures on the drawn window
        changed = False
        for links in (LA, LB):
            keep = largest_cluster(links, alive)
            for i in range(n):
                if alive[i] and i not in keep:
                    alive[i] = False; changed = True
        if not changed:
            break
    print("functional nodes:", [i for i in range(n) if alive[i]], " failed:", [i for i in range(n) if not alive[i]])
ok = lambda i: i < 0 or i >= n or alive[i]     # nodes outside the window are not drawn; their links are kept

yA, yB = 1.0, 0.0
fig, a = plt.subplots(figsize=(3.4, 1.72 if not PLAIN else 1.55))
clip = Rectangle((-EDGE, -5), n - 1 + 2 * EDGE, 10, transform=a.transData)
for i in range(n):
    a.plot([i, i], [yB, yA], ls=(0, (2, 2)), color=GREY if alive[i] else LIGHT, lw=0.6, zorder=1)
for links, y, sgn, col in [(LA, yA, 1, BLUE), (LB, yB, -1, ORANGE)]:
    for i, j in links:
        r = j - i
        live = ok(i) and ok(j)
        c, z = (col, 2) if live else (LIGHT, 1.5)
        if r == 1:
            art = a.plot([i, j], [y, y], color=c, lw=1.0 if live else 0.8, zorder=z)[0]
        else:
            h = 0.13 * r ** 0.75
            art = Arc(((i + j) / 2, y), r, 2 * h, theta1=0 if sgn > 0 else 180, theta2=180 if sgn > 0 else 360,
                      color=c, lw=0.9 if live else 0.8, zorder=z)
            a.add_patch(art)
        art.set_clip_path(clip)
    for i in range(n):
        if PLAIN:
            a.scatter([i], [y], s=16, color="white", edgecolor=col, lw=1.0, zorder=3)
        elif alive[i]:
            a.scatter([i], [y], s=17, color=col, edgecolor=col, lw=1.0, zorder=3)
        else:
            a.scatter([i], [y], s=16, color="white", edgecolor=GREY, lw=0.9, zorder=3)
    for x in (-EDGE - 0.55, n - 1 + EDGE + 0.55):
        a.text(x, y, r"$\cdots$", ha="center", va="center", fontsize=10, color=col)
a.text(-2.3, yA, r"$\mathcal{A}$", ha="right", va="center", fontsize=11, color=BLUE)
a.text(-2.3, yB, r"$\mathcal{B}$", ha="right", va="center", fontsize=11, color=ORANGE)
i, j = 1, 9
h = 0.13 * (j - i) ** 0.75
a.annotate("", xy=(j, yA + h + 0.13), xytext=(i, yA + h + 0.13),
           arrowprops=dict(arrowstyle="<->", lw=0.6, color="k", shrinkA=0, shrinkB=0))
a.text((i + j) / 2, yA + h + 0.2, r"$r=|i-j|$,  probability $C/r^{1+\sigma}$", ha="center", va="bottom", fontsize=8.5)
a.text(i - 0.22, yA + 0.07, "$i$", ha="right", va="bottom", fontsize=9)
a.text(j + 0.22, yA + 0.07, "$j$", ha="left", va="bottom", fontsize=9)   # top right: the arc from i enters j from the top left
a.annotate("dependency\nlink", xy=(15, 0.5), xytext=(16.5, 0.5), fontsize=8, va="center", color=GREY,
           arrowprops=dict(arrowstyle="-", lw=0.5, color=GREY))
if PLAIN:
    a.set_xlim(-2.9, 19.6); a.set_ylim(-0.6, 2.1)
else:
    handles = [Line2D([], [], ls="", marker="o", ms=4.2, mfc=GREY, mec=GREY, label="functional"),
               Line2D([], [], ls="", marker="o", ms=4.2, mfc="white", mec=GREY, mew=0.9, label="not functional")]
    a.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.47, -0.06), ncol=2, fontsize=8,
             handletextpad=0.1, columnspacing=1.6, borderaxespad=0)
    a.set_xlim(-2.9, 19.6); a.set_ylim(-0.95, 2.1)
a.axis("off")
name = "fig_model_plain.pdf" if PLAIN else "fig_model_v11.pdf"
fig.savefig(f"{OUT}/{name}", pad_inches=0.02)
print("saved", name)
