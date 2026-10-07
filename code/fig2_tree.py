"""Fig. 2 of the paper: the tree approximation.  (a) Graphical solution of P = q(CP)^2,
(b) C = u/q(u) and u/q(u)^2, (c) P_inf^mul(C) from the tree equation against the simulation of the multiplex
(exp1_s*_q1.00_N16.npz, from experiment.py).
Usage: python fig2_tree.py   -> ../figures/fig_tree_v11.pdf"""
import paths                                    # the result files are read and written in ../data
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import brentq
from bethe import q_active, C_bethe_mux, C_schulman
import paper_style
from paper_style import label, panel

OUT = paths.FIGURES
SUF = "_v11"
paper_style.use()
BLUE, ORANGE, GREEN, GREY, RED = "#2a5d9f", "#d97b28", "#3f8f5f", "#777777", "#b03a48"
CT = r"C^{\rm tree}_{\rm mul}"

fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.45))

# (a) graphical solution of P = q(CP)^2 at sigma = 0.5
s = 0.5
CB, us, Pj = C_bethe_mux(s)
P = np.linspace(1e-4, 1, 600)
a = ax[0]
CHI = 0.54                                     # the example with C above the threshold
for C, col, ls, lab in [(CHI, GREEN, "-.", rf"$C>{CT}$"), (CB, BLUE, "-", rf"$C={CT}$"),
                        (0.36, GREY, "--", rf"$C<{CT}$")]:
    a.plot(P, [q_active(C * p, s) ** 2 for p in P], color=col, ls=ls, label=lab)
a.plot([0, 1], [0, 1], color="k", lw=0.7)
a.plot([Pj], [Pj], "o", color=BLUE, ms=3.5)
a.annotate(r"$P_\infty^\star$", (Pj, Pj), (0.62, 0.38), fontsize=10, arrowprops=dict(arrowstyle="-", lw=0.5))
# the two non-zero solutions for C > C_tree: the curve crosses the diagonal (stable: filled, unstable: open)
h = lambda p: q_active(CHI * p, s) ** 2 - p
P_un, P_st = brentq(h, 1e-3, Pj), brentq(h, Pj, 1.0)
a.plot([P_st], [P_st], "o", color=GREEN, ms=4.2, zorder=5)
a.plot([P_un], [P_un], "o", mfc="white", mec=GREEN, mew=1.0, ms=4.2, zorder=5)
print(f"C = {CHI}: unstable solution P = {P_un:.3f}, stable solution P = {P_st:.3f}")
a.set_xlim(0, 1); a.set_ylim(0, 1)
label(a, x=r"$P_\infty$", y=r"$q(CP_\infty)^2$")
a.legend(loc="upper left", fontsize=8.5, handlelength=1.8)
panel(a, "(a)")
a.text(0.95, 0.06, r"$\sigma=0.5$", transform=a.transAxes, ha="right")

# (b) C as a function of u: one layer and two layers
a = ax[1]
u = np.linspace(0.004, 0.8, 500)
Q = np.array([q_active(x, s) for x in u])
a.plot(u, u / Q, color=ORANGE, label=r"one layer, $u/q(u)$")
a.plot(u, u / Q ** 2, color=BLUE, label=r"multiplex, $u/q(u)^2$")
a.axhline(C_schulman(s), color=ORANGE, lw=0.6, ls=":")
a.plot([us], [CB], "o", color=BLUE, ms=3.5)
a.annotate(rf"${CT}$", (us, CB), (0.33, 0.24), fontsize=10, arrowprops=dict(arrowstyle="-", lw=0.5))
a.annotate(r"$C_{\rm S}$", (0.02, C_schulman(s)), (0.1, 0.07), fontsize=10, arrowprops=dict(arrowstyle="-", lw=0.5))
a.set_xlim(0, 0.8); a.set_ylim(0, 1.2)
label(a, x="$u$", y="$C$")
a.legend(loc="upper center", fontsize=8.5)
panel(a, "(b)")
a.text(0.95, 0.06, r"$\sigma=0.5$", transform=a.transAxes, ha="right")

# (c) P(C): tree equation against the multiplex simulation
a = ax[2]
handles = []
for s, col in [(0.2, RED), (0.5, BLUE), (0.8, GREEN)]:
    CB, us, Pj = C_bethe_mux(s)
    uu = np.linspace(us, 0.97, 400); QQ = np.array([q_active(x, s) for x in uu])
    Cc = uu / QQ ** 2; k = Cc <= 0.92
    a.plot(Cc[k], (QQ ** 2)[k], color=col, lw=0.9)
    ul = np.linspace(0.004, us, 400); Ql = np.array([q_active(x, s) for x in ul])
    Cl = ul / Ql ** 2; k = Cl <= 0.92
    a.plot(Cl[k], (Ql ** 2)[k], color=col, lw=0.8, ls=(0, (1.2, 1.2)))
    a.plot([CB, CB], [0, Pj], color=col, lw=0.6, ls=":")
    d = np.load(f"exp1_s{s:.3f}_q1.00_N16.npz")
    a.plot(d["Cgrid"], d["mu_id"].mean(0), "o", ms=2.2, color=col, mfc="none", mew=0.6)
    handles.append(Line2D([], [], color=col, lw=0.9, marker="o", ms=3, mfc="white", mew=0.7,
                          label=rf"$\sigma={s}$" if not handles else f"${s}$"))
a.set_xlim(0.15, 0.92); a.set_ylim(0, 1)
label(a, x="$C$", y=r"$P_\infty^{\rm mul}$")
# the legend is above the axes, clear of the data
a.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.03, 0.995), ncol=3, fontsize=8.5, handletextpad=0.25,
         columnspacing=0.9, handlelength=1.2, borderaxespad=0.1)
panel(a, "(c)")

fig.tight_layout(w_pad=1.0)
fig.savefig(f"{OUT}/fig_tree{SUF}.pdf")
print("saved")
