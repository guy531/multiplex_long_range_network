"""Figs. 3 and 4 of the paper.
  Fig. 3 (two-column width, four panels):
     (a) Method 1 on one realization: P_inf^mul(C), the values of C at which it was calculated, and the
         largest jump Delta                (dense_s0.200_N20_k6.npz, from dense_realization.py)
     (b) NOI against C for N = 10^5 (../method2_N1e5/noi_N100000.csv) and N = 2^20 (noi_ring_s*_N20.npz)
     (c) single realizations for sigma = 0.2 and 0.5, each for N = 2^10 and 2^20, against C; the realization
         with the median largest jump                (dense_s*_N*_k*.npz, from dense_realization.py)
     (d) largest jump against N         (jump_s*_q1.00_N*.npy, from experiment2.py; open circles: noi_ring_*)
  Fig. 4 (one column): thresholds against sigma, with a gray band at 0.35 < sigma < 0.4, where the
     simulations place the change of the order of the transition.
Usage: python fig3_fig4_simulations.py           -> ../figures/fig_sim_v11_row.pdf (Fig. 3, one row, as in
                                                   the paper) and fig_thresholds_v11.pdf (Fig. 4)
       python fig3_fig4_simulations.py tworows   -> fig_sim_v11.pdf (the panels of Fig. 3 in two rows) and Fig. 4"""
import paths                                    # the result files are read and written in ../data
import glob, re, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import brentq
from scipy.special import zeta
from bethe import C_bethe_mux, C_schulman, C_star
import paper_style
from paper_style import label, panel

OUT = paths.FIGURES
ROW = not (len(sys.argv) > 1 and sys.argv[1] == "tworows")   # the four panels of Fig. 3 in one row
paper_style.use()
if ROW:                                               # narrow panels: smaller ticks, legends and symbols
    plt.rcParams.update({"xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 6.5,
                         "xtick.major.pad": 2.0, "ytick.major.pad": 2.0, "axes.labelpad": 2.0})
FS = 6.5 if ROW else 8                                # legends and notes inside the panels of Fig. 3
MS = 0.8 if ROW else 1.0                              # scale of the symbols
BLUE, ORANGE, GREEN, GREY, RED = "#2a5d9f", "#d97b28", "#3f8f5f", "#777777", "#b03a48"
LBLUE, LRED = "#9db9dc", "#e0a3aa"
GORI = {0.1: 0.047685, 0.2: 0.093211, 0.3: 0.140546, 0.35: 0.16610, 0.4: 0.193471, 0.45: 0.22293,
        0.5: 0.25482, 0.55: 0.289410, 0.6: 0.327098, 0.65: 0.368333, 0.7: 0.413752, 0.75: 0.464202,
        0.8: 0.521001, 0.85: 0.586264, 0.9: 0.66408, 0.95: 0.76501}
N1E5 = paths.N1E5                                    # results of Method 2 with N = 10^5
PL = r"$P_\infty^{\rm mul}$"


def C_closed(s):
    z = zeta(1 + s)
    f = lambda C: 2 * C * z * (1 - ((1 - C) * (1 - C / -np.expm1(-2 * C * z))) ** z) - 1
    return brentq(f, 1e-4, 0.6)


# columns: multiplex [C_jump, jump, P_after], random dependency links [C_jump, jump, P_after]
jump = {}
for f in sorted(glob.glob("jump_s*_q1.00_N*.npy")):
    m = re.match(r"jump_s([\d.]+)_q1.00_N(\d+)\.npy", f)
    jump[(float(m.group(1)), int(m.group(2)))] = np.load(f)[:, [0, 1, 2, 5, 6, 7]]
jump[(0.2, 18)] = np.load("exp2_s0.200_q1.00_N18.npy")
sig = sorted({k[0] for k in jump})
Lmax = {s: max(L for (t, L) in jump if t == s) for s in sig}
mux = {s: jump[(s, Lmax[s])][:, 0].mean() for s in sig}
tab = pd.read_csv(f"{N1E5}/thresholds_N100000.csv").iloc[:, :3]
tab.columns = ["sigma", "Cc", "steps"]
PLAIN = dict(zip(np.round(tab.sigma, 3), tab.Cc))
for f in sorted(glob.glob("noi_plainring_s*_N100000.csv")):   # sigma = 0.35 (method2_N1e5_sigma.py)
    g = pd.read_csv(f)
    s = round(float(g.Sigma.iloc[0]), 3)
    if s not in PLAIN:
        PLAIN[s] = g.C.values[g.Avg_Cascade_Steps.values.argmax()]


def mean_err(s, col):
    Ls = sorted(L for (t, L) in jump if t == s)
    M = np.array([jump[(s, L)][:, col].mean() for L in Ls])
    E = np.array([jump[(s, L)][:, col].std() / np.sqrt(len(jump[(s, L)])) for L in Ls])
    return np.array(Ls), M, E


# ====================================================================== Fig. 3: four panels
if ROW:
    fig = plt.figure(figsize=(7.0, 2.2))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1.25, 1.05], wspace=0.40, left=0.058, right=0.995, top=0.90,
                          bottom=0.215)
    SLOT = [gs[0, 0], gs[0, 1], gs[0, 2], gs[0, 3]]
else:
    fig = plt.figure(figsize=(7.0, 5.9))
    gs = fig.add_gridspec(2, 2, hspace=0.36, wspace=0.27, left=0.075, right=0.985, top=0.955, bottom=0.085)
    SLOT = [gs[0, 0], gs[0, 1], gs[1, 0], gs[1, 1]]

# ---------------------------------------------------------------- (a) Method 1 on one realization
a = fig.add_subplot(SLOT[0])
d = np.load("dense_s0.200_N20_k6.npz")
Cj, D, Pa, Pb_, w = d["jump"]
a.step(d["C"], d["P"], where="post", color=GREY, lw=0.9, zorder=1)
# all the values of C at which P was calculated (the 25 initial ones and those added by the halving), drawn alike
a.plot(np.concatenate([d["Cg"], d["Cb"]]), np.concatenate([d["Pg"], d["Pb"]]), "o", ms=3.6 * MS, mfc="white",
       mec=BLUE, mew=0.8, zorder=3)
x = Cj - (0.02 if ROW else 0.012)
a.annotate("", xy=(x, Pb_), xytext=(x, Pa), arrowprops=dict(arrowstyle="<->", lw=0.7, color="k", shrinkA=0, shrinkB=0))
a.plot([x - 0.004, Cj], [Pb_, Pb_], color="k", lw=0.5, ls=":")
a.text(x - 0.006, 0.5 * (Pa + Pb_), r"$\Delta$", ha="right", va="center", fontsize=10 if ROW else 12)
a.set_xlim(d["Cg"][0], d["Cg"][-1]); a.set_ylim(-0.03, 1)
label(a, x="$C$", y=PL)
if ROW:
    a.text(0.04, 0.91, r"$\sigma=0.2$", transform=a.transAxes, fontsize=FS + 0.5)
    a.text(0.04, 0.81, r"$N=2^{20}$", transform=a.transAxes, fontsize=FS + 0.5)
else:
    a.text(0.035, 0.91, r"$\sigma=0.2$, $N=2^{20}$", transform=a.transAxes, fontsize=FS + 0.5)
if ROW:
    a.set_xticks([0.1, 0.2, 0.3])
panel(a, "(a)")

# ---------------------------------------------------------------- (b) NOI, two sizes
a = fig.add_subplot(SLOT[1])
dd = pd.read_csv(f"{N1E5}/noi_N100000.csv")
for s, col in [(0.3, RED), (0.5, BLUE), (0.8, GREEN)]:
    g = dd[np.isclose(dd.Sigma, s)]
    a.plot(g.C, g.Avg_Cascade_Steps, "--", lw=0.8, color=col)
    z = np.load(f"noi_ring_s{s:.3f}_N20.npz")
    C = np.concatenate([z["C1"], z["C2"]]); I = np.concatenate([z["I1"].mean(0), z["I2"].mean(0)])
    C, i = np.unique(np.round(C, 6), return_index=True)
    a.plot(C, I[i], "-", lw=0.9, color=col, label=rf"$\sigma={s}$")
a.set_xlim(0.15, 0.92); a.set_ylim(0, 58)
label(a, x="$C$", y="NOI")
a.legend(loc="upper right", handletextpad=0.3, handlelength=1.1 if ROW else 1.4, borderaxespad=0.3)
if ROW:
    a.text(0.97, 0.52, "dashed: $N=10^5$\n" + r"solid: $N\approx10^6$", transform=a.transAxes, fontsize=FS,
           ha="right", va="top", linespacing=1.3)
    a.set_xticks([0.2, 0.4, 0.6, 0.8])
else:
    a.text(0.45, 0.9, r"dashed: $N=10^5$", transform=a.transAxes, fontsize=8)
    a.text(0.45, 0.81, r"solid: $N\approx10^6$", transform=a.transAxes, fontsize=8)
panel(a, "(b)")

# ---------------------------------------------------------------- (c) single realizations, two sizes
# Two halves, each with its own window of C around the threshold (dense_realization.py).  The small system is drawn
# wider and lighter, below the large one.  The two ends of the largest jump of each realization are marked
# (open circles: small system, filled circles: large system).
sub = SLOT[2].subgridspec(1, 2, wspace=0.09)
CASES = [(0.2, [(10, 15), (20, 6)], LRED, RED, (0.2085, 0.2335), "first order"),
         (0.5, [(10, 15), (20, 2)], LBLUE, BLUE, (0.431, 0.479), "continuous")]
for j, (s, runs, light, dark, xl, name) in enumerate(CASES):
    a = fig.add_subplot(sub[0, j])
    handles = []
    for L, k in runs:
        d = np.load(f"dense_s{s:.3f}_N{L}_k{k}.npz")
        i = int(np.diff(d["P"]).argmax())
        Cb, Pa, Pb = d["C"][i + 1], d["P"][i], d["P"][i + 1]
        if L == 20:
            col, lw, z = dark, 0.9, 3
            mk = dict(marker="o", ms=3.0 * MS, mfc=dark, mec=dark, mew=0.8)
        else:
            col, lw, z = light, 1.7, 2
            mk = dict(marker="o", ms=5.4 * MS, mfc="none", mec=dark, mew=0.9)
        a.step(d["C"], d["P"], where="post", color=col, lw=lw, zorder=z)
        a.plot([Cb, Cb], [Pa, Pb], ls="", zorder=z + 3, clip_on=False, **mk)
        handles.append(Line2D([], [], color=col, lw=lw, label=rf"$N=2^{{{L}}}$", **mk))
        print(f"panel (c): sigma={s} N=2^{L}: largest jump {Pb - Pa:.3f} at C={Cb:.4f}")
    a.set_xlim(*xl); a.set_ylim(0, 0.8)
    if ROW:
        a.set_xticks([0.21, 0.23] if j == 0 else [0.44, 0.47])
    else:
        a.set_xticks(np.round(np.linspace(xl[0] + 0.0015, xl[1] - 0.0015, 3), 2) if j == 0 else [0.44, 0.46])
    a.text(0.96, 0.05, rf"$\sigma={s}$" + "\n" + name, transform=a.transAxes, ha="right", va="bottom",
           fontsize=FS + 0.5)
    a.legend(handles=handles, loc="upper left", fontsize=FS, handlelength=1.2 if ROW else 1.5, handletextpad=0.3,
             borderaxespad=0.2)
    if j == 0:
        label(a, y=PL); panel(a, "(c)")
    else:
        a.set_yticklabels([])
    a.set_xlabel("$C$", fontsize=paper_style.MATH_LABEL)

# ---------------------------------------------------------------- (d) largest jump against N
a = fig.add_subplot(SLOT[3])
cm = plt.get_cmap("viridis")
colour = lambda s: cm((s - 0.1) / 0.75)
for s in [0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.65, 0.8]:
    Ls, M, E = mean_err(s, 1)
    a.errorbar(2.0 ** Ls, M, E, color=colour(s), marker="s", ms=2.6 * MS, lw=0.8, capsize=1.2, label=rf"${s:g}$")
for s in [0.1, 0.2, 0.3, 0.35]:               # the runs in which the two methods were compared (open symbols)
    pts = []
    for L in [14, 16, 18, 20]:
        try:
            J = np.load(f"noi_ring_s{s:.3f}_N{L}.npz")["jump"][:, 1]
        except (OSError, KeyError):
            continue
        pts.append((2.0 ** L, J.mean(), J.std() / np.sqrt(len(J))))
    if pts:
        P = np.array(pts)
        a.errorbar(P[:, 0], P[:, 1], P[:, 2], color=colour(s), marker="o", ms=3.4 * MS, mfc="white", mew=0.7,
                   lw=0, elinewidth=0.6, capsize=1.2, zorder=5)
for s in [0.4, 0.65, 0.8]:                    # the same layers with random dependency links (dashed)
    Ls, M, E = mean_err(s, 4)
    a.errorbar(2.0 ** Ls, M, E, color=colour(s), marker="^", ms=2.8 * MS, mfc="white", mew=0.7, lw=0.8, ls="--",
               capsize=1.2)
a.set_xscale("log", base=2); a.set_yscale("log")
a.set_ylim(0.012 if ROW else 0.022, 4.0 if ROW else 1.5)
a.set_yticks([0.05, 0.1, 0.2, 0.5, 1]); a.set_yticklabels(["0.05", "0.1", "0.2", "0.5", "1"])
a.minorticks_off()
label(a, x="$N$", y=r"largest jump $\Delta$")
if ROW:
    a.set_xticks([2.0 ** 12, 2.0 ** 16, 2.0 ** 20])
    a.legend(title=r"$\sigma$", fontsize=6, ncol=3, loc="lower left", title_fontsize=7.5, columnspacing=0.5,
             handlelength=1.0, handletextpad=0.2, borderaxespad=0.2, labelspacing=0.25)
    a.text(0.04, 0.965, "dashed: random\ndependency links", transform=a.transAxes, fontsize=6.5, va="top",
           linespacing=1.2)
else:
    a.legend(title=r"$\sigma$", fontsize=7.5, ncol=3, loc="lower left", title_fontsize=9.5, columnspacing=0.8,
             handlelength=1.4)
    a.text(0.03, 0.92, "dashed: random dependency links", transform=a.transAxes, fontsize=8)
panel(a, "(d)")
name = "fig_sim_v11_row.pdf" if ROW else "fig_sim_v11.pdf"
fig.savefig(f"{OUT}/{name}")
print("saved", name)
plt.rcdefaults(); paper_style.use()                   # Fig. 4 is a one-column figure

# ====================================================================== Fig. 4: thresholds against sigma
fig, a = plt.subplots(figsize=(3.4, 3.15))
a.axvspan(0.35, 0.40, color="0.86", lw=0, zorder=0)
ss = np.linspace(0.02, 0.98, 97)
a.plot(ss, [C_schulman(s) for s in ss], color=GREY, ls=":", label=r"$C_{\rm S}$ (Schulman)")
a.plot(ss, [C_star(s) for s in ss], color=RED, ls="--", label=r"$C_{*}$ (Theorem 1)")       # C_* above C_** in the legend
a.plot(ss, [C_closed(s) for s in ss], color=RED, ls="-.", lw=0.9, label=r"$C_{**}$ (Theorem 2)")
gg = sorted(GORI)
a.plot(gg, [GORI[s] for s in gg], "^", color=GREY, ms=3, label=r"$C_c^{\rm single}$ (Gori et al.)")
a.plot(ss, [C_bethe_mux(s)[0] for s in ss], color="k", lw=1.0, label=r"$C^{\rm tree}_{\rm mul}$")
st = sorted(PLAIN)
a.plot([s for s in st if 0 < s < 1], [PLAIN[s] for s in st if 0 < s < 1], "o", color=ORANGE, ms=3.2, mfc="none",
       mew=0.7, label=r"Method 2, $N=10^5$")
a.plot(sig, [mux[s] for s in sig], "s", color=BLUE, ms=2.8, label=r"Method 1, $N=2^{16}$–$2^{20}$")
rs, rc = [], []
for f in sorted(glob.glob("noi_ring_s*_N20.npz")):
    z = np.load(f)
    rs.append(float(re.search(r"_s([\d.]+)_", f).group(1))); rc.append(z["C2"][z["I2"].mean(0).argmax()])
a.plot(rs, rc, "D", color=ORANGE, ms=1.9, label=r"Method 2, $N\approx10^6$")
a.set_xlim(0, 1); a.set_ylim(0, 1.2); a.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
label(a, x=r"$\sigma$", y="threshold value of $C$")
# order of the legend: Method 1, then Method 2 with N = 10^5, then Method 2 with N ~ 10^6
H, Lb = a.get_legend_handles_labels()
order = [i for i, s in enumerate(Lb) if "Method" not in s] + [i for i, s in enumerate(Lb) if "Method 1" in s] \
    + [i for i, s in enumerate(Lb) if "Method 2" in s and "10^5" in s] + [i for i, s in enumerate(Lb) if "Method 2" in s and "10^6" in s]
assert sorted(order) == list(range(len(Lb)))
a.legend([H[i] for i in order], [Lb[i] for i in order], loc="upper left", fontsize=7.5, ncol=2, columnspacing=0.6,
         handlelength=1.5, handletextpad=0.5, borderaxespad=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_thresholds_v11.pdf")
print("saved fig_thresholds_v11.pdf")
