"""Table I of the paper, from the result files in ../data and from the analytical formulas of bethe.py.

Columns
  C_S        Schulman's bound for a single layer, 1 / (2 zeta(1+sigma))
  C_**       closed-form bound of Theorem 2
  C_*        bound of Theorem 1
  C_single   single-layer threshold of Gori et al., Phys. Rev. E 96, 012108 (2017)
  C_tree     tree approximation for the multiplex
  Method 1   mean position of the largest jump at the largest size simulated (jump_s*_q1.00_N*.npy), with its
             standard error, and the mean height of the jump, Delta
  Method 2   position of the maximum of the mean NOI, for N = 10^5 (plain ring, ../method2_N1e5) and for
             N = 2^20 (ring with periodic images, noi_ring_s*_N20.npz)
  random     Method 1 with random dependency links: position and height of the largest jump
The second table gives the ratios that are quoted in the text of the paper.
Usage: python table1.py"""
import paths                                    # the result files are read and written in ../data
import glob, re
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import zeta
from bethe import C_bethe_mux, C_schulman, C_star

SIGMAS = [0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
GORI = {0.1: 0.047685, 0.2: 0.093211, 0.3: 0.140546, 0.35: 0.16610, 0.4: 0.193471, 0.45: 0.22293,
        0.5: 0.25482, 0.55: 0.289410, 0.6: 0.327098, 0.65: 0.368333, 0.7: 0.413752, 0.75: 0.464202,
        0.8: 0.521001, 0.85: 0.586264, 0.9: 0.66408, 0.95: 0.76501}


def C_closed(s):
    """C_**: the value of C at which the left-hand side of Theorem 2 is equal to 1"""
    z = zeta(1 + s)
    f = lambda C: 2 * C * z * (1 - ((1 - C) * (1 - C / -np.expm1(-2 * C * z))) ** z) - 1
    return brentq(f, 1e-4, 0.6)


def se(x):
    return x.std(ddof=1) / np.sqrt(len(x))


# Method 1.  Columns: multiplex [C_jump, jump, P_after], random dependency links [C_jump, jump, P_after]
jump = {}
for f in sorted(glob.glob("jump_s*_q1.00_N*.npy")):
    m = re.match(r"jump_s([\d.]+)_q1.00_N(\d+)\.npy", f)
    jump[(float(m.group(1)), int(m.group(2)))] = np.load(f)[:, [0, 1, 2, 5, 6, 7]]
jump[(0.2, 18)] = np.load("exp2_s0.200_q1.00_N18.npy")

# Method 2 with N = 10^5
tab = pd.read_csv(f"{paths.N1E5}/thresholds_N100000.csv").iloc[:, :2]
PLAIN = dict(zip(np.round(tab.iloc[:, 0], 3), tab.iloc[:, 1]))
for f in sorted(glob.glob("noi_plainring_s*_N100000.csv")):        # sigma = 0.35 (method2_N1e5_sigma.py)
    g = pd.read_csv(f)
    s = round(float(g.Sigma.iloc[0]), 3)
    if s not in PLAIN:
        PLAIN[s] = g.C.values[g.Avg_Cascade_Steps.values.argmax()]

print("sigma   C_S     C_**    C_*    C_single  C_tree |  Method 1 (N)          Delta  | Method 2: 1e5   2^20 |"
      "  random links    Delta")
rows = {}
for s in SIGMAS:
    L = max(L for (t, L) in jump if t == s)
    a = jump[(s, L)]
    z = np.load(f"noi_ring_s{s:.3f}_N20.npz")
    m2 = z["C2"][z["I2"].mean(0).argmax()]
    tree = C_bethe_mux(s)[0]
    rows[s] = (tree, a[:, 0].mean(), a[:, 3].mean())
    print(f"{s:5.2f}  {C_schulman(s):.4f}  {C_closed(s):.4f}  {C_star(s):.4f}  {GORI[s]:.4f}   {tree:.4f} |"
          f"  {a[:, 0].mean():.4f}+-{se(a[:, 0]):.4f} (2^{L})  {a[:, 1].mean():.3f} |"
          f"        {PLAIN[s]:.2f}   {m2:.3f} |  {a[:, 3].mean():.4f}+-{se(a[:, 3]):.4f}  {a[:, 4].mean():.3f}")

print("\nRatios quoted in the text (M1: Method 1; rnd: random dependency links)")
print("sigma   M1/C_single   (M1-C_tree)/M1   (rnd-M1)/M1   (rnd-C_tree)/C_tree")
for s in SIGMAS:
    tree, m1, rnd = rows[s]
    print(f"{s:5.2f}     {m1 / GORI[s]:5.2f}        {100 * (m1 - tree) / m1:+6.1f}%        {100 * (rnd - m1) / m1:+5.1f}%"
          f"          {100 * (rnd - tree) / tree:+5.1f}%")
