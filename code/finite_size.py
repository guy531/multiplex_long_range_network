"""Numbers of the paragraph "Finite-size effects at small sigma" of the paper.

On a ring of N nodes without periodic images there are no links longer than N/2.  For N = 10^5 the script prints:
  - the share of the missing links in the mean degree, sum_{r > N/2} r^-(1+sigma) / zeta(1+sigma), and its
    approximation (N/2)^-sigma / (sigma zeta(1+sigma));
  - the factor zeta(1+sigma) / sum_{r <= N/2} r^-(1+sigma) by which the threshold of the ring is shifted, if the
    mean degree at the threshold is the same on the ring and in the infinite chain;
  - the thresholds of Method 2 with N = 10^5 (plain ring), the same divided by this factor, and the thresholds
    found with periodic images (Method 2 with N = 2^20, and Method 1);
  - the mean degree of the plain ring at sigma = 0 and C = 0.11, where the NOI maximum was found for N = 10^5.
Usage: python finite_size.py"""
import paths                                    # the result files are read and written in ../data
import glob, re
import numpy as np
import pandas as pd
from scipy.special import zeta

N = 100000
SIGMAS = [0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
r = np.arange(1, N // 2 + 1, dtype=np.float64)

tab = pd.read_csv(f"{paths.N1E5}/thresholds_N100000.csv").iloc[:, :2]
PLAIN = dict(zip(np.round(tab.iloc[:, 0], 3), tab.iloc[:, 1]))
for f in sorted(glob.glob("noi_plainring_s*_N100000.csv")):        # sigma = 0.35 (method2_N1e5_sigma.py)
    g = pd.read_csv(f)
    s = round(float(g.Sigma.iloc[0]), 3)
    if s not in PLAIN:
        PLAIN[s] = g.C.values[g.Avg_Cascade_Steps.values.argmax()]
jump = {}
for f in sorted(glob.glob("jump_s*_q1.00_N*.npy")):
    m = re.match(r"jump_s([\d.]+)_q1.00_N(\d+)\.npy", f)
    jump[(float(m.group(1)), int(m.group(2)))] = np.load(f)[:, 0]
jump[(0.2, 18)] = np.load("exp2_s0.200_q1.00_N18.npy")[:, 0]

print(f"N = {N}")
print("sigma  missing share  (approx.)   factor | Method 2, N=1e5   divided by factor   Method 2, N=2^20   Method 1")
for s in SIGMAS:
    z = zeta(1 + s); part = np.sum(r ** -(1 + s))
    L = max(L for (t, L) in jump if t == s)
    q = np.load(f"noi_ring_s{s:.3f}_N20.npz")
    m2 = q["C2"][q["I2"].mean(0).argmax()]
    print(f"{s:5.2f}     {100 * (1 - part / z):5.1f}%      {100 * (N / 2) ** -s / (s * z):5.1f}%     {z / part:.3f} |"
          f"      {PLAIN[s]:.2f}              {PLAIN[s] * part / z:.3f}              {m2:.3f}         "
          f"{jump[(s, L)].mean():.4f}")
print(f"\nsigma = 0, C = 0.11: mean degree of the plain ring 2 C sum_(r <= N/2) 1/r = {2 * 0.11 * np.sum(1 / r):.2f}"
      "   (two interdependent random networks: 2.4554 at the threshold)")
