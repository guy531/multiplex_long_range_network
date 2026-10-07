"""Growth of the NOI at the critical point of each realization with N (data of critical_noi.py).
For every sigma: mean NOI of the last cascade before the jump (`below`), of the first one after it (`above`) and
of the largest NOI found, for each N, and the exponent x of NOI ~ N^x from a weighted least-squares fit of
ln<NOI> against ln N.  Reference: x = 1/3 for randomly interdependent networks (Zhou et al. 2014).
Usage: python critical_noi_fit.py [Lmin]      (sizes below 2^Lmin are left out of the fit; default 12)"""
import paths                                    # the result files are read and written in ../data
import glob, re, sys
import numpy as np

Lmin = int(sys.argv[1]) if len(sys.argv) > 1 else 12
data = {}
for f in sorted(glob.glob("critnoi_s*_N*.npy")):
    m = re.match(r"critnoi_s([\d.]+)_N(\d+)\.npy", f)
    data[(float(m.group(1)), int(m.group(2)))] = np.load(f)


def fit(Ls, M, E):
    x = np.log(2.0) * np.array(Ls); y = np.log(M); w = (M / E) ** 2          # weights 1/var(ln M)
    A = np.vstack([x, np.ones_like(x)]).T
    cov = np.linalg.inv(A.T @ (A * w[:, None]))
    p = cov @ (A.T @ (w * y))
    chi2 = float(np.sum(w * (y - A @ p) ** 2))
    return p[0], np.sqrt(cov[0, 0]), chi2


for s in sorted({k[0] for k in data}):
    Ls = sorted(L for (t, L) in data if t == s and L >= Lmin)
    print(f"sigma = {s}")
    for name, col in [("below", 4), ("above", 5), ("max", 6)]:
        M = np.array([data[(s, L)][:, col].mean() for L in Ls])
        E = np.array([data[(s, L)][:, col].std(ddof=1) / np.sqrt(len(data[(s, L)])) for L in Ls])
        x, dx, chi2 = fit(Ls, M, E)
        vals = "  ".join(f"2^{L}: {m:6.1f}+-{e:4.1f}" for L, m, e in zip(Ls, M, E))
        print(f"   {name:5s}  {vals}   x = {x:.3f} +- {dx:.3f}  (chi2 = {chi2:.1f}, {len(Ls) - 2} dof)")
    D = "  ".join(f"2^{L}: {data[(s, L)][:, 1].mean():.3f}" for L in Ls)
    print(f"   jump   {D}")
