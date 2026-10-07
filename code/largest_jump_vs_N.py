"""The largest jump Delta of the order parameter against the system size N, in numbers (Fig. 3(d) of the paper).

For every sigma: the mean Delta for each N, with its standard error, for the multiplex and for random dependency
links (jump_s*_q1.00_N*.npy, from experiment2.py); the ratio of Delta at the smallest and at the largest size;
and the decay exponent y of Delta ~ N^-y, from a weighted least-squares fit of ln(Delta) against ln(N) over the
sizes N >= 2^14.  Delta tends to a constant at a first-order transition (y = 0) and decreases at a continuous one.
Then, Delta in the runs in which the two methods were compared (noi_ring_s*_N*.npz, from noi_open.py).
Usage: python largest_jump_vs_N.py"""
import paths                                    # the result files are read and written in ../data
import glob, re
import numpy as np

# columns: multiplex [C_jump, jump, P_after], random dependency links [C_jump, jump, P_after]
jump = {}
for f in sorted(glob.glob("jump_s*_q1.00_N*.npy")):
    m = re.match(r"jump_s([\d.]+)_q1.00_N(\d+)\.npy", f)
    jump[(float(m.group(1)), int(m.group(2)))] = np.load(f)[:, [0, 1, 2, 5, 6, 7]]
jump[(0.2, 18)] = np.load("exp2_s0.200_q1.00_N18.npy")


def mean_err(s, L, col):
    a = jump[(s, L)][:, col]
    return a.mean(), a.std() / np.sqrt(len(a))


def decay_exponent(s, col, Lmin=14):
    Ls = [L for L in sizes(s) if L >= Lmin]
    y, e = np.array([mean_err(s, L, col) for L in Ls]).T
    x = np.log(2.0) * np.array(Ls); w = (y / e) ** 2
    X = np.vstack([np.ones_like(x), x]).T
    cov = np.linalg.inv(X.T @ (w[:, None] * X))
    b = cov @ X.T @ (w * np.log(y))
    return -b[1], np.sqrt(cov[1, 1])


def sizes(s):
    return sorted(L for (t, L) in jump if t == s)


for name, col in [("multiplex", 1), ("random dependency links", 4)]:
    print(f"\n{name}: mean largest jump (standard error of the last digits)")
    print("sigma " + "".join(f"   N=2^{L}  " for L in (12, 14, 16, 18, 20)) + " first/last   decay exponent  (1-sigma)/2")
    for s in sorted({k[0] for k in jump}):
        cells = ""
        for L in (12, 14, 16, 18, 20):
            if (s, L) in jump:
                m, e = mean_err(s, L, col)
                cells += f"  {m:.3f}({1000 * e:2.0f}) "
            else:
                cells += "     --    "
        first, last = mean_err(s, sizes(s)[0], col)[0], mean_err(s, sizes(s)[-1], col)[0]
        y, dy = decay_exponent(s, col)
        print(f"{s:5.2f}{cells}    {first / last:5.2f}    {y:+.3f} +- {dy:.3f}      {(1 - s) / 2:.3f}")

print("\nRuns in which the two methods were compared: mean largest jump of the multiplex")
print("sigma    N=2^14   N=2^16   N=2^18   N=2^20")
for s in [0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    print(f"{s:5.2f}  " + "".join(f"   {np.load(f'noi_ring_s{s:.3f}_N{L}.npz')['jump'][:, 1].mean():.3f} "
                                  for L in (14, 16, 18, 20)))
