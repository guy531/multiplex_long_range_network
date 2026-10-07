"""
Check that the thresholds found with N = 10^5 on the plain ring (links only up to r = N/2, no periodic
images) differ from those of the ring with periodic images because of the missing long links.

Runs the multiplex cascade with BOTH regularisations at the same N and records the position of
the largest jump of P(C) per sample (same estimator as experiment2.py).
Usage: python verify_truncation.py N nreal sigma1 sigma2 ...
"""
import sys, time
import numpy as np
from multiprocessing import Pool
from scipy.special import zeta
from lrsim import _gen, ring_probabilities, mutual_giant
from experiment2 import largest_jump
from bethe import C_bethe_mux


def bare_probabilities(N, sigma, C):
    """p_r = C / r^(1+sigma) for r = 1..N/2, nothing beyond (plain ring, as in the runs of Method 2 with N = 10^5)."""
    half = N // 2
    r = np.arange(1, half + 1, dtype=np.float64)
    p = np.zeros(half + 1)
    p[1:] = np.minimum(1.0, C * r ** (-(1.0 + sigma)))
    return p


def one(args):
    sigma, N, kind, Clo, Chi, seed = args
    pfun = bare_probabilities if kind == "bare" else ring_probabilities
    UA, VA, MA = _gen(N, pfun(N, sigma, Chi), 2 * seed + 1)
    UB, VB, MB = _gen(N, pfun(N, sigma, Chi), 2 * seed + 2)
    dep = np.ones(N, np.bool_)
    f = lambda C: mutual_giant(N, UA, VA, MA, UB, VB, MB, C / Chi, dep)[0] / N
    return largest_jump(f, Clo, Chi)


if __name__ == "__main__":
    N = int(sys.argv[1]); nreal = int(sys.argv[2]); sigmas = [float(s) for s in sys.argv[3:]]
    with Pool(7) as pool:
        for s in sigmas:
            kinds = ["bare"] if s == 0 else ["bare", "image"]
            for kind in kinds:
                t0 = time.time()
                Clo, Chi = (0.05, 0.3) if s < 0.15 else (0.12, 0.45)
                res = np.array(pool.map(one, [(s, N, kind, Clo, Chi, k) for k in range(nreal)]))
                m = res.mean(0); e = res.std(0) / np.sqrt(nreal)
                half = N // 2
                Htrunc = np.sum(np.arange(1, half + 1, dtype=float) ** (-(1 + s)))
                print(f"sigma={s} N={N} {kind:5s}: C_j={m[0]:.4f}+-{e[0]:.4f} jump={m[1]:.3f}  "
                      f"[truncated sum={Htrunc:.4f}, zeta={zeta(1 + s) if s > 0 else float('inf'):.4f}]  {time.time() - t0:.0f}s",
                      flush=True)
