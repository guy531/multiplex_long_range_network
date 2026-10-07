"""Is P_inf^mul(C) of one realization non-decreasing in C?
Links are only added as C grows, so the largest mutually connected set cannot shrink.  The cascade, however,
keeps the largest cluster of each layer in turn, and can end in a mutually connected set that is not the largest
one.  Here P is computed on a fine uniform grid of C around the threshold and the largest decrease between
consecutive grid values is recorded, for several realizations (image-summed ring).
Usage: python monotonicity_check.py log2N nreal sigma [sigma ...]"""
import sys
import numpy as np
from multiprocessing import Pool
from noi_open import Sample, seed_of, C_RING

NGRID = 400


def one(args):
    sigma, L, k = args
    f = Sample("ring", 1 << L, sigma, seed_of("ring", sigma, 100 + k))
    Cc = C_RING[round(sigma, 2)]
    C = np.linspace(0.8 * Cc, 1.3 * Cc, NGRID)
    P = np.array([f(c)[0] for c in C])
    d = np.diff(P)
    return sigma, k, d.min(), int((d < 0).sum()), d.max()


if __name__ == "__main__":
    L, nreal = int(sys.argv[1]), int(sys.argv[2]); sigmas = [float(x) for x in sys.argv[3:]]
    with Pool(7) as pool:
        res = pool.map(one, [(s, L, k) for s in sigmas for k in range(nreal)])
    for s in sigmas:
        r = np.array([x[2:] for x in res if x[0] == s])
        print(f"sigma={s} N=2^{L}, {nreal} realizations, {NGRID} values of C each: largest decrease {-r[:,0].min():.4f} "
              f"(median over realizations {-np.median(r[:,0]):.4f}), realizations with a decrease: {(r[:,1] > 0).sum()}, "
              f"mean number of decreases {r[:,1].mean():.1f}; largest increase, mean {r[:,2].mean():.3f}", flush=True)
