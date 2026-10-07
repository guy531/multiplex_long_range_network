"""Number of cascade iterations (NOI) at the critical point of each realization, on the image-summed ring.

Zhou et al., Phys. Rev. E 90, 012803 (2014): in randomly interdependent networks the NOI at the critical point
of each realization grows as N^(1/3) (and as N^(1/4) at the mean-field threshold, Buldyrev et al. 2010).
A grid of C cannot test this: the critical window is as narrow as N^(-2/3).  Here the largest jump of
P_inf^mul(C) of every realization is located as in Method 1 of the paper, and the halving is then continued
until the interval of the jump contains a single link (the marks of the links are followed exactly), so that
the two networks on the two sides of the jump differ by one link.

Row per realization:
  [C_jump, jump, P_below, P_above, NOI_below, NOI_above, NOI_max, evaluations, links in the last interval]
NOI_below: the cascade that ends in the collapse, one link before the jump (the quantity of Zhou et al.);
NOI_above: the cascade one link after it; NOI_max: the largest NOI among all the values of C evaluated.

Usage: python critical_noi.py log2N nreal sigma [sigma ...]   ->   critnoi_s{sigma}_N{L}.npy
"""
import paths                                    # the result files are read and written in ../data
import sys, time, heapq
import numpy as np
from multiprocessing import Pool
from lrsim import ring_probabilities, _gen
from noi_open import cascade_noi, C_RING

MAXEVAL = 400


def layer(N, sigma, Cmax, seed):
    U, V, M = _gen(N, ring_probabilities(N, sigma, Cmax), seed)
    o = np.argsort(M, kind="stable")
    return U[o], V[o], M[o]


def one(args):
    sigma, L, k = args
    N = 1 << L
    t0 = time.time()
    Cc = C_RING[round(sigma, 2)]
    Cmax = min(0.999, 1.3 * Cc + 0.02); Clo = 0.7 * Cc
    seed = 90_000 + 1000 * int(round(100 * sigma)) + k
    UA, VA, MA = layer(N, sigma, Cmax, 2 * seed + 1)
    UB, VB, MB = layer(N, sigma, Cmax, 2 * seed + 2)
    best = [0]

    def f(t):                                   # t = C / Cmax; returns (P, NOI, number of links)
        kA = int(np.searchsorted(MA, np.float32(t))); kB = int(np.searchsorted(MB, np.float32(t)))
        n, it = cascade_noi(N, UA, VA, kA, UB, VB, kB)
        best[0] = max(best[0], it)
        return n / N, it, kA + kB

    ts = np.linspace(Clo / Cmax, 1.0, 25)
    v = [f(t) for t in ts]
    heap = [(-(v[i + 1][0] - v[i][0]), ts[i], ts[i + 1], v[i], v[i + 1]) for i in range(24)]
    heapq.heapify(heap)
    nev = 25
    # The marks are float32, and several links can share one mark: they enter together and cannot be separated.
    # Stop when the two ends are the same or neighbouring float32 numbers (without this test the loop repeated
    # the same cascade until MAXEVAL; the result was right, but hundreds of long cascades were wasted).
    while (heap[0][4][2] - heap[0][3][2] > 1 and nev < MAXEVAL
           and np.nextafter(np.float32(heap[0][1]), np.float32(2.0)) < np.float32(heap[0][2])):
        d, a, b, fa, fb = heapq.heappop(heap)
        m = 0.5 * (a + b)
        fm = f(m); nev += 1
        heapq.heappush(heap, (-(fm[0] - fa[0]), a, m, fa, fm))
        heapq.heappush(heap, (-(fb[0] - fm[0]), m, b, fm, fb))
    d, a, b, fa, fb = heap[0]
    row = [0.5 * (a + b) * Cmax, -d, fa[0], fb[0], fa[1], fb[1], best[0], nev, fb[2] - fa[2]]
    return sigma, k, np.array(row, dtype=np.float64), time.time() - t0


if __name__ == "__main__":
    L = int(sys.argv[1]); nreal = int(sys.argv[2]); sigmas = [float(x) for x in sys.argv[3:]]
    t0 = time.time()
    R = {s: np.zeros((nreal, 9)) for s in sigmas}
    with Pool(7) as pool:
        for s, k, row, dt in pool.imap_unordered(one, [(s, L, k) for s in sigmas for k in range(nreal)]):
            R[s][k] = row
            print(f"sigma={s} N=2^{L} k={k}: Cj={row[0]:.5f} D={row[1]:.3f} NOI below/above/max={row[4]:.0f}/{row[5]:.0f}/"
                  f"{row[6]:.0f} nev={row[7]:.0f} links={row[8]:.0f} ({dt:.0f}s, total {time.time()-t0:.0f}s)", flush=True)
    for s in sigmas:
        np.save(f"critnoi_s{s:.3f}_N{L}.npy", R[s])
        r = R[s]
        print(f"== sigma={s} N=2^{L} n={nreal}: D={r[:,1].mean():.3f}  NOI below {r[:,4].mean():.1f}+-{r[:,4].std(ddof=1)/np.sqrt(nreal):.1f}"
              f"  above {r[:,5].mean():.1f}  max {r[:,6].mean():.1f}+-{r[:,6].std(ddof=1)/np.sqrt(nreal):.1f}", flush=True)
    print(f"done in {time.time()-t0:.0f}s")
