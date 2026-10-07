"""
Experiment 2: sample-by-sample size of the largest jump of the mutual giant component.

For one sample (two layers with marked bonds) P(C) is a non-decreasing step function of C.
We locate its largest increase over a C-window of width WREL*Chi by branch and bound:
the interval with the largest increase is bisected until that interval is narrower than
the resolution.  Every other interval then has a smaller increase, so the result is the
largest jump at that resolution.  Recorded per sample and per matching:
    C_jump (position), D (height), P just above the jump, number of evaluations, final width.
A first-order transition has D -> const > 0 as N -> infinity; a continuous one has D -> 0.

Usage: python experiment2.py sigma log2N nreal q [Clo Chi]
"""
import paths                                    # the result files are read and written in ../data
import sys, time, heapq
import numpy as np
from multiprocessing import Pool
from lrsim import gen_layer, mutual_giant, relabel
from bethe import C_bethe_mux

GORI = {0.1: 0.047685, 0.2: 0.093211, 0.3: 0.140546, 0.35: 0.16610, 0.4: 0.193471, 0.45: 0.22293,
        0.5: 0.25482, 0.55: 0.289410, 0.6: 0.327098, 0.65: 0.368333, 0.7: 0.413752, 0.8: 0.521001,
        0.9: 0.66408}
NCOARSE, WREL, MAXEVAL = 25, 1.0e-4, 600


def largest_jump(f, Clo, Chi):
    Cs = np.linspace(Clo, Chi, NCOARSE)
    mus = [f(c) for c in Cs]
    heap = [(-(mus[i + 1] - mus[i]), Cs[i], Cs[i + 1], mus[i], mus[i + 1]) for i in range(NCOARSE - 1)]
    heapq.heapify(heap)
    nev = NCOARSE
    wmin = WREL * Chi
    while heap[0][2] - heap[0][1] > wmin and nev < MAXEVAL:
        d, a, b, fa, fb = heapq.heappop(heap)
        m = 0.5 * (a + b)
        fm = f(m)
        nev += 1
        heapq.heappush(heap, (-(fm - fa), a, m, fa, fm))
        heapq.heappush(heap, (-(fb - fm), m, b, fm, fb))
    d, a, b, fa, fb = heap[0]
    return 0.5 * (a + b), -d, fb, nev, b - a


def one(args):
    sigma, N, q, Clo, Chi, seed = args
    rng = np.random.default_rng(10_000 + seed)
    UA, VA, MA = gen_layer(N, sigma, Chi, 2 * seed + 1)
    UB, VB, MB = gen_layer(N, sigma, Chi, 2 * seed + 2)
    UBr, VBr = relabel(UB, VB, N, rng)
    dep = rng.random(N) < q
    f_id = lambda C: mutual_giant(N, UA, VA, MA, UB, VB, MB, C / Chi, dep)[0] / N
    f_rd = lambda C: mutual_giant(N, UA, VA, MA, UBr, VBr, MB, C / Chi, dep)[0] / N
    return largest_jump(f_id, Clo, Chi) + largest_jump(f_rd, Clo, Chi)


if __name__ == "__main__":
    sigma = float(sys.argv[1]); L = int(sys.argv[2]); N = 1 << L; nreal = int(sys.argv[3]); q = float(sys.argv[4])
    CB = C_bethe_mux(sigma)[0]
    if len(sys.argv) > 5:
        Clo, Chi = float(sys.argv[5]), float(sys.argv[6])
    else:
        Clo = GORI.get(round(sigma, 2), 0.5 * CB); Chi = min(0.999, 1.35 * CB + 0.05)
    t0 = time.time()
    with Pool(7) as pool:
        res = np.array(pool.map(one, [(sigma, N, q, Clo, Chi, s) for s in range(nreal)]))
    np.save(f"jump_s{sigma:.3f}_q{q:.2f}_N{L}.npy", res)
    m = res.mean(0); sd = res.std(0); conv = (res[:, [4, 9]] <= WREL * Chi * 1.0001).mean(0)
    print(f"sigma={sigma} q={q} N=2^{L} n={nreal} [{time.time()-t0:.0f}s] | multiplex: Cj={m[0]:.4f}+-{sd[0]:.4f} "
          f"D={m[1]:.4f}+-{sd[1]:.4f} Pafter={m[2]:.3f} nev={m[3]:.0f} conv={conv[0]:.2f} | random: Cj={m[5]:.4f}+-{sd[5]:.4f} "
          f"D={m[6]:.4f}+-{sd[6]:.4f} Pafter={m[7]:.3f} nev={m[8]:.0f} conv={conv[1]:.2f}", flush=True)
