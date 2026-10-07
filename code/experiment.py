"""
Experiment 1.  For given sigma, N and coupling q, on a grid of C:
  (a) single layer with random site dilution -> g_N(x, C) = (largest cluster)/(x N)
  (b) direct cascade for two layers with the multiplex matching (i <-> i)
      and with a uniformly random matching.
Usage: python experiment.py sigma log2N nreal q Clo Chi nC
"""
import paths                                    # the result files are read and written in ../data
import sys, time
import numpy as np
from multiprocessing import Pool
from lrsim import gen_layer, giant, mutual_giant, site_sweep, relabel

XG = np.linspace(0.005, 1.0, 400)

def one(args):
    sigma, N, q, Cgrid, seed = args
    Cmax = Cgrid[-1]
    rng = np.random.default_rng(10_000 + seed)
    UA, VA, MA = gen_layer(N, sigma, Cmax, 2 * seed + 1)
    UB, VB, MB = gen_layer(N, sigma, Cmax, 2 * seed + 2)
    UBr, VBr = relabel(UB, VB, N, rng)
    dep = rng.random(N) < q
    nC = len(Cgrid)
    mu_id = np.zeros(nC); mu_rd = np.zeros(nC); it_id = np.zeros(nC); it_rd = np.zeros(nC)
    theta = np.zeros(nC); g = np.zeros((nC, len(XG)))
    idx = np.maximum(1, np.rint(XG * N).astype(np.int64)) - 1
    allow = np.ones(N, np.bool_); out = np.ones(N, np.bool_)
    for k, C in enumerate(Cgrid):
        thr = C / Cmax
        s, it = mutual_giant(N, UA, VA, MA, UB, VB, MB, thr, dep)
        mu_id[k] = s / N; it_id[k] = it
        s, it = mutual_giant(N, UA, VA, MA, UBr, VBr, MB, thr, dep)
        mu_rd[k] = s / N; it_rd[k] = it
        theta[k] = giant(N, UA, VA, MA, thr, allow, out) / N
        big = site_sweep(N, UA, VA, MA, thr, 77 + seed)
        g[k] = big[idx] / (idx + 1.0)
    return mu_id, mu_rd, it_id, it_rd, theta, g

def closure(g, q):
    """largest x with 1 - q + q g(x) >= x ; returns P = x g(x)."""
    d = 1 - q + q * g - XG
    ok = np.where(d >= 0)[0]
    k = ok[-1]
    return XG[k] * g[k], XG[k]

if __name__ == "__main__":
    sigma = float(sys.argv[1]); L = int(sys.argv[2]); N = 1 << L; nreal = int(sys.argv[3]); q = float(sys.argv[4])
    Clo, Chi, nC = float(sys.argv[5]), float(sys.argv[6]), int(sys.argv[7])
    Cgrid = np.linspace(Clo, Chi, nC)
    t0 = time.time()
    with Pool(7) as pool:
        res = pool.map(one, [(sigma, N, q, Cgrid, s) for s in range(nreal)])
    arr = [np.array([r[i] for r in res]) for i in range(6)]
    gm = arr[5].mean(0)
    cl = np.array([closure(gm[k], q) for k in range(nC)])
    np.savez(f"exp1_s{sigma:.3f}_q{q:.2f}_N{L}.npz", sigma=sigma, N=N, q=q, Cgrid=Cgrid, XG=XG,
             mu_id=arr[0], mu_rd=arr[1], it_id=arr[2], it_rd=arr[3], theta=arr[4], g=gm, P_cl=cl[:, 0], x_cl=cl[:, 1])
    print(f"sigma={sigma} q={q} N=2^{L} nreal={nreal} done in {time.time()-t0:.0f}s")
    print("   C      theta    P_closure  P_random  P_multiplex   NOI_rd NOI_id")
    for k, C in enumerate(Cgrid):
        print(f" {C:.4f}  {arr[4][:,k].mean():.4f}   {cl[k,0]:.4f}     {arr[1][:,k].mean():.4f}    {arr[0][:,k].mean():.4f}      {arr[3][:,k].mean():5.1f} {arr[2][:,k].mean():5.1f}")
