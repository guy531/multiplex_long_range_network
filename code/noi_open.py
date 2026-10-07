"""
Method 2 of the paper (maximum of the mean number of cascade iterations, NOI) and Method 1 (largest
jump of the mutual giant component), measured on the same samples.

Boundary conditions:
  open : a chain of N sites, bond (i, i+r) for 0 <= i < N-r with probability C / r^(1+sigma), no wrap-around
  ring : the ring with periodic images of lrsim.py
The paper uses `ring` only.

Every sample is generated once at C = CMAX = 1 with uniform bond marks; the bonds are sorted by mark, so
the graph at C is a prefix of the bond list (exact, because p_r is proportional to C).
The cascade is counted as in numofiterations2.py: A then B make one pass, and a pass in which at least one
node fails adds one iteration (checked against a copy of that routine: identical counts).

Pass 1, per sample: NOI and P on a coarse grid (step 0.01), restricted to +-0.15 around a guess
         (ring: the thresholds of Method 1; open: the same rescaled by the mean degree, see C_guess).
Pass 2, per sample: NOI and P on a fine grid (step 0.001, +-0.015 around the peak of the pass-1 mean NOI),
         then the largest jump of P, refined from the merged grid by branch and bound to width WMIN.

Usage: python noi_open.py bc log2N nreal sigma [sigma ...]
Output: noi_{bc}_s{sigma}_N{L}.npz
"""
import paths                                    # the result files are read and written in ../data
import sys, time, heapq, math
import numpy as np
import numba as nb
from multiprocessing import Pool
from scipy.special import zeta
from lrsim import ring_probabilities, _gen

CMAX = 1.0
WMIN, MAXEVAL = 2.5e-4, 40
HALF1, HALF2, STEP2 = 0.15, 0.015, 0.001
C_RING = {0.1: 0.1153, 0.2: 0.2162, 0.3: 0.3039, 0.35: 0.3442, 0.4: 0.3793, 0.5: 0.4433, 0.6: 0.514,
          0.7: 0.590, 0.8: 0.677, 0.9: 0.780}


def mean_degree_ratio(N, sigma):
    """zeta(1+sigma) / sum_{r<N} (1 - r/N) r^-(1+sigma): infinite-chain mean degree over open-chain mean degree."""
    r = np.arange(1, N, dtype=np.float64)
    return zeta(1 + sigma) / np.sum((1 - r / N) * r ** -(1 + sigma))


def C_guess(bc, N, sigma):
    c = C_RING[round(sigma, 2)]
    return c * mean_degree_ratio(N, sigma) if bc == "open" else c


@nb.njit(cache=True)
def _gen_open(N, sigma, Cmax, seed):
    np.random.seed(seed)
    s = 1.0 + sigma
    est = 0.0
    for r in range(1, N):
        est += min(1.0, Cmax * r ** (-s)) * (N - r)
    cap = int(est * 1.05 + 10.0 * math.sqrt(est) + 1000)
    U = np.empty(cap, np.int32)
    V = np.empty(cap, np.int32)
    M = np.empty(cap, np.float32)
    m = 0
    for r in range(1, N):
        pr = min(1.0, Cmax * r ** (-s))
        ns = N - r
        if pr >= 1.0:
            for i in range(ns):
                U[m] = i; V[m] = i + r; M[m] = np.random.random(); m += 1
            continue
        lg = math.log1p(-pr)
        i = -1
        while True:
            u = 1.0 - np.random.random()
            i += int(math.log(u) / lg) + 1
            if i >= ns:
                break
            U[m] = i; V[m] = i + r; M[m] = np.random.random(); m += 1
    return U[:m].copy(), V[:m].copy(), M[:m].copy()


def layer(bc, N, sigma, seed):
    if bc == "open":
        U, V, M = _gen_open(N, sigma, CMAX, seed)
    else:
        U, V, M = _gen(N, ring_probabilities(N, sigma, CMAX), seed)
    o = np.argsort(M, kind="stable")
    return U[o], V[o], M[o]


@nb.njit(cache=True)
def _find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


@nb.njit(cache=True)
def giant_k(N, U, V, k, allowed, out):
    """Largest cluster of the first k bonds restricted to `allowed`; indicator in `out`, returns size."""
    parent = np.arange(N)
    size = np.ones(N, np.int64)
    for e in range(k):
        u = U[e]; v = V[e]
        if allowed[u] and allowed[v]:
            ru = _find(parent, u); rv = _find(parent, v)
            if ru != rv:
                if size[ru] < size[rv]:
                    ru, rv = rv, ru
                parent[rv] = ru
                size[ru] += size[rv]
    best = -1; bs = 0
    for i in range(N):
        if allowed[i] and parent[i] == i and size[i] > bs:
            bs = size[i]; best = i
    for i in range(N):
        out[i] = allowed[i] and (_find(parent, i) == best)
    return bs


@nb.njit(cache=True)
def cascade_noi(N, UA, VA, kA, UB, VB, kB):
    """Returns (size of the mutual giant component, number of iterations counted as in numofiterations2.py)."""
    alive = np.ones(N, np.bool_)
    FA = np.empty(N, np.bool_)
    n = N
    steps = 0
    while True:
        sA = giant_k(N, UA, VA, kA, alive, FA)
        sB = giant_k(N, UB, VB, kB, FA, alive)
        changed = (n - sA) + (sA - sB)
        n = sB
        if changed == 0:
            break
        steps += 1
    return n, steps


class Sample:
    def __init__(self, bc, N, sigma, seed):
        self.N = N
        self.A = layer(bc, N, sigma, 2 * seed + 1)
        self.B = layer(bc, N, sigma, 2 * seed + 2)

    def __call__(self, C):
        t = np.float32(C / CMAX)
        kA = int(np.searchsorted(self.A[2], t)); kB = int(np.searchsorted(self.B[2], t))
        n, it = cascade_noi(self.N, self.A[0], self.A[1], kA, self.B[0], self.B[1], kB)
        return n / self.N, it


def largest_jump(f, Cs, Ps):
    heap = [(-(Ps[i + 1] - Ps[i]), Cs[i], Cs[i + 1], Ps[i], Ps[i + 1]) for i in range(len(Cs) - 1)]
    heapq.heapify(heap)
    nev = 0
    while heap[0][2] - heap[0][1] > WMIN * 1.0001 and nev < MAXEVAL:
        d, a, b, fa, fb = heapq.heappop(heap)
        m = 0.5 * (a + b)
        fm = f(m)[0]
        nev += 1
        heapq.heappush(heap, (-(fm - fa), a, m, fa, fm))
        heapq.heappush(heap, (-(fb - fm), m, b, fm, fb))
    d, a, b, fa, fb = heap[0]
    return 0.5 * (a + b), -d, fb, nev, b - a


def seed_of(bc, sigma, k):
    return (50_000 if bc == "open" else 70_000) + 1000 * int(round(100 * sigma)) + k


def pass1(args):
    bc, N, sigma, k, grid = args
    t0 = time.time()
    f = Sample(bc, N, sigma, seed_of(bc, sigma, k))
    res = np.array([f(C) for C in grid])
    return sigma, k, res[:, 0], res[:, 1], time.time() - t0


def pass2(args):
    bc, N, sigma, k, grid1, P1, grid2 = args
    t0 = time.time()
    f = Sample(bc, N, sigma, seed_of(bc, sigma, k))
    res = np.array([f(C) for C in grid2])
    Cs = np.concatenate([grid1, grid2]); Ps = np.concatenate([P1, res[:, 0]])
    Cs, i = np.unique(np.round(Cs, 6), return_index=True)
    jump = largest_jump(f, Cs, Ps[i])
    return sigma, k, res[:, 0], res[:, 1], np.array(jump), time.time() - t0


if __name__ == "__main__":
    bc = sys.argv[1]; L = int(sys.argv[2]); N = 1 << L; nreal = int(sys.argv[3])
    sigmas = [float(x) for x in sys.argv[4:]]
    t0 = time.time()
    G1 = {}
    for s in sigmas:
        c = C_guess(bc, N, s)
        G1[s] = np.round(np.arange(max(0.01, round(c - HALF1, 2)), min(1.0, round(c + HALF1, 2)) + 1e-9, 0.01), 2)
        print(f"sigma={s}: guess {c:.4f}, coarse grid {G1[s][0]:.2f}..{G1[s][-1]:.2f}", flush=True)
    P1 = {s: np.zeros((nreal, G1[s].size)) for s in sigmas}
    I1 = {s: np.zeros((nreal, G1[s].size)) for s in sigmas}
    with Pool(7) as pool:
        for s, k, P, I, dt in pool.imap_unordered(pass1, [(bc, N, s, k, G1[s]) for s in sigmas for k in range(nreal)]):
            P1[s][k], I1[s][k] = P, I
            print(f"pass1 sigma={s} k={k} peak NOI {I.max():.0f} at {G1[s][I.argmax()]:.2f} ({dt:.0f}s, total {time.time()-t0:.0f}s)",
                  flush=True)
        G2 = {}
        for s in sigmas:
            j = I1[s].mean(0).argmax()
            if j == 0 or j == G1[s].size - 1:
                print(f"WARNING sigma={s}: coarse NOI peak at the edge of the window", flush=True)
            c1 = G1[s][j]
            G2[s] = np.round(np.arange(c1 - HALF2, c1 + HALF2 + 1e-9, STEP2), 4)
            np.savez(f"noi_{bc}_s{s:.3f}_N{L}.npz", C1=G1[s], P1=P1[s], I1=I1[s])
        P2 = {s: np.zeros((nreal, G2[s].size)) for s in sigmas}
        I2 = {s: np.zeros((nreal, G2[s].size)) for s in sigmas}
        J = {s: np.zeros((nreal, 5)) for s in sigmas}
        tasks = [(bc, N, s, k, G1[s], P1[s][k], G2[s]) for s in sigmas for k in range(nreal)]
        for s, k, P, I, jmp, dt in pool.imap_unordered(pass2, tasks):
            P2[s][k], I2[s][k], J[s][k] = P, I, jmp
            print(f"pass2 sigma={s} k={k} Cj={jmp[0]:.4f} D={jmp[1]:.3f} nev={jmp[3]:.0f} ({dt:.0f}s, total {time.time()-t0:.0f}s)",
                  flush=True)
    for s in sigmas:
        np.savez(f"noi_{bc}_s{s:.3f}_N{L}.npz", C1=G1[s], P1=P1[s], I1=I1[s], C2=G2[s], P2=P2[s], I2=I2[s], jump=J[s])
        c1 = G1[s][I1[s].mean(0).argmax()]; c2 = G2[s][I2[s].mean(0).argmax()]
        print(f"{bc} sigma={s} N=2^{L} n={nreal}: NOI peak coarse {c1:.2f}, fine {c2:.3f} (max NOI {I2[s].mean(0).max():.1f})"
              f" | largest jump Cj={J[s][:,0].mean():.4f}+-{J[s][:,0].std():.4f} D={J[s][:,1].mean():.3f}", flush=True)
    print(f"done in {time.time()-t0:.0f}s")
