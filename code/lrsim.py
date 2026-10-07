"""
Simulation tools for one-dimensional long-range percolation, p_r = C / r^(1+sigma),
on a ring of N sites, and for two such layers coupled by dependency links.

Ring regularisation: the bond probability between two sites at ring distance r is the
sum over periodic images,  p_r^(N) = C * sum_n |r + n N|^-(1+sigma).  This keeps the mean
degree equal to 2 C zeta(1+sigma) (1 - N^-(1+sigma)) and removes the large truncation error
that a bare cut-off at r = N/2 produces when sigma is small.

Every bond carries a uniform mark m in (0,1).  Bonds are generated once at C = CMAX and
the graph at any C <= CMAX is the sub-graph {m < C/CMAX}.  This is exact because p_r is
proportional to C, and it makes every observable monotone in C for a fixed sample.
"""
import math
import numpy as np
import numba as nb
from scipy.special import zeta as hzeta


def ring_probabilities(N, sigma, C):
    """p[r] for r = 1..N/2 (index 0 unused), image-summed, and number of pairs per class."""
    s = 1.0 + sigma
    half = N // 2
    r = np.arange(1, half + 1, dtype=np.float64)
    a = r / N
    w = N ** (-s) * (hzeta(s, a) + hzeta(s, 1.0 - a + (a == 1.0)))
    if N % 2 == 0:                       # r = N/2 : the two images are the same pair
        w[-1] = 2.0 * N ** (-s) * hzeta(s, 0.5)
    p = np.zeros(half + 1)
    p[1:] = np.minimum(1.0, C * w)
    return p


@nb.njit(cache=True)
def _gen(N, p, seed):
    np.random.seed(seed)
    half = N // 2
    est = 0.0
    for r in range(1, half + 1):
        est += p[r]
    cap = int(N * est * 1.05 + 10.0 * math.sqrt(N * est) + 1000)
    U = np.empty(cap, np.int32)
    V = np.empty(cap, np.int32)
    M = np.empty(cap, np.float32)
    m = 0
    for r in range(1, half + 1):
        pr = p[r]
        ns = N
        if 2 * r == N:
            ns = half
        if pr >= 1.0:
            for i in range(ns):
                U[m] = i; V[m] = (i + r) % N; M[m] = np.random.random(); m += 1
            continue
        if pr <= 0.0:
            continue
        lg = math.log1p(-pr)
        i = -1
        while True:
            u = 1.0 - np.random.random()
            i += int(math.log(u) / lg) + 1
            if i >= ns:
                break
            U[m] = i; V[m] = (i + r) % N; M[m] = np.random.random(); m += 1
    return U[:m].copy(), V[:m].copy(), M[:m].copy()


def gen_layer(N, sigma, Cmax, seed):
    """Bonds (U,V) of one layer at C = Cmax with uniform marks M."""
    return _gen(N, ring_probabilities(N, sigma, Cmax), seed)


@nb.njit(cache=True)
def _find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


@nb.njit(cache=True)
def giant(N, U, V, M, thr, allowed, out):
    """Largest cluster of the graph {bonds with M < thr} restricted to `allowed` sites.
    Writes its indicator into `out`, returns its size."""
    parent = np.arange(N)
    size = np.ones(N, np.int64)
    for e in range(U.shape[0]):
        if M[e] < thr:
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
def mutual_giant(N, UA, VA, MA, UB, VB, MB, thr, dep):
    """Cascade of failures for two layers.  Node i of A and node i of B are a dependency
    pair when dep[i] is True (a random matching is obtained by relabelling layer B before
    the call).  Returns (size of the functional giant component of A, number of iterations)."""
    FA = np.ones(N, np.bool_)
    FB = np.ones(N, np.bool_)
    RA = np.ones(N, np.bool_)
    RB = np.ones(N, np.bool_)
    sA_old = -1; sB_old = -1
    it = 0
    while True:
        it += 1
        for i in range(N):
            RA[i] = (not dep[i]) or FB[i]
        sA = giant(N, UA, VA, MA, thr, RA, FA)
        for i in range(N):
            RB[i] = (not dep[i]) or FA[i]
        sB = giant(N, UB, VB, MB, thr, RB, FB)
        if sA == sA_old and sB == sB_old:
            break
        sA_old = sA; sB_old = sB
        if sA < 3 or sB < 3:
            break
    return sA, it


@nb.njit(cache=True)
def bond_sweep(N, U, V, order):
    """Newman-Ziff sweep: bonds are added in the order `order`; returns the largest
    cluster size after each addition."""
    parent = np.arange(N)
    size = np.ones(N, np.int64)
    big = np.empty(order.shape[0], np.int64)
    b = 1
    for k in range(order.shape[0]):
        e = order[k]
        ru = _find(parent, U[e]); rv = _find(parent, V[e])
        if ru != rv:
            if size[ru] < size[rv]:
                ru, rv = rv, ru
            parent[rv] = ru
            size[ru] += size[rv]
            if size[ru] > b:
                b = size[ru]
        big[k] = b
    return big


@nb.njit(cache=True)
def site_sweep(N, U, V, M, thr, seed):
    """Sites are occupied one at a time in random order on the graph {M < thr};
    returns big[n] = largest cluster when n+1 sites are occupied."""
    np.random.seed(seed)
    deg = np.zeros(N + 1, np.int64)
    for e in range(U.shape[0]):
        if M[e] < thr:
            deg[U[e] + 1] += 1; deg[V[e] + 1] += 1
    for i in range(N):
        deg[i + 1] += deg[i]
    pos = deg[:N].copy()
    adj = np.empty(deg[N], np.int32)
    for e in range(U.shape[0]):
        if M[e] < thr:
            u = U[e]; v = V[e]
            adj[pos[u]] = v; pos[u] += 1
            adj[pos[v]] = u; pos[v] += 1
    order = np.random.permutation(N)
    occ = np.zeros(N, np.bool_)
    parent = np.arange(N)
    size = np.ones(N, np.int64)
    big = np.empty(N, np.int64)
    b = 1
    for n in range(N):
        s = order[n]
        occ[s] = True
        for k in range(deg[s], deg[s + 1]):
            t = adj[k]
            if occ[t]:
                ru = _find(parent, s); rv = _find(parent, t)
                if ru != rv:
                    if size[ru] < size[rv]:
                        ru, rv = rv, ru
                    parent[rv] = ru
                    size[ru] += size[rv]
                    if size[ru] > b:
                        b = size[ru]
        big[n] = b
    return big


def relabel(U, V, N, rng):
    """Apply a uniformly random permutation to the node labels of a layer."""
    perm = rng.permutation(N).astype(np.int32)
    return perm[U], perm[V]
