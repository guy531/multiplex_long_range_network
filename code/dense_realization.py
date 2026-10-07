"""P_inf^mul(C) of one realization of the image-summed ring, resolved step by step (for the figures of the paper).
The realization is number k of the runs in noi_ring_s*_N*.npz (same seeds), so its largest jump is known.
The curve is a non-decreasing step function; an interval of C is halved as long as P changes over it by more
than DP and it is wider than (Chi - Clo)/RES, so the points concentrate on the steps.

With the argument `method1` the script also records the values of C that Method 1 of the paper evaluates on
this realization: 25 equally spaced values, and then the middle points of the intervals with the largest
increase, until the interval is narrower than 1e-4 * Chi.

Usage: python dense_realization.py sigma log2N k Clo Chi [method1]
Output: dense_s{sigma}_N{L}_k{k}.npz  (C, P, NOI sorted by C;  with method1 also Cg, Pg, Cb, Pb, jump)"""
import paths                                    # the result files are read and written in ../data
import sys, time, heapq
import numpy as np
from noi_open import Sample, seed_of

DP, RES, N0 = 0.004, 4000, 61

sigma = float(sys.argv[1]); L = int(sys.argv[2]); k = int(sys.argv[3])
Clo, Chi = float(sys.argv[4]), float(sys.argv[5])
method1 = len(sys.argv) > 6 and sys.argv[6] == "method1"
N = 1 << L
t0 = time.time()
f = Sample("ring", N, sigma, seed_of("ring", sigma, k))
print(f"sample built ({time.time()-t0:.0f}s)", flush=True)
val = {}


def g(C):
    C = float(C)
    if C not in val:
        val[C] = f(C)
    return val[C]


extra = {}
if method1:
    Cg = np.linspace(Clo, Chi, 25)
    Pg = np.array([g(C)[0] for C in Cg])
    heap = [(-(Pg[i + 1] - Pg[i]), Cg[i], Cg[i + 1], Pg[i], Pg[i + 1]) for i in range(24)]
    heapq.heapify(heap)
    Cb, Pb = [], []
    while heap[0][2] - heap[0][1] > 1e-4 * Chi * 1.0001:
        d, a, b, fa, fb = heapq.heappop(heap)
        m = 0.5 * (a + b); fm = g(m)[0]
        Cb.append(m); Pb.append(fm)
        heapq.heappush(heap, (-(fm - fa), a, m, fa, fm))
        heapq.heappush(heap, (-(fb - fm), m, b, fm, fb))
    d, a, b, fa, fb = heap[0]
    extra = dict(Cg=Cg, Pg=Pg, Cb=np.array(Cb), Pb=np.array(Pb), jump=np.array([0.5 * (a + b), -d, fa, fb, b - a]))
    print(f"method 1: jump {-d:.4f} at C={0.5*(a+b):.5f}, {len(Cb)} halvings ({time.time()-t0:.0f}s)", flush=True)

wmin = (Chi - Clo) / RES
grid = np.linspace(Clo, Chi, N0)
stack = [(grid[i], grid[i + 1]) for i in range(N0 - 1)]
while stack:
    a, b = stack.pop()
    if g(b)[0] - g(a)[0] > DP and b - a > wmin:
        m = 0.5 * (a + b)
        stack.append((a, m)); stack.append((m, b))
    if len(val) % 50 == 0:
        print(f"{len(val)} values, {len(stack)} intervals left ({time.time()-t0:.0f}s)", flush=True)
C = np.array(sorted(val)); P = np.array([val[c][0] for c in C]); I = np.array([val[c][1] for c in C])
np.savez(f"dense_s{sigma:.3f}_N{L}_k{k}.npz", C=C, P=P, NOI=I, **extra)
print(f"done: {len(C)} values in {time.time()-t0:.0f}s; largest step {np.diff(P).max():.3f} at C={C[np.diff(P).argmax()]:.5f}")
