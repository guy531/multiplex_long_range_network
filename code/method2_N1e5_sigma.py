"""Method 2 (NOI maximum) at N = 10^5 for one more value of sigma, with the code of the other runs of
Method 2 at this size, unchanged.
The functions (links on the plain ring up to distance N/2, union-find, cascade and its iteration count) are
executed verbatim from ../method2_N1e5/numofiterations2.py.  The loop below is the loop of that file with
the parameters of the runs in ../method2_N1e5/noi_N100000.csv: N = 10^5, 30 realizations, C = 0.01, 0.02, ...,
1.00, new networks for every C and every realization.
Usage: python method2_N1e5_sigma.py 0.35   ->   noi_plainring_s0.350_N100000.csv / .npz"""
import paths                                    # the result files are read and written in ../data
import os, sys, time
import numpy as np
from numba import njit

SRC = os.path.join(paths.N1E5, "numofiterations2.py")
src = open(SRC, encoding="utf-8").read()
# find, union, generate_bonds, reFindFamily, find_LCC, cascade: the definitions of that file, as they are
exec(src[src.index("@njit"):src.index("sigma_values = np.array")])

N = 100000
Avg_times = 30
C_values = np.linspace(0.01, 1.00, 100)
times_C = len(C_values)
r = float(sys.argv[1])

steps_all = np.zeros((Avg_times, times_C))
lcc_all = np.zeros((Avg_times, times_C))
t0 = time.time()
for q in range(Avg_times):
    for i in range(times_C):
        # dtype given explicitly: on Windows np.arange(N) is int32, which the compiled functions reject
        A = np.arange(N, dtype=np.int64)
        B = np.arange(N, dtype=np.int64)
        parent_A = np.arange(N, dtype=np.int64)
        parent_B = np.arange(N, dtype=np.int64)
        size_A = np.ones(N, dtype=np.int64)
        size_B = np.ones(N, dtype=np.int64)

        bonds_A = generate_bonds(C_values[i], r, N, parent_A, size_A)
        bonds_B = generate_bonds(C_values[i], r, N, parent_B, size_B)

        lcc_size, steps = cascade(A, B, parent_A, parent_B, size_A, size_B, bonds_A, bonds_B)
        steps_all[q, i] = steps
        lcc_all[q, i] = lcc_size / N
    print(f"realization {q + 1}/{Avg_times} done [{time.time() - t0:.0f}s]", flush=True)

AVG_cascade_steps = steps_all.mean(0)
critical_index = np.argmax(AVG_cascade_steps)
print(f"-> Critical C for sigma={r:.2f} is C_c = {C_values[critical_index]:.4f} "
      f"(Max Steps: {AVG_cascade_steps[critical_index]:.2f})")

name = f"noi_plainring_s{r:.3f}_N{N}"
np.savez(name + ".npz", C=C_values, steps=steps_all, lcc=lcc_all)
with open(name + ".csv", "w", encoding="utf-8") as f:
    f.write("Sigma,C,Avg_Cascade_Steps\n")
    for c_val, steps in zip(C_values, AVG_cascade_steps):
        f.write(f"{r:.4f},{c_val:.4f},{steps:.4f}\n")
print("saved", name + ".csv")
