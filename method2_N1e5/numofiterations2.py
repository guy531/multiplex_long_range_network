from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
from numba import njit

# --- פרמטרי הריצה ---
N = 100000
Avg_times = 30
times_C = 200

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"cascade_results_N{N}_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

@njit
def find(parent, i):
    root = i
    while parent[root] != root:
        root = parent[root]
    while parent[i] != root:
        next_node = parent[i]
        parent[i] = root
        i = next_node
    return root


@njit
def union(parent, size, i, j):
    root_i = find(parent, i)
    root_j = find(parent, j)
    if root_i != root_j:
        if size[root_i] < size[root_j]:
            parent[root_i] = root_j
            size[root_j] += size[root_i]
        else:
            parent[root_j] = root_i
            size[root_i] += size[root_j]


@njit
def generate_bonds(C, r, N, parent, size):
    # עושים שימוש ברשימת Numba יעילה
    bonds_list = []
    for i in range(1, N // 2 + 1):
        j = -1
        jump_p = C / ((i) ** (1 + r))
        jump_prob = min(jump_p, 1.0)
        while True:
            step = np.random.geometric(jump_prob) - 1
            j = j + step + 1
            if N % 2 == 0 and i == N // 2 and j >= N // 2:
                break
            if j >= N or j < 0:
                break
            bonds_list.append((j, (j + i) % N))
            union(parent, size, j, (j + i) % N)

    n_bonds = len(bonds_list)
    bonds = np.empty((n_bonds, 2), dtype=np.int64)
    for k in range(n_bonds):
        bonds[k, 0] = bonds_list[k][0]
        bonds[k, 1] = bonds_list[k][1]

    return bonds
@njit
def reFindFamily(N, bonds, arrAB):
    parent = np.arange(N)
    size = np.ones(N, dtype=np.int64)
    for i in range(N):
        if arrAB[i] == -1:
            size[i] = 0
    for k in range(len(bonds)):
        union(parent, size, bonds[k, 0], bonds[k, 1])
    return parent, size


@njit
def find_LCC(parent, size, N):
    max_size_index = np.argmax(size)
    in_lcc = np.zeros(N, dtype=np.bool_)
    if size[max_size_index] == 0:
        return in_lcc
    for i in range(N):
        if find(parent, i) == max_size_index:
            in_lcc[i] = True
    return in_lcc


@njit
def cascade(A, B, parent_A, parent_B, size_A, size_B, bonds_A, bonds_B):
    N = len(A)
    cascade_steps = 0

    while True:
        LCC1 = find_LCC(parent_A, size_A, N)
        changeA = 0
        for i in range(N):
            if B[i] != -1 and not LCC1[i]:
                B[i] = -1
                A[i] = -1
                changeA += 1

        if changeA != 0:
            mask = (A[bonds_A[:, 0]] != -1) & (A[bonds_A[:, 1]] != -1)
            bonds_A = bonds_A[mask]
            mask = (B[bonds_B[:, 0]] != -1) & (B[bonds_B[:, 1]] != -1)
            bonds_B = bonds_B[mask]
            parent_A, size_A = reFindFamily(N, bonds_A, A)
            parent_B, size_B = reFindFamily(N, bonds_B, B)

        LCC2 = find_LCC(parent_B, size_B, N)
        changeB = 0
        for j in range(N):
            if B[j] != -1 and not LCC2[j]:
                B[j] = -1
                A[j] = -1
                changeB += 1

        if changeB != 0:
            mask = (A[bonds_A[:, 0]] != -1) & (A[bonds_A[:, 1]] != -1)
            bonds_A = bonds_A[mask]
            mask = (B[bonds_B[:, 0]] != -1) & (B[bonds_B[:, 1]] != -1)
            bonds_B = bonds_B[mask]
            parent_A, size_A = reFindFamily(N, bonds_A, A)
            parent_B, size_B = reFindFamily(N, bonds_B, B)

        if changeA == 0 and changeB == 0:
            break
        cascade_steps += 1

    return np.max(size_A), cascade_steps

sigma_values = np.array([0.4])
C_values = np.linspace(0.380, 0.382, times_C)

critical_C_results = []
max_iterations_results = []

all_cascade_steps = {}

for r in sigma_values:
    print(f"\nEvaluating for Sigma = {r:.2f}")
    AVG_cascade_steps = np.zeros(times_C)

    for q in range(Avg_times):
        for i in range(times_C):
            A = np.arange(N)
            B = np.arange(N)
            parent_A = np.arange(N)
            parent_B = np.arange(N)
            size_A = np.ones(N, dtype=np.int64)
            size_B = np.ones(N, dtype=np.int64)

            bonds_A = generate_bonds(C_values[i], r, N, parent_A, size_A)
            bonds_B = generate_bonds(C_values[i], r, N, parent_B, size_B)

            lcc_size, steps = cascade(
                A, B, parent_A, parent_B, size_A, size_B, bonds_A, bonds_B
            )

            AVG_cascade_steps[i] += steps / Avg_times

    all_cascade_steps[r] = AVG_cascade_steps.copy()

    critical_index = np.argmax(AVG_cascade_steps)
    C_crit = C_values[critical_index]
    max_steps = AVG_cascade_steps[critical_index]

    critical_C_results.append(C_crit)
    max_iterations_results.append(max_steps)

    print(
        f"-> Critical C for r={r:.2f} is C_c = {C_crit:.4f} (Max Steps: {max_steps:.2f})"
    )

# saving results: iterations
steps_filename = os.path.join(output_dir, "cascade_steps_per_C_final.csv")
with open(steps_filename, "w", encoding="utf-8") as f:
    f.write("Sigma,C,Avg_Cascade_Steps\n")
    for r in sigma_values:
        for c_val, steps in zip(C_values, all_cascade_steps[r]):
            f.write(f"{r:.4f},{c_val:.4f},{steps:.4f}\n")

print(f"\nפירוט האיטרציות נשמר בהצלחה בתיקייה: {steps_filename}")

# -saving results
output_filename = os.path.join(output_dir, "critical_results_send.csv")
with open(output_filename, "w", encoding="utf-8") as f:
    f.write("Sigma,Critical_C,Max_Cascade_Steps\n")
    for sig, c_crit, steps in zip(
        sigma_values, critical_C_results, max_iterations_results
    ):
        f.write(f"{sig:.4f},{c_crit:.4f},{steps:.2f}\n")

print(f"תוצאות ה-Critical C נשמרו בהצלחה בתיקייה: {output_filename}")

# plotting the graph
plt.figure(figsize=(9, 6))

for r in sigma_values:
    plt.plot(
        C_values,
        all_cascade_steps[r],
        label=f"$\sigma = {r:.2f}$",
        linewidth=2,
        marker="o",
        markersize=3,
    )

plt.xlabel(r"Parameter $C$", fontsize=12)
plt.ylabel(r"Average Cascade Steps (Iterations)", fontsize=12)
plt.title(r"Cascade Iterations vs. $C$ for Different $\sigma$ Values", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(title=r"$\sigma$ Values", fontsize=11)

plt.tight_layout()

# שמירת תמונת הגרף בתוך תיקיית התוצאות
plot_filename = os.path.join(output_dir, "cascade_plot.png")
plt.savefig(plot_filename, dpi=300)
print(f"הגרף נשמר בהצלחה כתמונה בתיקייה: {plot_filename}")

plt.show()