# Percolation on interdependent one-dimensional long-range networks

Code and data of the paper

> A. Sharabi, A. Bashan, S. V. Buldyrev, and G. Amit,
> *Percolation on interdependent one-dimensional long-range networks* (2026).

The paper studies a multiplex network of two one-dimensional lattices. In each layer, two nodes at a
distance `r` are linked with probability `C / r^(1+sigma)`, with `0 < sigma < 1`, and node `i` of one layer
depends on node `i` of the other layer. The repository contains the simulations of the mutual giant
component, the analytical bounds and the tree approximation, the result files, and the scripts that make
the figures and the table of the paper from these files.

## Contents

| folder | contents |
|---|---|
| `code/` | Python scripts: simulations, analytical formulas, figures, table |
| `data/` | result files of the simulations (all the numbers of the paper are computed from them) |
| `method2_N1e5/` | code and results of Method 2 with `N = 10^5` on the plain ring (without periodic images) |
| `matlab/` | MATLAB functions for the bounds of Theorems 1 and 2 and for the tree approximation (no toolbox needed) |
| `figures/` | the four figures of the paper, as written by the scripts |

## Requirements

Python 3.9 with numpy 1.21, scipy 1.7, numba 0.55, matplotlib 3.5 and pandas 1.4 (the versions that were
used; see `requirements.txt`). The figures use the Times New Roman font; without it matplotlib uses its
default serif font.

Every script reads and writes its result files in `data/`, and the figure scripts write to `figures/`,
from whichever folder they are started (see `code/paths.py`).

## Figures, table and numbers of the paper

These scripts use the stored result files and take a few seconds.

```
cd code
python fig1_model.py              # Fig. 1: schematic of the model
python fig2_tree.py               # Fig. 2: tree approximation
python fig3_fig4_simulations.py   # Fig. 3 (simulations) and Fig. 4 (thresholds against sigma)
python table1.py                  # Table I, and the ratios quoted in the text
python largest_jump_vs_N.py       # largest jump against N, and its decay exponents
python critical_noi_fit.py        # growth of the number of iterations at the threshold with N
python finite_size.py             # numbers of the paragraph on the finite-size effect
```

The table of the paper gives the largest jump with two digits; `table1.py` prints three.

## Simulations

The simulations run on 7 processes (`Pool(7)` in the scripts). The run time grows from seconds for
`N = 2^12` to hours for `N = 2^20`.

| script | purpose | result files |
|---|---|---|
| `lrsim.py` | links of a layer on a ring with periodic images, clusters (union-find), cascade of failures | (module) |
| `bethe.py` | Schulman's bound, the bound `C_*` of Theorem 1, the tree approximation | (module) |
| `experiment2.py SIGMA LOG2N NREAL 1.0` | Method 1: largest jump of the order parameter in each realization, for the multiplex and for random dependency links | `jump_s*_q1.00_N*.npy` |
| `experiment.py SIGMA 16 28 1.0 CLO CHI 41` | order parameter on a grid of `C` (circles of Fig. 2(c)); the grids were 0.19-0.27, 0.4-0.6 and 0.6-0.9 for `sigma` = 0.2, 0.5 and 0.8 | `exp1_s*_q1.00_N16.npz` |
| `noi_open.py ring LOG2N NREAL SIGMA ...` | Method 2 (number of iterations of the cascade, NOI) and Method 1 on the same realizations; `run_noi_ring.py` runs all the sizes | `noi_ring_s*_N*.npz` |
| `dense_realization.py SIGMA LOG2N K CLO CHI [method1]` | order parameter of realization `K` of the runs above, resolved step by step (Fig. 3(a) and (c)) | `dense_s*_N*_k*.npz` |
| `critical_noi.py LOG2N NREAL SIGMA ...` | NOI at the threshold of each realization; `critical_noi_rest.py` completed the run with `N = 2^20` | `critnoi_s*_N*.npy` |
| `monotonicity_check.py LOG2N NREAL SIGMA ...` | largest decrease of the order parameter along `C` in single realizations | (printed) |
| `verify_truncation.py 100000 NREAL SIGMA ...` | Method 1 on the plain ring and on the ring with periodic images, same size | (printed) |
| `method2_N1e5_sigma.py 0.35` | Method 2 with `N = 10^5` on the plain ring for one value of `sigma`, with the functions of `method2_N1e5/numofiterations2.py` | `noi_plainring_s*_N100000.csv`, `.npz` |

Method 1 was run for `N = 2^12` to `2^20` with 28 realizations (14 at the largest sizes), and Method 2 on the
ring with periodic images for `N = 2^14` to `2^18` with 20 realizations and `N = 2^20` with 10.

## Result files

- `jump_s{sigma}_q1.00_N{log2N}.npy`: one row per realization. Columns: position of the largest jump,
  its height, the order parameter just above it, number of evaluations, width of the last interval, for
  the multiplex; then the same five for random dependency links.
- `exp2_s0.200_q1.00_N18.npy`: the same quantities (position, height, order parameter above the jump, for the
  multiplex and for random dependency links) from an earlier version of `experiment2.py`, with a fixed
  number of 16 halvings. It is used for `sigma = 0.2` and `N = 2^18`.
- `exp1_s{sigma}_q1.00_N16.npz`: `Cgrid`, and per realization the order parameter (`mu_id` for the multiplex,
  `mu_rd` for random dependency links) and the number of iterations (`it_id`, `it_rd`).
- `noi_ring_s{sigma}_N{log2N}.npz`: order parameter (`P1`, `P2`) and NOI (`I1`, `I2`) per realization on a
  coarse grid `C1` (step 0.01) and on a fine grid `C2` (step 0.001 around the maximum of the NOI);
  `jump`: position, height, order parameter above the jump, evaluations and final width of the largest jump
  of each realization.
- `dense_s{sigma}_N{log2N}_k{k}.npz`: `C`, `P`, `NOI` of one realization; with `method1` also the values
  evaluated by Method 1 (`Cg`, `Pg`: the 25 initial values; `Cb`, `Pb`: the values added by the halving)
  and its result (`jump`).
- `critnoi_s{sigma}_N{log2N}.npy`: one row per realization, the columns are listed in `critical_noi.py`.
  `critnoi_N20_part1.log` is the log of the first part of the run with `N = 2^20`, which
  `critical_noi_rest.py` reads.
- `noi_plainring_s{sigma}_N100000.csv`: mean NOI against `C` for `N = 10^5` on the plain ring (the `.npz`
  files have the values of each realization). `sigma = 0.35` is used in the paper; 0.3 and 0.4 repeat two of
  the runs of `method2_N1e5/` as a check.
- `method2_N1e5/noi_N100000.csv` and `thresholds_N100000.csv`: mean NOI against `C`, and the position of its
  maximum, for `N = 10^5` on the plain ring (30 realizations), from `numofiterations2.py`.

The single-layer thresholds in `table1.py` and in Fig. 4 are those of Gori, Michelangeli, Defenu and
Trombettoni, Phys. Rev. E **96**, 012108 (2017).
