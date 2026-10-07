"""Method 2, and Method 1 on the same realizations, on the ring with periodic images (noi_open.py with `ring`):
N = 2^20 with 10 realizations, and then N = 2^14, 2^16 and 2^18 with 20 realizations each, for all sigma.
Results: ../data/noi_ring_s*_N*.npz.  Logs: ../data/noi_ring_N*.log.  The run with N = 2^20 takes hours."""
import os, subprocess, sys

here = os.path.dirname(os.path.abspath(__file__))
SIG = ["0.1", "0.2", "0.3", "0.35", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]
for L, nreal in [("20", "10"), ("14", "20"), ("16", "20"), ("18", "20")]:
    with open(os.path.join(here, "..", "data", f"noi_ring_N{L}.log"), "w", encoding="utf-8") as log:
        subprocess.run([sys.executable, "-u", "noi_open.py", "ring", L, nreal] + SIG, cwd=here,
                       stdout=log, stderr=subprocess.STDOUT)
