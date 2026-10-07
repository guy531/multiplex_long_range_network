"""Second part of the N = 2^20 run of critical_noi.py.
The first run was stopped after 25 of its 49 realizations: four realizations were repeating the same cascade
(links with identical float32 marks, see the comment in critical_noi.one).  The finished realizations are read
from the log of that run (critnoi_N20_part1.log) and the others are run here, with the corrected stopping rule.
Output: critnoi_s{sigma}_N20.npy, as written by critical_noi.py (rows read from the log have C_jump to 5 and
the jump to 3 decimals; the NOI are integers and are exact)."""
import paths                                    # the result files are read and written in ../data
import re, time
import numpy as np
from multiprocessing import Pool
from critical_noi import one

L, NREAL = 20, 7
SIGMAS = [0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6]
PAT = re.compile(r"sigma=([\d.]+) N=2\^20 k=(\d+): Cj=([\d.]+) D=([\d.]+) NOI below/above/max=(\d+)/(\d+)/(\d+) "
                 r"nev=(\d+) links=(\d+)")

if __name__ == "__main__":
    R = {s: np.full((NREAL, 9), np.nan) for s in SIGMAS}
    for line in open("critnoi_N20_part1.log", encoding="utf-8", errors="replace"):
        m = PAT.match(line)
        if m:
            s, k = float(m.group(1)), int(m.group(2))
            cj, d, nb_, na, nm, nev, links = [float(x) for x in m.groups()[2:]]
            R[s][k] = [cj, d, np.nan, np.nan, nb_, na, nm, nev, links]
    todo = [(s, L, k) for s in SIGMAS for k in range(NREAL) if np.isnan(R[s][k, 0])]
    print(f"{49 - len(todo)} realizations read from the log, {len(todo)} to run: {[(s, k) for s, _, k in todo]}", flush=True)
    t0 = time.time()
    with Pool(7) as pool:
        for s, k, row, dt in pool.imap_unordered(one, todo):
            R[s][k] = row
            print(f"sigma={s} N=2^{L} k={k}: Cj={row[0]:.5f} D={row[1]:.3f} NOI below/above/max={row[4]:.0f}/{row[5]:.0f}/"
                  f"{row[6]:.0f} nev={row[7]:.0f} links={row[8]:.0f} ({dt:.0f}s, total {time.time()-t0:.0f}s)", flush=True)
    for s in SIGMAS:
        np.save(f"critnoi_s{s:.3f}_N{L}.npy", R[s])
        r = R[s]
        print(f"== sigma={s} N=2^{L} n={NREAL}: D={r[:,1].mean():.3f}  NOI below {r[:,4].mean():.1f}+-{r[:,4].std(ddof=1)/np.sqrt(NREAL):.1f}"
              f"  above {r[:,5].mean():.1f}  max {r[:,6].mean():.1f}", flush=True)
    print(f"done in {time.time()-t0:.0f}s")
