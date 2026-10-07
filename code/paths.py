"""Folders of the repository.  Importing this module makes ../data the working directory, so that every script
reads and writes its result files there, from wherever it is started."""
import os

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
DATA = os.path.join(ROOT, "data")
FIGURES = os.path.join(ROOT, "figures")
N1E5 = os.path.join(ROOT, "method2_N1e5")       # code and results of Method 2 with N = 10^5
os.makedirs(FIGURES, exist_ok=True)
os.chdir(DATA)
