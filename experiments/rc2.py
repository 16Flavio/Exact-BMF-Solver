"""RC2 (PySAT) on a WCNF. In verbose mode RC2 prints `c cost: X` after each
core: X is the sum of the weights of the cores found, hence a certified lower
bound at that instant. If the loop returns, X is the optimum.

Needs PySAT: pip install python-sat

usage, from the repository root:
  target/release/bmf -i data/zoo.txt -r 2 --wcnf zoo_r2.wcnf
  python experiments/rc2.py zoo_r2.wcnf

Single thread. RC2 has no time limit of its own: experiments/run.sh stops it
after 120 s and keeps the last bound printed.
"""
import sys, time
import pysat
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2
print(f"c PySAT {pysat.__version__}", flush=True)
f = WCNF(from_file=sys.argv[1])
t0 = time.time()
s = RC2(f, solver="g4", adapt=True, exhaust=True, minz=True, verbose=2)
done = s.compute_()
print(f"END {'OPTIMUM' if done else 'cut'} cost {s.cost} t {time.time()-t0:.1f}", flush=True)
