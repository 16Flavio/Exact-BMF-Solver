"""CP-SAT (OR-tools) on the same problem, encoded directly on the raw matrix
with its mask, sharing no code with bmf: a generic exact solver to compare
with, and an independent check of the optima that it proves.

Needs OR-tools: pip install ortools

usage, from the repository root:
  python experiments/cpsat.py <file> <rank> [seconds] [workers] [start]
  python experiments/cpsat.py data/zoo.txt 2 600 8 solutions/zoo_r2.txt

The optional start is a factorization in the format of `bmf -o`, given to
CP-SAT as a hint on every variable.

Prints a comment line with the OR-Tools version, then one CSV line:
instance,r,status,objective,bound,time (callers read the last line).
"""
import os, sys, time
import ortools
from ortools.sat.python import cp_model

def read(path):
    lines = [l.split() for l in open(path, encoding='utf-8') if l.strip()]
    if len(lines[0]) == 2 and len(lines) > 1 and len(lines[1]) != 2:
        lines = lines[1:]
    X = []
    for l in lines:
        if len(l) == 1 and len(l[0]) > 1:      # digits with no separator
            l = list(l[0])
        row = []
        for t in l:
            t = t.lower()
            if t in ('1', '1.0', '1.'): row.append(1)
            elif t in ('0', '0.0', '0.'): row.append(0)
            else: row.append(None)             # missing
        X.append(row)
    return X

path, r = sys.argv[1], int(sys.argv[2])
limit = float(sys.argv[3]) if len(sys.argv) > 3 else 600.0
workers = int(sys.argv[4]) if len(sys.argv) > 4 else 8
start = sys.argv[5] if len(sys.argv) > 5 else None

def read_factors(path):
    rows = [l.split() for l in open(path, encoding='utf-8') if l.strip()]
    mw, rw = int(rows[0][1]), int(rows[0][2])
    W = [[int(t) for t in l] for l in rows[1:1 + mw]]
    H = [[int(t) for t in l] for l in rows[2 + mw:2 + mw + rw]]
    return W, H
X = read(path)
m, n = len(X), len(X[0])
mod = cp_model.CpModel()
a = [[mod.NewBoolVar(f"a{i}_{k}") for k in range(r)] for i in range(m)]
b = [[mod.NewBoolVar(f"b{k}_{j}") for j in range(n)] for k in range(r)]
cost = []
y_index, z_index = {}, {}
for i in range(m):
    for j in range(n):
        if X[i][j] is None:
            continue
        ys = []
        for k in range(r):
            y = mod.NewBoolVar(f"y{i}_{k}_{j}")
            y_index[(i, k, j)] = y.Index()
            mod.AddBoolOr([a[i][k].Not(), b[k][j].Not(), y])   # a and b => y
            mod.AddImplication(y, a[i][k])
            mod.AddImplication(y, b[k][j])
            ys.append(y)
        z = mod.NewBoolVar(f"z{i}_{j}")
        z_index[(i, j)] = z.Index()
        mod.AddBoolOr(ys + [z.Not()])                          # z => a witness
        for y in ys:
            mod.AddImplication(y, z)                           # witness => z
        cost.append(z.Not() if X[i][j] == 1 else z)
# Symmetry of the r patterns: the columns of W by decreasing number of ones.
for k in range(r - 1):
    mod.Add(sum(a[i][k] for i in range(m)) >= sum(a[i][k + 1] for i in range(m)))
mod.Minimize(sum(cost))
if start:
    W, H = read_factors(start)
    order = sorted(range(len(H)), key=lambda k: -sum(W[i][k] for i in range(m)))
    W = [[row[k] for k in order] for row in W]
    H = [H[k] for k in order]
    for k in range(len(H), r):                 # pad a smaller rank with empty rectangles
        H.append([0] * n)
        for row in W:
            row.append(0)
    for i in range(m):
        for k in range(r):
            mod.AddHint(a[i][k], W[i][k])
    for k in range(r):
        for j in range(n):
            mod.AddHint(b[k][j], H[k][j])
    for i in range(m):
        for j in range(n):
            if X[i][j] is None:
                continue
            ys = [W[i][k] and H[k][j] for k in range(r)]
            for k in range(r):
                mod.AddHint(mod.GetBoolVarFromProtoIndex(y_index[(i, k, j)]), int(ys[k]))
            mod.AddHint(mod.GetBoolVarFromProtoIndex(z_index[(i, j)]), int(any(ys)))
print(f"# OR-Tools {ortools.__version__}", flush=True)
sol = cp_model.CpSolver()
sol.parameters.max_time_in_seconds = limit
sol.parameters.num_workers = workers
t0 = time.time()
st = sol.Solve(mod)
name = os.path.basename(path).replace('.txt', '')
status = {cp_model.OPTIMAL: 'optimal', cp_model.FEASIBLE: 'feasible', cp_model.INFEASIBLE: 'infeasible',
          cp_model.UNKNOWN: 'unknown', cp_model.MODEL_INVALID: 'invalid'}.get(st, str(st))
obj = int(sol.ObjectiveValue()) if st in (cp_model.OPTIMAL, cp_model.FEASIBLE) else ''
bound = int(sol.BestObjectiveBound()) if st in (cp_model.OPTIMAL, cp_model.FEASIBLE, cp_model.UNKNOWN) else ''
print(f"{name},{r},{status},{obj},{bound},{time.time() - t0:.1f}")
