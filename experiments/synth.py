"""The synthetic family of Gunluk, Hauser and Kovacs: bmf and their compact
integer program (CIP, written by `bmf --mip`, solved by Gurobi), each from the
empty and from the stored factorization. 600 s per run, four processes of four threads each, for both solvers.
Resumes: a row already in the CSV is not replayed. Stops starting new runs when
experiments/results/STOP exists.
usage, from the repository root:
    python experiments/generate_synth.py
    python experiments/synth.py [bmf binary] [gurobi_cl]
Writes experiments/results/synth.csv (instance,r,solver,start,lb,ub,proven,time).
"""
import csv
import glob
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

EXE = sys.argv[1] if len(sys.argv) > 1 else 'target/release/bmf'
GUROBI = sys.argv[2] if len(sys.argv) > 2 else 'gurobi_cl'
D = 'experiments/results'
OUT = os.path.join(D, 'synth.csv')
TMP = os.path.join(D, 'tmp', 'synth')
LIMIT, THREADS, WORKERS = 600, 4, 4


def done():
    if not os.path.exists(OUT):
        return set()
    return {(r['instance'], r['r'], r['solver'], r['start']) for r in csv.DictReader(open(OUT))}


def run(task):
    name, r, solver, start = task
    if os.path.exists(os.path.join(D, 'STOP')):
        return None
    data = f'experiments/data/synth/{name}.txt'
    sol = f'solutions/synth/{name}_r{r}.txt'
    if solver == 'bmf':
        args = [EXE, '-i', data, '-r', str(r), '-t', str(LIMIT), '--bb-time', '0',
                '--threads', str(THREADS), '--json']
        if start == 'start':
            args += ['--init', sol]
        out = subprocess.run(args, capture_output=True, text=True).stdout
        j = json.loads(out[out.index('{'):])
        return [name, r, solver, start, j['lb'], j['ub'], str(j['proven_optimal']).lower(), j['time_s']]
    lp = os.path.join(TMP, f'{name}_r{r}.lp')
    mst = os.path.join(TMP, f'{name}_r{r}.mst')
    write = [EXE, '-i', data, '-r', str(r), '--mip', lp]
    read = [GUROBI, f'TimeLimit={LIMIT}', f'Threads={THREADS}', 'LogFile=']
    if start == 'start':
        write += ['--init', sol, '--mst', mst]
        read += [f'InputFile={mst}']
    subprocess.run(write, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    log = subprocess.run(read + [lp], capture_output=True, text=True).stdout
    for p in (lp, mst):
        if os.path.exists(p):
            os.remove(p)
    obj = re.search(r'Best objective ([-0-9.e+]+)', log)
    bound = re.search(r'best bound ([-0-9.e+]+)', log)
    secs = re.findall(r'in ([0-9.]+) seconds', log)
    return [name, r, 'cip', start,
            round(float(bound.group(1))) if bound else '',
            round(float(obj.group(1))) if obj else '',
            'true' if 'Optimal solution found' in log else 'false',
            secs[-1] if secs else '']


def main():
    os.makedirs(TMP, exist_ok=True)
    seen = done()
    names = sorted(os.path.basename(p)[:-4] for p in glob.glob('experiments/data/synth/*.txt'))
    tasks = [(n, r, s, st) for r in (2, 5, 10) for n in names
             for s, st in (('bmf', 'none'), ('bmf', 'start'), ('cip', 'none'), ('cip', 'start'))
             if (n, str(r), s, st) not in seen]
    new = not os.path.exists(OUT)
    with open(OUT, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['instance', 'r', 'solver', 'start', 'lb', 'ub', 'proven', 'time'])
        with ThreadPoolExecutor(WORKERS) as pool:
            for row in pool.map(run, tasks):
                if row is None:
                    continue
                w.writerow(row)
                f.flush()
                print('synth', *row, flush=True)


if __name__ == '__main__':
    main()
