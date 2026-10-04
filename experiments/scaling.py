"""Scaling campaign: bmf on the instances of experiments/generate_scaling.py,
one run at a time on every thread, from the planted factorization (start)
and, if asked, from the empty one (none).

Along the columns and noise axes, where difficulty grows with the point, a
point is skipped when no instance of the previous point of the same axis, rank
and start was proven: its runs would only burn the time limit. The rows axis
is run in full, since more rows make each node dearer but the structure
clearer. Resumes: a run already in the CSV is not replayed, and the skips are
recomputed from it. Stops starting new runs when experiments/results/STOP
exists. The machine speed probe (experiments/probe.sh) runs before the first
run and every PROBE_EVERY runs, and the campaign stops if it exceeds
MAX_PROBE seconds, the machine being throttled; every probe is logged in
experiments/results/scaling-probe.log.
usage, from the repository root:
    python experiments/generate_scaling.py
    python experiments/scaling.py [bmf binary] [limit in s] [starts]
    python experiments/scaling.py target/release/bmf 300 start
Writes experiments/results/scaling.csv
(instance,m,n,r,e,k,start,lb,ub,proven,nodes,time).
"""
import csv
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_scaling import D as DATA, INSTANCES, RANKS, axes, name  # noqa: E402

EXE = sys.argv[1] if len(sys.argv) > 1 else 'target/release/bmf'
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 300
STARTS = (sys.argv[3] if len(sys.argv) > 3 else 'start').split(',')
D = 'experiments/results'
OUT = os.path.join(D, 'scaling.csv')
LOG = os.path.join(D, 'scaling-probe.log')
PROBE_EVERY = 20
MAX_PROBE = 13.0
FIELDS = ['instance', 'm', 'n', 'r', 'e', 'k', 'start', 'lb', 'ub', 'proven', 'nodes', 'time']
MONOTONE = ('columns', 'noise')


def probe():
    out = subprocess.run(['bash', 'experiments/probe.sh', EXE], capture_output=True, text=True).stdout
    t = re.search(r'in ([0-9.]+)s', out)
    secs = float(t.group(1)) if t else float('inf')
    with open(LOG, 'a') as f:
        f.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")} {out.strip()}\n')
    print(out.strip(), flush=True)
    return secs


def results():
    if not os.path.exists(OUT):
        return {}
    return {(x['instance'], x['start']): x for x in csv.DictReader(open(OUT))}


def run(m, n, r, e, k, start):
    base = name(m, n, r, e, k)
    args = [EXE, '-i', os.path.join(DATA, base + '.txt'), '-r', str(r),
            '-t', str(LIMIT), '--bb-time', '0', '--json']
    if start == 'start':
        args += ['--init', os.path.join(DATA, base + '.planted.txt')]
    out = subprocess.run(args, capture_output=True, text=True).stdout
    j = json.loads(out[out.index('{'):])
    return [base, m, n, r, e, k, start, j['lb'], j['ub'], str(j['proven_optimal']).lower(),
            j['bb_nodes'], j['time_s']]


def main():
    if probe() > MAX_PROBE:
        sys.exit(f'machine throttled (probe above {MAX_PROBE} s), not starting')
    os.makedirs(D, exist_ok=True)
    new = not os.path.exists(OUT)
    count = 0
    with open(OUT, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(FIELDS)
        # rank first, so that an interrupted campaign has covered every axis
        for r in RANKS:
            for axis, points in axes().items():
                for start in STARTS:
                    previous = None
                    for m, n, e in points:
                        res = results()
                        if axis in MONOTONE and previous is not None and not any(
                                res.get((name(*previous[:2], r, previous[2], k), start), {}).get('proven') == 'true'
                                for k in range(1, INSTANCES + 1)):
                            print(f'{axis} r={r} {start}: m={m} n={n} e={e} skipped', flush=True)
                            continue
                        previous = (m, n, e)
                        for k in range(1, INSTANCES + 1):
                            if (name(m, n, r, e, k), start) in res:
                                continue
                            if os.path.exists(os.path.join(D, 'STOP')):
                                sys.exit('STOP found, stopping; rerun to resume')
                            row = run(m, n, r, e, k, start)
                            w.writerow(row)
                            f.flush()
                            count += 1
                            print(f'{time.strftime("%H:%M:%S")} {row[0]} {start}: [{row[7]}, {row[8]}] '
                                  f'proven {row[9]} in {row[11]} s', flush=True)
                            if count % PROBE_EVERY == 0 and probe() > MAX_PROBE:
                                sys.exit(f'machine throttled (probe above {MAX_PROBE} s), '
                                         'stopping; rerun to resume')
    probe()
    print(f'done, {count} runs')


if __name__ == '__main__':
    main()
