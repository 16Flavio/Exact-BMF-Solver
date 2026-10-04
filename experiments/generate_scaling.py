"""Scaling instances: X = W o H of planted Boolean rank r, with every entry
flipped with probability e, along three axes, each crossed with the ranks of
RANKS and INSTANCES draws per point:
  columns  m = 200 rows, e = 5 %,  n in COLS
  rows     n = 20 columns, e = 5 %, m in ROWS
  noise    m = 200, n = 30,          e in NOISES (percent)
The densities follow the "normal" family of Gunluk, Hauser and Kovacs: the
same density p for W and H, such that 1 - (1 - p^2)^r = 1/2, so that half the
entries of W o H are ones whatever the rank.

usage, from the repository root:
  python experiments/generate_scaling.py

Writes experiments/data/scaling/m<m>-n<n>-r<r>-e<e>_<k>.txt (with the `m n`
header) and, next to each, <name>.planted.txt, the planted factorization in
the format of `bmf -o`, usable with --init. Python 3 only; deterministic, the
seed of each instance deriving from its name.
"""
import os
import random
import zlib

D = os.path.join('experiments', 'data', 'scaling')
COLS = [10, 20, 30, 40, 60, 80, 100]
ROWS = [50, 200, 800, 3200, 12800]
NOISES = [0, 5, 10, 15, 20]
RANKS = [2, 3, 4, 5, 6]
INSTANCES = 3


def axes():
    """The points (m, n, e) of each axis, in increasing difficulty."""
    return {'columns': [(200, n, 5) for n in COLS],
            'rows': [(m, 20, 5) for m in ROWS],
            'noise': [(200, 30, e) for e in NOISES]}


def name(m, n, r, e, k):
    return f'm{m}-n{n}-r{r}-e{e}_{k}'


def dump(rows):
    return '\n'.join(' '.join('1' if b else '0' for b in row) for row in rows)


def main():
    os.makedirs(D, exist_ok=True)
    points = {(m, n, r, e, k) for pts in axes().values() for m, n, e in pts
              for r in RANKS for k in range(1, INSTANCES + 1)}
    for m, n, r, e, k in sorted(points):
        base = name(m, n, r, e, k)
        rng = random.Random(zlib.crc32(base.encode()))
        p = (1 - 0.5 ** (1 / r)) ** 0.5
        W = [[rng.random() < p for _ in range(r)] for _ in range(m)]
        H = [[rng.random() < p for _ in range(n)] for _ in range(r)]
        X = [[any(W[i][l] and H[l][j] for l in range(r)) != (rng.random() < e / 100)
              for j in range(n)] for i in range(m)]
        with open(os.path.join(D, base + '.txt'), 'w', encoding='utf-8') as f:
            f.write(f'{m} {n}\n')
            f.write('\n'.join(''.join('1' if x else '0' for x in row) for row in X) + '\n')
        with open(os.path.join(D, base + '.planted.txt'), 'w', encoding='utf-8') as f:
            f.write(f'W {m} {r}\n{dump(W)}\nH {r} {n}\n{dump(H)}\n')
    print(f'{len(points)} instances in {D}')


if __name__ == '__main__':
    main()
