"""Synthetic instances after Gunluk, Hauser and Kovacs (
arXiv 2106.13434, appendix B, table 6): X = W o H of Boolean rank 10, n x m
with n in {20, 35, 50} and m = 20, 75 % zeros (sparse) or 50 % (normal), noise
0 % (clean) or 5 % (noisy, each entry flipped with that probability).
The densities of W and H are not given in the paper: the same density p is
taken for both, such that 1 - (1 - p^2)^10 is the wanted density of ones.

usage, from the repository root:
  python experiments/generate_synth.py [instances per configuration, default 5]

Writes experiments/data/synth/<n>-<sparsity>-<noise>_<k>.txt, with the `m n`
header. Needs Python 3 only; the generation is deterministic, the seed of each
instance deriving from its configuration.
"""
import os, random, sys
N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
D = os.path.join('experiments', 'data', 'synth')
os.makedirs(D, exist_ok=True)
KAPPA = 10
for n in (20, 35, 50):
    for sparsity, zeros in (('sparse', 0.75), ('normal', 0.50)):
        # wanted density of ones d = 1 - zeros; (1 - p^2)^kappa = zeros
        p = (1 - zeros ** (1 / KAPPA)) ** 0.5
        for noise, flip in (('clean', 0.0), ('noisy', 0.05)):
            for k in range(1, N + 1):
                rng = random.Random(1000 * n + 100 * (zeros == 0.75) + 10 * (flip > 0) + k)
                m = 20
                W = [[rng.random() < p for _ in range(KAPPA)] for _ in range(n)]
                H = [[rng.random() < p for _ in range(m)] for _ in range(KAPPA)]
                X = [[any(W[i][l] and H[l][j] for l in range(KAPPA)) for j in range(m)] for i in range(n)]
                if flip > 0:
                    X = [[(x != (rng.random() < flip)) for x in row] for row in X]
                name = f"{n}-{sparsity}-{noise}_{k}.txt"
                with open(os.path.join(D, name), 'w', encoding='utf-8') as f:
                    f.write(f"{n} {m}\n")
                    for row in X:
                        f.write("".join('1' if x else '0' for x in row) + "\n")
print(f"{12 * N} instances in {D}")
