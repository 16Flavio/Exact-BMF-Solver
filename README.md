# bmf: an exact solver for Boolean matrix factorization

Given a binary matrix `X` (`m x n`) and a rank `r`, find binary `W` (`m x r`)
and `H` (`r x n`) minimizing

    || X - min(1, WH) ||^2,

where `min(1, WH)` is the Boolean product, the union of `r` rank-one
rectangles. Entries are binary, so the objective is the number of entries where
`X` and the reconstruction differ. Entries of `X` may be missing, in which case
the error is counted on the known entries only (binary matrix completion).

`bmf` starts from a given factorization, or from the empty one, and returns a
certified interval `[LB, UB]` on the optimal error, or a proof of optimality
when the two meet. The lower bound comes from a combinatorial branch and bound
on column patterns: it owes nothing to any linear relaxation, every bound being
the exact cost of a subproblem. It is written in Rust with no external
dependency.

## Building and running

    cargo build --release
    cargo test  --release                # acceptance tests, brute force included

    target/release/bmf -i data/zoo.txt -r 5 --init solutions/zoo_r5.txt
    target/release/bmf -i data/zoo.txt -r 5                     # from the empty factorization
    target/release/bmf -i data/zoo.txt -r 5 -t 0 --bb-time 0    # until proven; Ctrl+C prints [LB, UB]
    target/release/bmf -i data/zoo.txt -r 5 --json              # machine-readable output

`.cargo/config.toml` sets `target-cpu=native`, so binaries and timings are
specific to the machine that built them. Always measure in `--release`.

Main options: `-r` rank, `--init` initial factorization (in the format written
by `-o`), `-t` total budget in seconds (0 for none), `--bb-time` budget of the
branch and bound, `--threads` number of threads, `-o` write `W` and `H`,
`--without residuals,ladder,inherit,empty,grid` switch components off,
`--margin` the watcher's margin (0 never cuts), `--order reverse` reverse the
column order, `--mip` and `--wcnf` write the compact integer program or a
weighted MaxSAT encoding for an outside solver. `bmf --help` lists them all.

Input format: an optional first line `m n`, then `m` rows of `n` entries, with
or without spaces; a missing entry is written `?` or `nan`.

## How it works

A rank-`r` factorization is determined by its `r` column patterns (the rows of
`H`): `W` then follows exactly, each row picking the subset of patterns that
minimizes its own error. The branch and bound fixes one column of `H` at a
time, `2^r` branches per column, and bounds a node by the exact cost of the
fixed columns plus a certified lower bound on the optimum of the free ones.

- A **chain of residual bounds**, computed before the search by solving blocks
  of columns exactly, charges the free columns; it is what proves the low
  ranks.
- A **single cost table per level**, read through a min-zeta transform on the
  subset lattice, bounds the `2^r` children of a node in `O(r 2^r m)` instead
  of `O(4^r m)`.
- **Symmetry breaking** keeps the patterns in decreasing lexicographic order.
- A **ladder of targets**: exhausting the tree at a threshold `T` below the
  best known error certifies `OPT >= T`, and that tree is much smaller than the
  one of the proof. A **watcher** stops attempts that will not finish.
- A **grid bound**, the isolation bound summed over a partition of the matrix
  into blocks, is the only bound left at rank 10.

The upper bound is recomputed from the returned factorization on the original
matrix, independently of the search.

## Results

Benchmark: zoo (101 x 17), heart (242 x 22), lymp (148 x 44), apb
(105 x 105), and the four instances with missing entries of Kolomvakis,
Bobille, Vandaele and Gillis (arXiv 2512.03807): tumor (339 x 24, 670
missing), hepatitis (155 x 38, 334), audio (226 x 92, 899), votes (435 x 16,
392). Machine: Intel i7-12650H, 16 threads, Windows 11. Every solver is run
without an initial solution (the empty factorization for bmf) and with the
stored one from [`solutions/`](solutions/) (a MIP start for Gurobi, a hint for
CP-SAT). bmf has 60 s on 16 threads unless stated otherwise.

### Sixteen certified optima

Median proof time over five runs (600 s budget at rank 3). Every certificate
is reproduced on one thread, most also in reversed column order or with the
node bound alone; Gurobi confirms zoo at ranks 1 and 2 and heart and lymp at
rank 1, CP-SAT zoo at rank 1. No route ever returned a different value. To our
knowledge these are the first certified optima on these instances at rank 2
and above.

| instance | r | optimum | from the stored start | from the empty start |
| --- | --- | --- | --- | --- |
| zoo | 1, 2, 3, 4, 5 | 415, 271, 196, 156, 125 | < 1, < 1, < 1, 4, 20 s | < 1, < 1, < 1, 8 s, not in 60 s |
| heart | 1, 2 | 1421, 1185 | < 1, 10 s | < 1, 10 s |
| lymp | 1, 2 | 1381, 1174 | < 1, 2 s | < 1, 3 s |
| apb | 1 | 826 | 6 s | 6 s |
| tumor (missing) | 2, 3 | 1351, 1148 | < 1, 5 s | < 1, 9 s |
| hepatitis (missing) | 2 | 1264 | 3 s | 3 s |
| audio (missing) | 2 | 1411 | 5 s | 5 s |
| votes (missing) | 2, 3 | 1246, 1033 | 1, 3 s | 1, 2 s |

### The 24 main cells

Ranks 2, 5 and 10, as in the heuristic literature. "Published": best value of
Kolomvakis et al. (2023, 2025) over all their methods at 300 s; "stored": error
of the starting factorization (bold where it improves on the published value).
Median over five runs of 60 s; "proven": runs that certify the optimum.

| instance | r | published | stored | empty start: LB / UB | stored start: LB / UB | proven (empty / stored) |
| --- | --- | --- | --- | --- | --- | --- |
| zoo | 2 | 271 | 271 | 271 / 271 | 271 / 271 | 5 / 5 |
| zoo | 5 | 125 | 125 | 124 / 247 | 125 / 125 | 0 / 5 |
| zoo | 10 | 39 | 39 | 14 / 320 | 14 / 39 | 0 / 0 |
| heart | 2 | 1185 | 1185 | 1185 / 1185 | 1185 / 1185 | 5 / 5 |
| heart | 5 | 736 | 736 | 363 / 1215 | 367 / 736 | 0 / 0 |
| heart | 10 | 419 | 419 | 88 / 1099 | 88 / 419 | 0 / 0 |
| lymp | 2 | 1174 | 1174 | 1174 / 1174 | 1174 / 1174 | 5 / 5 |
| lymp | 5 | 944 | **933** | 463 / 1324 | 462 / 933 | 0 / 0 |
| lymp | 10 | 694 | **669** | 92 / 1564 | 92 / 669 | 0 / 0 |
| apb | 2 | 776 | 776 | 737 / 812 | 737 / 776 | 0 / 0 |
| apb | 5 | 677 | 677 | 355 / 803 | 355 / 677 | 0 / 0 |
| apb | 10 | 566 | 566 | 208 / 805 | 208 / 566 | 0 / 0 |
| tumor | 2 | 1351 | 1351 | 1351 / 1351 | 1351 / 1351 | 5 / 5 |
| tumor | 5 | 943 | 943 | 468 / 1113 | 469 / 943 | 0 / 0 |
| tumor | 10 | 510 | 510 | 100 / 1416 | 100 / 510 | 0 / 0 |
| hepatitis | 2 | 1264 | 1264 | 1264 / 1264 | 1264 / 1264 | 5 / 5 |
| hepatitis | 5 | 990 | **984** | 391 / 1894 | 393 / 984 | 0 / 0 |
| hepatitis | 10 | 749 | **695** | 60 / 1989 | 60 / 695 | 0 / 0 |
| audio | 2 | 1411 | 1411 | 1411 / 1411 | 1411 / 1411 | 5 / 5 |
| audio | 5 | 1035 | 1035 | 504 / 1893 | 501 / 1035 | 0 / 0 |
| audio | 10 | 749 | 749 | 199 / 2109 | 202 / 749 | 0 / 0 |
| votes | 2 | 1246 | 1246 | 1246 / 1246 | 1246 / 1246 | 5 / 5 |
| votes | 5 | 701 | 701 | 503 / 1175 | 504 / 701 | 0 / 0 |
| votes | 10 | 225 | **218** | 21 / 1064 | 21 / 218 | 0 / 0 |

The lower bound hardly depends on the start; the upper bound found by the tree
alone is poor at high rank, so a good heuristic solution and the branch and
bound are complementary. With ten minutes from the stored start, tumor and
votes at rank 3 are proven; zoo at rank 6 ends at [90, 102], heart at rank 3
at [943, 976], apb at rank 2 at [746, 776].

### Against generic exact solvers

Gurobi on the compact integer program (16 threads, 120 s), RC2 core-guided
MaxSAT (PySAT, one thread, 120 s) and CP-SAT (OR-tools, 8 workers, 600 s).
Lower bound reached without / with the initial solution; "p": proven; RC2
takes no start.

| instance | r | Gurobi | RC2 | CP-SAT | bmf, 16 threads | bmf, 1 thread |
| --- | --- | --- | --- | --- | --- | --- |
| zoo | 2 | 271 p / 271 p | 254 | 253 / 250 | 271 p / 271 p | 271 p / 271 p |
| zoo | 5 | 0 / 0 | 42 | 23 / 24 | 124 / 125 p | 114 / 113 |
| heart | 2 | 338 / 352 | 731 | 713 / 713 | 1185 p / 1185 p | 1185 p / 1185 p |
| lymp | 2 | 142 / 192 | 693 | 692 / 686 | 1174 p / 1174 p | 1174 p / 1174 p |
| apb | 1 | 677 / 683 | 704 | 663 / 663 | 826 p / 826 p | 826 p / 826 p |
| tumor | 2 | 491 / 463 | 847 | 834 / 834 | 1351 p / 1351 p | 1351 p / 1351 p |
| hepatitis | 2 | 161 / 168 | 773 | 787 / 781 | 1264 p / 1264 p | 1264 p / 1264 p |

The linear relaxation of the compact program is 0 for every `r >= 2`, which is
why the initial solution does not help it.

### The synthetic family of Gunluk, Hauser and Kovacs

Their family (Mathematics of Operations Research 49(2), 2024), regenerated:
planted Boolean rank 10, `n x 20` for `n` in 20, 35, 50, 75 or 50 % zeros, 0
or 5 % noise, five instances per configuration. bmf and their compact program
(CIP, under Gurobi) on the same instances, 600 s and four threads each.
Instances certified out of 60, without / with the initial solution:

| r | bmf | CIP |
| --- | --- | --- |
| 2 | 60 / 60 | 37 / 37 |
| 5 | 19 / 21 | 4 / 1 |
| 10 | 22 / 30 | 19 / 30 |

At rank 10 the 30 noise-free instances have optimum 0, which a start of error
0 certifies by itself; no noisy instance is certified by either solver.

### Scaling

Planted matrices of rank `r` with half of their entries ones, each entry then
flipped with probability `e`; three instances per point, 300 s, from the
planted factorization. Largest value at which all three instances are proven:

| r | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- |
| columns `n` (`m` = 200, `e` = 5 %) | 80 | 40 | 30 | 30 | 10 |
| noise `e` (200 x 30) | 20 % | 15 % | 5 % | 5 % | 0 % |

The number of rows is not the limit: with 20 columns and 5 % noise, every
instance up to rank 4 is proven at every height up to 12 800 rows, with a
median proof time of at most 21 s.

### Ablation

With the residual chain switched off, five of the eight proofs at ranks 2 and
5 are lost; without the ladder the bound drops at rank 5, and without the grid
bound every rank-10 bound falls to almost nothing.

## Reproducing the measurements

    cargo build --release
    bash experiments/probe.sh          # heart r=2: about 10 s at full speed
    bash experiments/run.sh            # every table above, about 45 h in all
    ONLY="1 3" bash experiments/run.sh # only some blocks

`run.sh` writes its CSV files to `experiments/results/`. Its blocks are: 1, the
24 main cells and the optima (five runs each way); 2, the other routes to each
certificate; 3, Gurobi, CP-SAT and RC2, and bmf on one thread; 4, the
ablation; 5, the synthetic family; 6, the ten-minute runs and the watcher's
margin; 7, the scaling study. Blocks 3 and 5 need `gurobi_cl` on the `PATH`
(or in `GUROBI_CL`) and Python 3 with `pip install ortools python-sat`; the
others need only the binary. Every block resumes where it stopped, and a
machine-speed probe before each block waits while the machine is throttled.

Time-limited values move with the machine's speed, and multi-threaded runs are
not bit-reproducible since the work split between threads depends on timing:
compare several runs as a distribution, not single values.

## Starting points

[`solutions/`](solutions/) holds the best known factorization of every
measured cell, in the format of `-o`: `solutions/<instance>_r<k>.txt` for the
benchmark, `solutions/synth/` for the synthetic family, and
`solutions/index.csv` with the error of each file. They were found by the
evolutionary algorithm EvoMF (F. Drogo, N. Gillis, A. Vandaele, "An
evolutionary algorithm for discrete factorization problems", EUSIPCO 2026).

## Data

`data/` holds the eight benchmark matrices. zoo, heart (SPECT Heart), lymp
(Lymphography) and the matrices behind tumor (Primary Tumor), hepatitis, audio
(Audiology Standardized) and votes (Congressional Voting Records) come from the
UCI Machine Learning Repository (CC BY 4.0), in the binarization of Kovacs,
Gunluk and Hauser (AAAI 2021); the lymphography and primary tumor data were
donated by M. Zwitter and M. Soklic (Institute of Oncology, Ljubljana). apb is
the political books network of V. Krebs (unpublished, http://www.orgnet.com/),
free for scientific use. The four `_w_missing` files are those of the GitLab
repository `ckolomvakis/boolean-matrix-factorization-ip-and-heuristics`. The
synthetic instances are regenerated by `experiments/generate_synth.py` and
`experiments/generate_scaling.py`.

## License

MIT, see [`LICENSE`](LICENSE). The matrices in `data/` stay under the terms of
their sources.
