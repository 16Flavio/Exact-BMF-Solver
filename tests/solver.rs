//! End-to-end tests: file parsing, reduction, the exact subproblem, the
//! bounds, and the branch and bound against brute force on small matrices.

use bmf::io;
use bmf::reduce::Reduced;
use bmf::subproblem::{ao_exact, Patterns};
use bmf::verify;
use bmf::bits::BitVec;
use bmf::rng::Rng;
use std::io::Write;
use std::path::PathBuf;

/// Writes `content` to a file in the temp directory and returns its path.
fn write_tmp(file_name: &str, content: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("bmf_test_{file_name}"));
    let mut f = std::fs::File::create(&p).unwrap();
    f.write_all(content.as_bytes()).unwrap();
    p
}

/// Uniformly random m x n matrix with every entry known.
fn random_matrix(m: usize, n: usize, rng: &mut Rng) -> io::RawMat {
    let mut rows = vec![BitVec::zeros(n); m];
    for i in 0..m {
        for j in 0..n {
            if rng.bool() {
                rows[i].set(j, true);
            }
        }
    }
    io::RawMat {
        name: "random".into(),
        m,
        n,
        rows,
        known: vec![{
            let mut t = BitVec::zeros(n);
            for j in 0..n {
                t.set(j, true);
            }
            t
        }; m],
    }
}

// ---------------------------------------------------------------- parsing and reduction

/// Packed, space-separated and commented files parse to the same matrix.
#[test]
fn reads_both_formats() {
    let packed = write_tmp("packed.txt", "0111\n1011\n1101\n1110\n");
    let spaced = write_tmp("spaced.txt", "4 4\n0 1 1 1\n1 0 1 1\n1 1 0 1\n1 1 1 0\n");
    let commented = write_tmp("commented.txt", "# test\n\n0111\n1011\n1101\n1110\n");
    let a = io::load(&packed).unwrap();
    let b = io::load(&spaced).unwrap();
    let c = io::load(&commented).unwrap();
    assert_eq!((a.m, a.n), (4, 4));
    assert_eq!((b.m, b.n), (4, 4));
    for i in 0..4 {
        for j in 0..4 {
            assert_eq!(a.get(i, j), b.get(i, j));
            assert_eq!(a.get(i, j), c.get(i, j));
            assert_eq!(a.get(i, j), i != j);
        }
    }
}

/// Non-binary values and ragged rows are rejected.
#[test]
fn rejects_non_binary_value() {
    let p = write_tmp("bad.txt", "0121\n1011\n");
    assert!(io::load(&p).is_err());
    let q = write_tmp("ragged.txt", "011\n1011\n");
    assert!(io::load(&q).is_err());
}

/// For random factorizations, the weighted error on the reduced matrix equals
/// the error on the original matrix. This catches mistakes in weight handling.
#[test]
fn reduction_is_consistent() {
    let mut rng = Rng::new(7);
    for trial in 0..200 {
        let (m, n) = (3 + trial % 7, 3 + (trial * 3) % 9);
        let raw = random_matrix(m, n, &mut rng);
        if raw.ones() == 0 {
            continue;
        }
        let red = Reduced::build(&raw);
        let r = 1 + trial % 4;
        let rects: Vec<(BitVec, BitVec)> = (0..r)
            .map(|_| {
                let mut w = BitVec::zeros(red.m);
                let mut h = BitVec::zeros(red.n);
                for i in 0..red.m {
                    if rng.bool() {
                        w.set(i, true);
                    }
                }
                for j in 0..red.n {
                    if rng.bool() {
                        h.set(j, true);
                    }
                }
                (w, h)
            })
            .collect();
        verify::check_reduction(&raw, &red, &rects).unwrap();
    }
}

/// The reduction preserves the optimum: brute force on the original and on the
/// reduced matrix agree.
#[test]
fn reduction_preserves_optimum() {
    let mut rng = Rng::new(11);
    for _ in 0..30 {
        let raw = random_matrix(4, 4, &mut rng);
        if raw.ones() == 0 {
            continue;
        }
        let red = Reduced::build(&raw);
        for r in 1..=2 {
            assert_eq!(
                brute_force(&raw, r),
                brute_force_reduced(&red, r),
                "the reduction changed the optimum"
            );
        }
    }
}

/// Brute force optimum at rank `r`, computed on the reduced matrix.
fn brute_force(raw: &io::RawMat, r: usize) -> u64 {
    let red = Reduced::build(raw);
    brute_force_reduced(&red, r)
}

/// Enumerates every W; for each one the optimal H is solved exactly, column by
/// column, with the zeta transform.
fn brute_force_reduced(red: &Reduced, r: usize) -> u64 {
    let mut best = u64::MAX;
    let total = 1u64 << (red.m * r);
    for code in 0..total {
        // Decode W from the bits of `code`, m bits per column.
        let mut w = vec![BitVec::zeros(red.m); r];
        for k in 0..r {
            for i in 0..red.m {
                if code >> (k * red.m + i) & 1 == 1 {
                    w[k].set(i, true);
                }
            }
        }
        let mut pats = Patterns::new(&w, red.m, &red.rho);
        let mut e = 0u64;
        for j in 0..red.n {
            let (_, err) = pats.best(&red.cols[j], &red.rho);
            e += err * red.gamma[j];
        }
        best = best.min(e);
    }
    best
}

// ---------------------------------------------------------------- fixed-factor subproblem

/// The zeta transform finds the same optimal error as enumerating every subset,
/// and the subset it returns actually achieves that error.
#[test]
fn zeta_matches_brute_force() {
    let mut rng = Rng::new(3);
    for _ in 0..1000 {
        let m = 1 + rng.below(8);
        let r = 1 + rng.below(4);
        let weight: Vec<u64> = (0..m).map(|_| 1 + rng.below(3) as u64).collect();
        let mut w = vec![BitVec::zeros(m); r];
        for k in 0..r {
            for i in 0..m {
                if rng.bool() {
                    w[k].set(i, true);
                }
            }
        }
        let mut x = BitVec::zeros(m);
        for i in 0..m {
            if rng.bool() {
                x.set(i, true);
            }
        }

        let mut pats = Patterns::new(&w, m, &weight);
        let (s, e) = pats.best(&x, &weight);

        // Reference: try every subset.
        let mut exact = u64::MAX;
        for cand in 0..(1u32 << r) {
            let mut err = 0u64;
            for i in 0..m {
                let covered = (0..r).any(|k| cand >> k & 1 == 1 && w[k].get(i));
                if covered != x.get(i) {
                    err += weight[i];
                }
            }
            exact = exact.min(err);
        }
        assert_eq!(e, exact, "the zeta transform does not return the optimum");
        // The returned subset must achieve that error.
        let mut err = 0u64;
        for i in 0..m {
            let covered = (0..r).any(|k| s >> k & 1 == 1 && w[k].get(i));
            if covered != x.get(i) {
                err += weight[i];
            }
        }
        assert_eq!(err, exact, "the returned subset does not match its error");
    }
}

/// Exact alternating optimization never increases the error, and the error it
/// reports matches the returned rectangles.
#[test]
fn alternation_is_monotone() {
    let mut rng = Rng::new(5);
    for _ in 0..50 {
        let raw = random_matrix(6, 6, &mut rng);
        if raw.ones() == 0 {
            continue;
        }
        let red = Reduced::build(&raw);
        let r = 3;
        let mut rects: Vec<(BitVec, BitVec)> = (0..r)
            .map(|_| {
                let mut w = BitVec::zeros(red.m);
                let mut h = BitVec::zeros(red.n);
                for i in 0..red.m {
                    if rng.bool() {
                        w.set(i, true);
                    }
                }
                for j in 0..red.n {
                    if rng.bool() {
                        h.set(j, true);
                    }
                }
                (w, h)
            })
            .collect();
        let before = red.rects_error(&rects);
        let after = ao_exact(&red, &mut rects);
        assert!(
            after <= before,
            "the exact alternation made the error go back up"
        );
        assert_eq!(after, red.rects_error(&rects), "returned error inconsistent");
    }
}

// ---------------------------------------------------------------- J4 - I4

/// J4 - I4 (all ones except the diagonal): Boolean rank 4, isolation number 3,
/// optimum 1 at rank 3. Its LP relaxation is 0, which makes it a useful trap.
#[test]
fn j4_minus_i4() {
    let p = write_tmp("j4i4_test.txt", "0111\n1011\n1101\n1110\n");
    let raw = io::load(&p).unwrap();
    let red = Reduced::build(&raw);
    assert_eq!(
        (red.m, red.n),
        (4, 4),
        "no reduction is possible here"
    );
    // The greedy isolated set must reach the maximum on this instance.
    assert_eq!(max_isolated(&red), 3, "isolation number expected 3");
    let iso = bmf::bounds::isolated_set(&red);
    assert!(is_isolated(&red, &iso), "the returned set is not isolated");
    assert_eq!(iso.len(), 3, "the greedy had to reach the maximum here");
    assert_eq!(
        brute_force_reduced(&red, 3),
        1,
        "optimum expected 1 at rank 3"
    );
    assert_eq!(
        brute_force_reduced(&red, 4),
        0,
        "optimum expected 0 at rank 4"
    );
}

/// True if every element is a one and no two share a row, a column, or a 2x2
/// block of ones.
fn is_isolated(red: &Reduced, e: &[(usize, usize)]) -> bool {
    for (a, &(i1, j1)) in e.iter().enumerate() {
        if !red.rows[i1].get(j1) {
            return false;
        }
        for &(i2, j2) in &e[a + 1..] {
            if i1 == i2 || j1 == j2 || (red.rows[i1].get(j2) && red.rows[i2].get(j1)) {
                return false;
            }
        }
    }
    true
}

/// Size of the largest isolated set, by enumeration over all subsets of ones.
/// Only usable on tiny matrices.
fn max_isolated(red: &Reduced) -> usize {
    let one_cells: Vec<(usize, usize)> = (0..red.m)
        .flat_map(|i| (0..red.n).map(move |j| (i, j)))
        .filter(|&(i, j)| red.rows[i].get(j))
        .collect();
    let mut best = 0;
    for code in 0..(1u32 << one_cells.len()) {
        let e: Vec<(usize, usize)> = (0..one_cells.len())
            .filter(|&t| code >> t & 1 == 1)
            .map(|t| one_cells[t])
            .collect();
        if e.len() > best && is_isolated(red, &e) {
            best = e.len();
        }
    }
    best
}

// ---------------------------------------------------------------- branch and bound

/// The branch and bound proves the brute force optimum, and the factorization
/// rebuilt from its patterns achieves it.
#[test]
fn branch_and_bound_matches_brute_force() {
    let mut rng = Rng::new(31);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    for trial in 0..40 {
        let raw = random_matrix(3 + trial % 5, 3 + (trial * 2) % 6, &mut rng);
        if raw.ones() == 0 {
            continue;
        }
        let red = Reduced::build(&raw);
        for r in 1..=3 {
            let opt = brute_force_reduced(&red, r);
            // Start from a loose upper bound so the search has to find the
            // optimum itself.
            let s = bmf::bb::solve(&red, r, opt + 7, 0, 4, deadline, false, bmf::bb::Settings::default());
            assert!(s.complete, "the tree had to be exhausted within the budget");
            assert_eq!(
                s.lb, opt,
                "the branch and bound at rank {r} does not return the optimum"
            );
            assert_eq!(s.ub, opt, "inconsistent upper bound at rank {r}");
            let rects = bmf::bb::factorize(&red, &s.patterns);
            assert_eq!(
                red.rects_error(&rects),
                opt,
                "the rebuilt factorization does not match the optimum"
            );
            verify::check_reduction(&raw, &red, &rects).unwrap();
        }
    }
}

/// The branch and bound proves optimality of J4 - I4 at rank 3, a case where an
/// LP-based bound gives nothing.
#[test]
fn j4_minus_i4_proven() {
    let p = write_tmp("j4i4_bb.txt", "0111\n1011\n1101\n1110\n");
    let raw = io::load(&p).unwrap();
    let red = Reduced::build(&raw);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    let s = bmf::bb::solve(&red, 3, 4, 0, 4, deadline, false, bmf::bb::Settings::default());
    assert!(s.complete, "the tree had to be exhausted");
    assert_eq!(s.ub, 1, "optimum expected 1 at rank 3");
    assert_eq!(s.lb, 1, "the optimality of J4 - I4 at rank 3 must be proven");
}

// ---------------------------------------------------------------- missing entries

/// Random matrix where each entry is missing with probability 1/5.
fn matrix_with_holes(m: usize, n: usize, rng: &mut Rng) -> io::RawMat {
    let mut rows = vec![BitVec::zeros(n); m];
    let mut known = vec![BitVec::zeros(n); m];
    for i in 0..m {
        for j in 0..n {
            if rng.below(5) != 0 {
                known[i].set(j, true);
                if rng.bool() {
                    rows[i].set(j, true);
                }
            }
        }
    }
    io::RawMat {
        name: "holes".into(),
        m,
        n,
        rows,
        known,
    }
}

/// Brute force optimum on the original matrix, counting errors on known
/// entries only. Independent of the solver code: enumerates every H and lets
/// each row pick its best subset of rows of H.
fn brute_force_holes(x: &io::RawMat, r: usize) -> u64 {
    let mut best = u64::MAX;
    for code in 0..(1u64 << (r * x.n)) {
        let h: Vec<Vec<bool>> = (0..r)
            .map(|k| (0..x.n).map(|j| code >> (k * x.n + j) & 1 == 1).collect())
            .collect();
        let mut e = 0u64;
        for i in 0..x.m {
            // Row i takes the subset of rows of H that minimizes its error.
            let mut min_err = u64::MAX;
            for s in 0..(1usize << r) {
                let mut cost = 0u64;
                for j in 0..x.n {
                    if !x.is_known(i, j) {
                        continue;
                    }
                    let covered = (0..r).any(|k| s >> k & 1 == 1 && h[k][j]);
                    if covered != x.get(i, j) {
                        cost += 1;
                    }
                }
                min_err = min_err.min(cost);
            }
            e += min_err;
        }
        best = best.min(e);
    }
    best
}

/// With missing entries, the branch and bound still proves the brute force
/// optimum, with the error counted on known entries only.
#[test]
fn holes_match_brute_force() {
    let mut rng = Rng::new(17);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    for trial in 0..30 {
        let m = 3 + trial % 3;
        let n = 3 + (trial * 2) % 3;
        let raw = matrix_with_holes(m, n, &mut rng);
        if raw.ones() == 0 {
            continue;
        }
        let red = Reduced::build(&raw);
        for r in 1..=2 {
            let opt = brute_force_holes(&raw, r);
            let s = bmf::bb::solve(&red, r, opt + 5, 0, 4, deadline, false, bmf::bb::Settings::default());
            assert!(s.complete, "the tree had to be exhausted within the budget");
            assert_eq!(
                s.lb, opt,
                "trial {trial} rank {r}: lower bound inconsistent with brute force"
            );
            assert_eq!(s.ub, opt, "trial {trial} rank {r}: upper bound");
            let rects = bmf::bb::factorize(&red, &s.patterns);
            assert_eq!(
                red.rects_error(&rects),
                opt,
                "trial {trial} rank {r}: the rebuilt factorization does not match the optimum"
            );
            verify::check_reduction(&raw, &red, &rects).unwrap();
        }
    }
}

/// On complete matrices, the independent brute force and the branch and bound
/// agree, so the missing-entry handling does not disturb the complete case.
#[test]
fn no_holes_unchanged() {
    let mut rng = Rng::new(23);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    for trial in 0..12 {
        let raw = random_matrix(3 + trial % 3, 3 + (trial * 2) % 3, &mut rng);
        if raw.ones() == 0 {
            continue;
        }
        assert_eq!(raw.missing(), 0, "complete matrix expected");
        let red = Reduced::build(&raw);
        for r in 1..=2 {
            let opt = brute_force_holes(&raw, r);
            let s = bmf::bb::solve(&red, r, opt + 5, 0, 4, deadline, false, bmf::bb::Settings::default());
            assert!(s.complete);
            assert_eq!(s.lb, opt, "trial {trial} rank {r}");
        }
    }
}

// ---------------------------------------------------------------- combinatorial bounds

/// Random matrix whose rows (and columns, if `repeat_cols`) are duplicated one
/// to three times, so the reduced matrix carries weights greater than one.
fn weighted_matrix(m: usize, n: usize, rng: &mut Rng, holes: bool, repeat_cols: bool) -> io::RawMat {
    let base = if holes {
        matrix_with_holes(m, n, rng)
    } else {
        random_matrix(m, n, rng)
    };
    let rep_l: Vec<usize> = (0..m).map(|_| 1 + rng.below(3)).collect();
    let rep_c: Vec<usize> = (0..n)
        .map(|_| if repeat_cols { 1 + rng.below(3) } else { 1 })
        .collect();
    let nn: usize = rep_c.iter().sum();
    let mut rows = Vec::new();
    let mut known = Vec::new();
    for i in 0..m {
        // Repeat each column rep_c[j] times, then the whole row rep_l[i] times.
        let mut row_bits = BitVec::zeros(nn);
        let mut row_known = BitVec::zeros(nn);
        let mut q = 0;
        for j in 0..n {
            for _ in 0..rep_c[j] {
                if base.rows[i].get(j) {
                    row_bits.set(q, true);
                }
                if base.known[i].get(j) {
                    row_known.set(q, true);
                }
                q += 1;
            }
        }
        for _ in 0..rep_l[i] {
            rows.push(row_bits.clone());
            known.push(row_known.clone());
        }
    }
    io::RawMat {
        name: "weighted".into(),
        m: rows.len(),
        n: nn,
        rows,
        known,
    }
}

/// The isolation bound and the grid bound never exceed the brute force optimum
/// on weighted matrices, with and without missing entries.
#[test]
fn combinatorial_bound_below_brute_force() {
    let mut rng = Rng::new(41);
    let mut weighted_runs = 0;
    for trial in 0..40 {
        let holes = trial % 2 == 1;
        // Brute force with holes enumerates H over the original columns, so
        // only rows are repeated in that case to keep it small.
        let raw = weighted_matrix(3 + trial % 4, 3 + (trial * 2) % 4, &mut rng, holes, !holes);
        if raw.ones() == 0 {
            continue;
        }
        let red = Reduced::build(&raw);
        // Count the cases where the reduction produced actual weights.
        if red.rho.iter().any(|&p| p > 1) || red.gamma.iter().any(|&p| p > 1) {
            weighted_runs += 1;
        }
        let rmax = if holes { 2 } else { 3 };
        for r in 1..=rmax {
            let opt = if holes {
                brute_force_holes(&raw, r)
            } else {
                brute_force_reduced(&red, r)
            };
            let iso = bmf::bounds::isolation_bound(&red, r);
            assert!(
                iso <= opt,
                "trial {trial} rank {r}: isolation bound {iso} above the optimum {opt}"
            );
            let deadline = std::time::Instant::now() + std::time::Duration::from_millis(300);
            let grid = bmf::bounds::grid_bound(&red, r, 2, 0, true, deadline, false);
            assert!(
                grid <= opt,
                "trial {trial} rank {r}: grid bound {grid} above the optimum {opt}"
            );
        }
    }
    assert!(weighted_runs >= 20, "the test must run on weighted matrices");
}
