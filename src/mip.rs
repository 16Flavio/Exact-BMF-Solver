//! Export of the compact integer program in LP format, for an external solver.
//!
//! `a[i][k]` is W, `b[k][j]` is H, `z[i][j]` is the Boolean product, and
//! `y[i][k][j]` linearizes `a[i][k] b[k][j]` with McCormick inequalities:
//!
//! ```text
//! y <= a,  y <= b,  y >= a + b - 1,  y >= 0
//! y[i][k][j] <= z[i][j] <= sum_k y[i][k][j]
//! ```
//!
//! The objective is the weighted error on the known entries of the reduced
//! matrix; missing entries get no variable and no constraint. Pattern
//! permutations are broken by ordering the rows of H by decreasing size.

use crate::reduce::Reduced;
use std::fmt::Write;

/// Returns the model as an LP file. The objective's constant is carried by a
/// variable `un` fixed to one, since not every LP reader accepts a constant.
pub fn write_lp(red: &Reduced, r: usize) -> String {
    let (m, n) = (red.m, red.n);
    let mut s = String::new();
    let _ = writeln!(
        s,
        "\\ boolean matrix factorization of rank {r} on {} reduced to {m}x{n}",
        red.name
    );
    let _ = writeln!(s, "\\ objective: weighted error on the known entries");

    // Total weight of the known ones; covering a one subtracts its weight.
    let mut constant = 0u64;
    for i in 0..m {
        for j in 0..n {
            if red.is_known(i, j) && red.rows[i].get(j) {
                constant += red.weight(i, j);
            }
        }
    }

    let _ = writeln!(s, "Minimize");
    let _ = write!(s, " obj:");
    if constant > 0 {
        let _ = write!(s, " {constant} un");
    }
    for i in 0..m {
        for j in 0..n {
            if !red.is_known(i, j) {
                continue;
            }
            let p = red.weight(i, j);
            if red.rows[i].get(j) {
                let _ = write!(s, " - {p} z_{i}_{j}");
            } else {
                let _ = write!(s, " + {p} z_{i}_{j}");
            }
        }
    }
    let _ = writeln!(s);

    let _ = writeln!(s, "Subject To");
    let mut nc = 0usize;
    for i in 0..m {
        for j in 0..n {
            if !red.is_known(i, j) {
                continue;
            }
            for k in 0..r {
                // McCormick, and z >= y.
                let _ = writeln!(s, " c{nc}: y_{i}_{k}_{j} - a_{i}_{k} <= 0");
                nc += 1;
                let _ = writeln!(s, " c{nc}: y_{i}_{k}_{j} - b_{k}_{j} <= 0");
                nc += 1;
                let _ = writeln!(s, " c{nc}: y_{i}_{k}_{j} - a_{i}_{k} - b_{k}_{j} >= -1");
                nc += 1;
                let _ = writeln!(s, " c{nc}: z_{i}_{j} - y_{i}_{k}_{j} >= 0");
                nc += 1;
            }
            // z <= sum_k y.
            let _ = write!(s, " c{nc}: z_{i}_{j}");
            nc += 1;
            for k in 0..r {
                let _ = write!(s, " - y_{i}_{k}_{j}");
            }
            let _ = writeln!(s, " <= 0");
        }
    }
    // Patterns ordered by decreasing number of columns.
    for k in 0..r.saturating_sub(1) {
        let _ = write!(s, " sym{k}:");
        for j in 0..n {
            let _ = write!(s, " + b_{k}_{j}");
        }
        for j in 0..n {
            let _ = write!(s, " - b_{}_{j}", k + 1);
        }
        let _ = writeln!(s, " >= 0");
    }

    let _ = writeln!(s, "Bounds");
    let _ = writeln!(s, " un = 1");

    let _ = writeln!(s, "Binary");
    let _ = writeln!(s, " un");
    for i in 0..m {
        for k in 0..r {
            let _ = writeln!(s, " a_{i}_{k}");
        }
    }
    for k in 0..r {
        for j in 0..n {
            let _ = writeln!(s, " b_{k}_{j}");
        }
    }
    for i in 0..m {
        for j in 0..n {
            if !red.is_known(i, j) {
                continue;
            }
            let _ = writeln!(s, " z_{i}_{j}");
            for k in 0..r {
                let _ = writeln!(s, " y_{i}_{k}_{j}");
            }
        }
    }
    let _ = writeln!(s, "End");
    s
}

/// Projects a factorization of the original matrix onto the reduced frame by
/// taking, for each merged row and column, the factors of its first member.
/// Returns `(a, b)` with `a[i][k]` for the reduced rows and `b[k][j]` for the
/// reduced columns, the rectangles sorted by decreasing number of columns as
/// the symmetry constraints of `write_lp` require.
pub fn project_start(red: &Reduced, w: &[Vec<bool>], h: &[Vec<bool>]) -> (Vec<Vec<bool>>, Vec<Vec<bool>>) {
    let r = h.len();
    // Factors in the working frame, which is X transposed when `red.transposed`.
    let row_factor = |src: usize, k: usize| if red.transposed { h[k][src] } else { w[src][k] };
    let col_factor = |k: usize, src: usize| if red.transposed { w[src][k] } else { h[k][src] };
    let mut rects: Vec<(Vec<bool>, Vec<bool>)> = (0..r)
        .map(|k| {
            let a = (0..red.m).map(|i| row_factor(red.row_src[i][0], k)).collect();
            let b = (0..red.n).map(|j| col_factor(k, red.col_src[j][0])).collect();
            (a, b)
        })
        .collect();
    rects.sort_by_key(|(_, b)| std::cmp::Reverse(b.iter().filter(|&&x| x).count()));
    let a = (0..red.m).map(|i| rects.iter().map(|(a, _)| a[i]).collect()).collect();
    let b = rects.into_iter().map(|(_, b)| b).collect();
    (a, b)
}

/// Weighted error of a reduced start, on the known entries.
pub fn start_error(red: &Reduced, a: &[Vec<bool>], b: &[Vec<bool>]) -> u64 {
    let mut e = 0;
    for i in 0..red.m {
        for j in 0..red.n {
            if !red.is_known(i, j) {
                continue;
            }
            let covered = (0..b.len()).any(|k| a[i][k] && b[k][j]);
            if covered != red.rows[i].get(j) {
                e += red.weight(i, j);
            }
        }
    }
    e
}

/// Writes a start for the model of `write_lp` in Gurobi's MST format: every
/// variable of the model, computed from `(a, b)`.
pub fn write_mst(red: &Reduced, a: &[Vec<bool>], b: &[Vec<bool>]) -> String {
    let r = b.len();
    let bit = |v: bool| if v { 1 } else { 0 };
    let mut s = String::new();
    let _ = writeln!(s, "un 1");
    for i in 0..red.m {
        for k in 0..r {
            let _ = writeln!(s, "a_{i}_{k} {}", bit(a[i][k]));
        }
    }
    for k in 0..r {
        for j in 0..red.n {
            let _ = writeln!(s, "b_{k}_{j} {}", bit(b[k][j]));
        }
    }
    for i in 0..red.m {
        for j in 0..red.n {
            if !red.is_known(i, j) {
                continue;
            }
            let mut z = false;
            for k in 0..r {
                let y = a[i][k] && b[k][j];
                z |= y;
                let _ = writeln!(s, "y_{i}_{k}_{j} {}", bit(y));
            }
            let _ = writeln!(s, "z_{i}_{j} {}", bit(z));
        }
    }
    s
}
