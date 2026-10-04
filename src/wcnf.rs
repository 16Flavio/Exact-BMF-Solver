//! Export of the problem as weighted partial MaxSAT (WCNF), for an external
//! solver.
//!
//! `a[i][k]` is W, `b[k][j]` is H, `z[i][j]` is the Boolean product, and
//! `y[i][k][j]` witnesses `a[i][k] and b[k][j]`. Hard clauses, per known entry:
//!
//! ```text
//! z  or  not a[i][k]  or  not b[k][j]      for every k   (covered => z)
//! not z  or  y[i][1][j]  or ... or  y[i][r][j]           (z => a witness)
//! not y[i][k][j]  or  a[i][k]                            (witness => a)
//! not y[i][k][j]  or  b[k][j]                            (witness => b)
//! ```
//!
//! These clauses already force `z` to equal the product, so `a and b => y` is
//! not needed. Soft clauses carry the error: `z` for a known one, `not z` for a
//! known zero, weighted as in the reduced matrix. Missing entries are left out.
//! Pattern symmetry is not broken.

use crate::reduce::Reduced;
use std::fmt::Write;

/// Returns the problem as a WCNF file with a `p wcnf` header.
pub fn write_wcnf(red: &Reduced, r: usize) -> String {
    let (m, n) = (red.m, red.n);
    // Variables: all a, then all b, then each known entry's z followed by its
    // r witnesses y.
    let a = |i: usize, k: usize| -> usize { 1 + i * r + k };
    let b = |k: usize, j: usize| -> usize { 1 + m * r + k * n + j };
    let base = 1 + m * r + r * n;
    let mut cell: Vec<Vec<Option<usize>>> = vec![vec![None; n]; m];
    let mut next_var = base;
    for i in 0..m {
        for j in 0..n {
            if red.is_known(i, j) {
                cell[i][j] = Some(next_var);
                next_var += 1 + r;
            }
        }
    }
    let nvars = next_var - 1;

    let mut hard: Vec<String> = Vec::new();
    let mut soft: Vec<(u64, String)> = Vec::new();
    let mut total = 0u64;
    for i in 0..m {
        for j in 0..n {
            let z = match cell[i][j] {
                Some(v) => v,
                None => continue,
            };
            let y = |k: usize| -> usize { z + 1 + k };
            for k in 0..r {
                hard.push(format!("{} -{} -{} 0", z, a(i, k), b(k, j)));
                hard.push(format!("-{} {} 0", y(k), a(i, k)));
                hard.push(format!("-{} {} 0", y(k), b(k, j)));
            }
            let mut witness = format!("-{}", z);
            for k in 0..r {
                let _ = write!(witness, " {}", y(k));
            }
            witness.push_str(" 0");
            hard.push(witness);
            let p = red.weight(i, j);
            total += p;
            if red.rows[i].get(j) {
                soft.push((p, format!("{} 0", z)));
            } else {
                soft.push((p, format!("-{} 0", z)));
            }
        }
    }
    let top = total + 1;
    let mut s = String::new();
    let _ = writeln!(
        s,
        "c boolean matrix factorization of rank {r} on {} reduced to {m}x{n}",
        red.name
    );
    let _ = writeln!(
        s,
        "p wcnf {nvars} {} {top}",
        hard.len() + soft.len()
    );
    for c in &hard {
        let _ = writeln!(s, "{top} {c}");
    }
    for (p, c) in &soft {
        let _ = writeln!(s, "{p} {c}");
    }
    s
}
