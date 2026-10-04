//! Exact preprocessing of the input matrix.
//!
//! - Rows and columns with no known one are dropped: leaving them uncovered
//!   costs nothing.
//! - Identical rows are merged, then identical columns, with multiplicities
//!   `rho_i` and `gamma_j`. Some optimal factorization gives identical lines
//!   identical factors, so nothing is lost; entry `(i, j)` then weighs
//!   `rho_i * gamma_j`.
//! - The matrix is transposed if it has more columns than rows. The branch and
//!   bound fixes one column at a time, so fewer columns means a shallower tree.
//!
//! `verify::check_reduction` checks that the weights are consistent.

use crate::io::RawMat;
use crate::bits::BitVec;

pub struct Reduced {
    pub name: String,
    pub m: usize,
    pub n: usize,
    /// The m reduced rows, each of length n.
    pub rows: Vec<BitVec>,
    /// The n reduced columns, each of length m.
    pub cols: Vec<BitVec>,
    /// Known entries, row by row. Missing entries never count as error.
    pub known: Vec<BitVec>,
    pub rho: Vec<u64>,
    pub gamma: Vec<u64>,
    /// True if the working frame is that of X transposed.
    pub transposed: bool,
    /// Dimensions of the working frame before the merge (X, or X^T if
    /// transposed).
    pub om: usize,
    pub on: usize,
    /// Rows and columns of the working frame that each reduced index covers.
    pub row_src: Vec<Vec<usize>>,
    pub col_src: Vec<Vec<usize>>,
    /// Dimensions of the original X.
    pub orig_m: usize,
    pub orig_n: usize,
    /// Total weight of the ones, used for the relative error.
    pub ones_weight: u64,
}

/// Groups identical rows, in order of first appearance. Entries are `0`, `1`
/// or `2` (missing), so merged rows also share their missing entries.
fn groups(rows: &[Vec<u8>]) -> Vec<Vec<usize>> {
    let mut seen: Vec<(&Vec<u8>, usize)> = Vec::new();
    let mut out: Vec<Vec<usize>> = Vec::new();
    for (i, l) in rows.iter().enumerate() {
        match seen.iter().find(|(p, _)| *p == l) {
            Some(&(_, g)) => out[g].push(i),
            None => {
                seen.push((l, out.len()));
                out.push(vec![i]);
            }
        }
    }
    out
}

fn transpose_bool(a: &[Vec<bool>]) -> Vec<Vec<bool>> {
    if a.is_empty() {
        return Vec::new();
    }
    let (m, n) = (a.len(), a[0].len());
    let mut t = vec![vec![false; m]; n];
    for i in 0..m {
        for j in 0..n {
            t[j][i] = a[i][j];
        }
    }
    t
}

fn transpose(a: &[Vec<u8>]) -> Vec<Vec<u8>> {
    if a.is_empty() {
        return Vec::new();
    }
    let (m, n) = (a.len(), a[0].len());
    let mut t = vec![vec![0u8; m]; n];
    for i in 0..m {
        for j in 0..n {
            t[j][i] = a[i][j];
        }
    }
    t
}

impl Reduced {
    pub fn build(raw: &RawMat) -> Reduced {
        // Working frame: X, or X^T if X has more columns than rows.
        let transposed = raw.n > raw.m;
        let mut a: Vec<Vec<u8>> = (0..raw.m)
            .map(|i| {
                (0..raw.n)
                    .map(|j| {
                        if !raw.is_known(i, j) {
                            2
                        } else if raw.get(i, j) {
                            1
                        } else {
                            0
                        }
                    })
                    .collect()
            })
            .collect();
        if transposed {
            a = transpose(&a);
        }
        let (om, on) = (a.len(), a[0].len());

        // Keep only lines with at least one known one.
        let kept_rows: Vec<usize> = (0..om).filter(|&i| a[i].iter().any(|&b| b == 1)).collect();
        let kept_cols: Vec<usize> = (0..on).filter(|&j| (0..om).any(|i| a[i][j] == 1)).collect();
        let b: Vec<Vec<u8>> = kept_rows
            .iter()
            .map(|&i| kept_cols.iter().map(|&j| a[i][j]).collect())
            .collect();

        // Merge rows, then columns. Merging equal columns cannot make two
        // distinct rows equal, so one pass reaches a fixed point.
        let row_groups = groups(&b);
        let c: Vec<Vec<u8>> = row_groups.iter().map(|g| b[g[0]].clone()).collect();
        let col_groups = groups(&transpose(&c));
        let d: Vec<Vec<u8>> = c
            .iter()
            .map(|l| col_groups.iter().map(|g| l[g[0]]).collect::<Vec<u8>>())
            .collect();

        let (m, n) = (d.len(), if d.is_empty() { 0 } else { d[0].len() });
        let row_src: Vec<Vec<usize>> = row_groups
            .iter()
            .map(|g| g.iter().map(|&i| kept_rows[i]).collect())
            .collect();
        let col_src: Vec<Vec<usize>> = col_groups
            .iter()
            .map(|g| g.iter().map(|&j| kept_cols[j]).collect())
            .collect();
        let rho: Vec<u64> = row_src.iter().map(|g| g.len() as u64).collect();
        let gamma: Vec<u64> = col_src.iter().map(|g| g.len() as u64).collect();

        let mut rows = vec![BitVec::zeros(n); m];
        let mut cols = vec![BitVec::zeros(m); n];
        let mut known = vec![BitVec::zeros(n); m];
        let mut ones_weight = 0u64;
        for i in 0..m {
            for j in 0..n {
                if d[i][j] == 2 {
                    continue;
                }
                known[i].set(j, true);
                if d[i][j] == 1 {
                    rows[i].set(j, true);
                    cols[j].set(i, true);
                    ones_weight += rho[i] * gamma[j];
                }
            }
        }

        Reduced {
            name: raw.name.clone(),
            m,
            n,
            rows,
            cols,
            known,
            rho,
            gamma,
            transposed,
            om,
            on,
            row_src,
            col_src,
            orig_m: raw.m,
            orig_n: raw.n,
            ones_weight,
        }
    }

    #[inline]
    pub fn weight(&self, i: usize, j: usize) -> u64 {
        self.rho[i] * self.gamma[j]
    }

    /// True if the reduced entry is known, and so counts in the error.
    #[inline]
    pub fn is_known(&self, i: usize, j: usize) -> bool {
        self.known[i].get(j)
    }

    /// Weighted error of a cover given row by row, over known entries only.
    pub fn error(&self, cov: &[BitVec]) -> u64 {
        let mut e = 0u64;
        for i in 0..self.m {
            for j in 0..self.n {
                if self.is_known(i, j) && self.rows[i].get(j) != cov[i].get(j) {
                    e += self.weight(i, j);
                }
            }
        }
        e
    }

    /// Union of the rectangles, row by row.
    pub fn coverage(&self, rects: &[(BitVec, BitVec)]) -> Vec<BitVec> {
        let mut cov = vec![BitVec::zeros(self.n); self.m];
        for (w, h) in rects {
            for i in 0..self.m {
                if w.get(i) {
                    cov[i].or_in(h);
                }
            }
        }
        cov
    }

    pub fn rects_error(&self, rects: &[(BitVec, BitVec)]) -> u64 {
        self.error(&self.coverage(rects))
    }

    /// Maps a reduced factorization back to the original X. Returns `(W, H)`,
    /// `W` being `orig_m x r` and `H` being `r x orig_n`.
    pub fn expand(&self, rects: &[(BitVec, BitVec)]) -> (Vec<Vec<bool>>, Vec<Vec<bool>>) {
        let r = rects.len();
        // Undo the merge, in the working frame.
        let mut wt = vec![vec![false; r]; self.om];
        let mut ht = vec![vec![false; self.on]; r];
        for (k, (w, h)) in rects.iter().enumerate() {
            for i in 0..self.m {
                if w.get(i) {
                    for &src in &self.row_src[i] {
                        wt[src][k] = true;
                    }
                }
            }
            for j in 0..self.n {
                if h.get(j) {
                    for &src in &self.col_src[j] {
                        ht[k][src] = true;
                    }
                }
            }
        }
        if !self.transposed {
            return (wt, ht);
        }
        // Undo the transposition: (W o H)^T = H^T o W^T.
        let w = transpose_bool(&ht);
        let h = transpose_bool(&wt);
        (w, h)
    }
}
