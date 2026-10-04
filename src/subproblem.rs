//! Exact solve of the fixed-factor subproblem, used to polish a factorization.
//!
//! With `W` fixed, each column `x` of `X` is solved independently by choosing
//! the subset `S` of factors it uses. Row `i` is covered when its pattern `p_i`
//! (the factors that take it) meets `S`. With `n_t` the weight of rows of
//! pattern `t`, `a_t` the weight of those that are ones in `x`, and `B` the
//! weight of the zeros of `x`,
//!
//! ```text
//! Err(S) = B + F[complement(S)]   with   F = zeta(2a - n)
//! ```
//!
//! where `zeta` sums over subsets. One `2^r` table therefore gives the error
//! of every `S`. `zeta(n)` depends only on `W` and is computed once per side.
//! Swapping roles gives the solve for `W` with `H` fixed.

use crate::reduce::Reduced;
use crate::bits::BitVec;

/// Largest rank for which the `2^r` table is built.
pub const MAX_ZETA_RANK: usize = 24;

pub struct Patterns {
    pub r: usize,
    /// Pattern of each row: the factors that take it.
    pat: Vec<u32>,
    /// `zeta(n)`, which depends only on the factors.
    zn: Vec<i64>,
    za: Vec<i64>,
}

/// In-place zeta transform (Yates): `f[U]` becomes the sum of `f[t]` over `t`
/// included in `U`.
fn yates(f: &mut [i64], r: usize) {
    for bit in 0..r {
        let clock_step = 1usize << bit;
        for u in 0..f.len() {
            if u & clock_step != 0 {
                f[u] += f[u ^ clock_step];
            }
        }
    }
}

impl Patterns {
    /// `factors[k]` is the fixed side of rectangle `k`, over `len` rows with
    /// weights `weight`.
    pub fn new(factors: &[BitVec], len: usize, weight: &[u64]) -> Patterns {
        let r = factors.len();
        assert!(
            r <= MAX_ZETA_RANK,
            "rank {} too large for the zeta table",
            r
        );
        let mut pat = vec![0u32; len];
        for (k, f) in factors.iter().enumerate() {
            for i in 0..len {
                if f.get(i) {
                    pat[i] |= 1 << k;
                }
            }
        }
        let mut zn = vec![0i64; 1 << r];
        for i in 0..len {
            zn[pat[i] as usize] += weight[i] as i64;
        }
        yates(&mut zn, r);
        Patterns {
            r,
            pat,
            zn,
            za: vec![0i64; 1 << r],
        }
    }

    /// Best subset for `x` and its weighted error, not multiplied by the weight
    /// of `x` itself. Ties go to the smallest `|S|`.
    pub fn best(&mut self, x: &BitVec, weight: &[u64]) -> (u32, u64) {
        let size = 1usize << self.r;
        for v in self.za.iter_mut() {
            *v = 0;
        }
        let mut b = 0i64;
        for (i, &p) in self.pat.iter().enumerate() {
            if x.get(i) {
                self.za[p as usize] += weight[i] as i64;
            } else {
                b += weight[i] as i64;
            }
        }
        yates(&mut self.za, self.r);

        let full = (size - 1) as u32;
        let (mut bs, mut be, mut bc) = (0u32, i64::MAX, u32::MAX);
        for s in 0..size as u32 {
            let u = (!s) & full;
            let e = b + 2 * self.za[u as usize] - self.zn[u as usize];
            let c = s.count_ones();
            if e < be || (e == be && c < bc) {
                be = e;
                bs = s;
                bc = c;
            }
        }
        debug_assert!(be >= 0);
        (bs, be as u64)
    }
}

/// Alternates exact solves of `H` and `W` until the error stops decreasing.
/// Returns the final weighted error.
pub fn ao_exact(red: &Reduced, rects: &mut [(BitVec, BitVec)]) -> u64 {
    let r = rects.len();
    if r == 0 {
        return red.rects_error(rects);
    }
    let mut prev = u64::MAX;
    loop {
        // H with W fixed.
        let w: Vec<BitVec> = rects.iter().map(|(w, _)| w.clone()).collect();
        let mut pats = Patterns::new(&w, red.m, &red.rho);
        let mut e = 0u64;
        for j in 0..red.n {
            let (s, err) = pats.best(&red.cols[j], &red.rho);
            for k in 0..r {
                rects[k].1.set(j, s >> k & 1 == 1);
            }
            e += err * red.gamma[j];
        }

        // W with H fixed.
        let h: Vec<BitVec> = rects.iter().map(|(_, h)| h.clone()).collect();
        let mut pats = Patterns::new(&h, red.n, &red.gamma);
        let mut e2 = 0u64;
        for i in 0..red.m {
            let (s, err) = pats.best(&red.rows[i], &red.gamma);
            for k in 0..r {
                rects[k].0.set(i, s >> k & 1 == 1);
            }
            e2 += err * red.rho[i];
        }
        debug_assert!(e2 <= e, "the exact alternation must be monotone");
        if e2 >= prev {
            return e2;
        }
        prev = e2;
    }
}
