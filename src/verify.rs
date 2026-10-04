//! Consistency checks, independent of the optimized code.
//!
//! They use plain loops on the original matrix rather than bitsets on the
//! reduced one. They run in debug builds and with `--verify`.

use crate::io::RawMat;
use crate::reduce::Reduced;
use crate::bits::BitVec;

/// Error of a factorization on the known entries of X, computed naively.
pub fn naive_error(x: &RawMat, w: &[Vec<bool>], h: &[Vec<bool>]) -> u64 {
    let r = h.len();
    let mut e = 0u64;
    for i in 0..x.m {
        for j in 0..x.n {
            let mut z = false;
            for k in 0..r {
                if w[i][k] && h[k][j] {
                    z = true;
                    break;
                }
            }
            if x.is_known(i, j) && z != x.get(i, j) {
                e += 1;
            }
        }
    }
    e
}

/// Checks that the weighted error on the reduced matrix equals the raw error
/// of the expanded factorization on the original matrix.
pub fn check_reduction(
    x: &RawMat,
    red: &Reduced,
    rects: &[(BitVec, BitVec)],
) -> Result<(), String> {
    let weighted = red.rects_error(rects);
    let (w, h) = red.expand(rects);
    let raw_err = naive_error(x, &w, &h);
    if weighted != raw_err {
        return Err(format!(
            "reduction inconsistency: weighted error {weighted}, raw error {raw_err}"
        ));
    }
    Ok(())
}

/// Tracks the bracket and rejects a bound that moves the wrong way.
pub struct Invariants {
    lb: f64,
    ub: u64,
}

impl Invariants {
    pub fn new(ub: u64) -> Invariants {
        Invariants { lb: 0.0, ub }
    }

    /// The lower bound may only rise, and never above the upper bound.
    pub fn update_lb(&mut self, lb: f64) -> Result<(), String> {
        if lb < self.lb - 1e-6 {
            return Err(format!(
                "lower bound going backwards: {} then {}",
                self.lb, lb
            ));
        }
        if lb > self.ub as f64 + 1e-6 {
            return Err(format!(
                "lower bound {} above the solution {}",
                lb, self.ub
            ));
        }
        self.lb = lb.max(self.lb);
        Ok(())
    }

    /// The upper bound may only fall, and never below the lower bound.
    pub fn update_ub(&mut self, ub: u64) -> Result<(), String> {
        if ub > self.ub {
            return Err(format!(
                "upper bound rising: {} then {}",
                self.ub, ub
            ));
        }
        if (self.lb) > ub as f64 + 1e-6 {
            return Err(format!(
                "solution {} below the lower bound {}",
                ub, self.lb
            ));
        }
        self.ub = ub;
        Ok(())
    }

    pub fn lb(&self) -> f64 {
        self.lb
    }

    pub fn ub(&self) -> u64 {
        self.ub
    }
}
