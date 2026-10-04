//! Exact solver for fixed rank Boolean matrix factorization.
//!
//! Exposed as a library so that the tests can reach each layer separately.

pub mod bb;
pub mod bits;
pub mod bounds;
pub mod interruption;
pub mod io;
pub mod mip;
pub mod wcnf;
pub mod reduce;
pub mod rng;
pub mod subproblem;
pub mod verify;
