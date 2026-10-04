//! Combinatorial branch and bound on column patterns.
//!
//! A rank `r` factorization is determined by its `r` column patterns: given
//! them, each row independently picks the subset of patterns that minimizes
//! its error. The search fixes one column at a time, choosing which patterns
//! contain it (`2^r` children), most discriminating columns first. Patterns
//! are kept in decreasing lexicographic order to break the `r!` symmetry.
//!
//! With the columns `F` fixed, the other columns can only add error, so
//! `sum_i rho_i min_S error_F(i, S)` is a lower bound. It is strengthened by
//! charging the empty subset all the ones of its row, by adding the
//! precomputed optimum of the free suffix (`residuals`), and by inheriting
//! the parent's bound.
//!
//! A subset's cost in the column being fixed depends only on whether it is
//! covered, so all `2^r` children share one table per level, up to a per-row
//! constant `delta`, and a node costs `O(r 2^r m)` instead of `O(4^r m)`.
//!
//! `solve` first tries to prove the incumbent optimal, then runs a bisection
//! ladder of lower targets `T`: exhausting the tree pruned at `T` proves
//! `OPT >= T`. A watcher thread stops attempts that cannot finish in time.
//! After a cut, the minimum bound over the unexplored frontier stays valid.

use crate::interruption;
use crate::reduce::Reduced;
use crate::bits::BitVec;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Barrier, Condvar, Mutex};
use std::time::Instant;

pub struct Outcome {
    /// Valid lower bound, equal to `ub` when the tree has been exhausted.
    pub lb: u64,
    pub ub: u64,
    /// Column patterns of the best solution, one bit per reduced column.
    /// Empty if nothing better than the starting upper bound was found.
    pub patterns: Vec<BitVec>,
    /// True if the tree was fully explored, so the optimum is proven.
    pub complete: bool,
    pub nodes: u64,
}

/// State shared by the threads of one attempt. `threshold` mirrors the cost
/// in `best` for lock-free pruning; all writes go through the lock.
struct Shared {
    threshold: AtomicU64,
    best: Mutex<(u64, Vec<BitVec>)>,
    next_prefix: AtomicUsize,
    nodes: AtomicU64,
    stop: AtomicBool,
    active: AtomicUsize,
    /// Prefixes pruned without opening a node. The watcher excludes them
    /// from its progress estimate.
    pruned: AtomicUsize,
    /// Set by the last thread to finish, so the watcher can exit at once.
    last: (Mutex<bool>, Condvar),
}

/// Per-instance data that does not change during the search. Costs are
/// stored as `u16`; `solve` checks that the total column weight fits.
struct Ctx {
    r: usize,
    n: usize,
    m: usize,
    /// Reduced columns, in exploration order.
    order: Vec<usize>,
    rho: Vec<u64>,
    /// `gbit[depth * m + i]`: cost added to an uncovering subset by the
    /// column at `depth`, i.e. its weight if row `i` has a 1 there.
    gbit: Vec<u16>,
    /// Cost change from uncovered to covered, negative on 1 entries: stored
    /// modulo 2^16 and applied with wrapping adds, the results being in range.
    delta: Vec<u16>,
    /// `ones_suf[depth * m + i]`: weight of row `i`'s ones on the columns
    /// from `depth` onward.
    ones_suf: Vec<u16>,
    settings: Settings,
}

/// Switches for ablation runs. Everything is on by default; `margin` is the
/// watcher's margin, 0 to disable it.
#[derive(Clone, Copy, Debug)]
pub struct Settings {
    pub margin: u128,
    pub no_residuals: bool,
    pub no_ladder: bool,
    pub no_inherit: bool,
    pub no_empty: bool,
    pub reverse_order: bool,
    /// Open the children of a node by increasing bound instead of by mask.
    pub sort_children: bool,
}

impl Default for Settings {
    fn default() -> Settings {
        Settings {
            margin: MARGIN,
            no_residuals: false,
            no_ladder: false,
            no_inherit: false,
            no_empty: false,
            reverse_order: false,
            sort_children: false,
        }
    }
}

/// Column order, most discriminating first: balanced columns (by weighted
/// `min(ones, zeros)`) raise the bound fastest. Ties go to heavier columns.
fn column_order(red: &Reduced) -> Vec<usize> {
    let mut order: Vec<usize> = (0..red.n).collect();
    let score = |j: usize| -> (u64, u64) {
        let n_ones: u64 = (0..red.m)
            .filter(|&i| red.rows[i].get(j))
            .map(|i| red.rho[i])
            .sum();
        let total: u64 = red.rho.iter().sum();
        (red.gamma[j] * n_ones.min(total - n_ones), red.gamma[j])
    };
    order.sort_by_key(|&j| std::cmp::Reverse(score(j)));
    order
}

impl Ctx {
    fn new(red: &Reduced, r: usize, settings: Settings) -> Ctx {
        let mut order = column_order(red);
        if settings.reverse_order {
            order.reverse();
        }
        let (m, n) = (red.m, red.n);
        let mut gbit = vec![0u16; n * m];
        let mut delta = vec![0u16; n * m];
        for (t, &j) in order.iter().enumerate() {
            let g = red.gamma[j] as u16;
            for i in 0..m {
                // A missing entry costs nothing whether covered or not.
                let (b, nb) = if !red.is_known(i, j) {
                    (0, 0)
                } else if red.rows[i].get(j) {
                    (g, 0)
                } else {
                    (0, g)
                };
                gbit[t * m + i] = b;
                delta[t * m + i] = nb.wrapping_sub(b);
            }
        }
        let mut ones_suf = vec![0u16; (n + 1) * m];
        for depth in (0..n).rev() {
            for i in 0..m {
                ones_suf[depth * m + i] = ones_suf[(depth + 1) * m + i] + gbit[depth * m + i];
            }
        }
        Ctx {
            r,
            n,
            m,
            order,
            rho: red.rho.clone(),
            gbit,
            delta,
            ones_suf,
            settings,
        }
    }
}

/// Approximate cost of opening a node, in cost-cell updates.
#[inline]
fn node_cost(r: usize, m: usize) -> u64 {
    (m as u64)
        .saturating_mul(1u64 << r.min(40))
        .saturating_mul(r as u64 + 4)
}

/// Nodes between two clock reads, shrinking with the node cost so the
/// interval stays roughly constant in time.
#[inline]
fn clock_step(r: usize) -> u64 {
    (16_384u64 >> r.min(14)).max(1)
}

/// Nodes at `depth` allowed by the symmetry rule: multisets of `r` words of
/// `depth` bits, `C(2^depth + r - 1, r)`. Saturates at `u64::MAX`.
fn count_prefixes(r: usize, depth: usize) -> u64 {
    let nwords = 1u64.checked_shl(depth as u32).unwrap_or(u64::MAX);
    let mut c = 1u64;
    for k in 1..=r as u64 {
        c = c.saturating_mul(nwords.saturating_add(k - 1)) / k;
        if c == u64::MAX {
            return u64::MAX;
        }
    }
    c
}

/// Applies the symmetry rule to the column choice `taken`. `equal` has one
/// bit per pair of adjacent patterns still equal so far. Returns the new
/// state, or `None` if the choice breaks the decreasing order.
#[inline]
fn symmetry(r: usize, equal: u32, taken: usize) -> Option<u32> {
    let mut still_equal = equal;
    for k in 0..r.saturating_sub(1) {
        if equal >> k & 1 == 0 {
            continue;
        }
        let (a, b) = (taken >> k & 1, taken >> (k + 1) & 1);
        if a < b {
            return None;
        }
        if a > b {
            still_equal &= !(1 << k);
        }
    }
    Some(still_equal)
}

/// Per-thread scratch space. Tables are laid out by subset then by row,
/// `t[S * m + i]`, so inner loops run over contiguous rows with no branch.
/// The cell for `S = 0` is unused; the empty subset is priced from `ones`.
struct Space {
    /// `levels[depth]`: the shared table at `depth` (see `extend_table`).
    levels: Vec<Vec<u16>>,
    /// `bounds[depth * 2^r + taken]`: bound of child `taken` at `depth`.
    bounds: Vec<u64>,
    /// Transform buffers used only while opening a node.
    z: Vec<u16>,
    gg: Vec<u16>,
    g: Vec<u16>,
    /// `delta + gbit`: increment for a subset covered at the previous level.
    gplus: Vec<u16>,
    mins: Vec<u16>,
    /// `order[depth * 2^r + i]`: the `i`th child to open at `depth`.
    order: Vec<u32>,
}

impl Space {
    fn new(m: usize, r: usize, subsets: usize, depth: usize) -> Space {
        Space {
            levels: (0..=depth).map(|_| vec![0u16; subsets * m]).collect(),
            bounds: vec![0u64; (depth + 1) * subsets],
            z: vec![0u16; subsets * m],
            gg: vec![0u16; subsets * m],
            g: vec![0u16; r.max(1) * m],
            gplus: vec![0u16; m],
            mins: vec![0u16; m],
            order: vec![0u32; (depth + 1) * subsets],
        }
    }

    fn clear_from(&mut self, depth: usize) {
        for v in self.levels[depth].iter_mut() {
            *v = 0;
        }
    }
}

/// Builds the table for level `depth + 1`, storing each subset's cost with
/// the column at `depth` uncovered; a child's true cost differs by `delta` on
/// the subsets meeting its mask. `mask` is the previous column's choice,
/// whose `delta` is applied here.
#[inline]
fn extend_table(
    c: &Ctx,
    depth: usize,
    mask: usize,
    subsets: usize,
    gplus: &mut [u16],
    prev: &[u16],
    next: &mut [u16],
) {
    let m = c.m;
    let base = depth * m;
    let gbit = &c.gbit[base..base + m];
    let gplus = &mut gplus[..m];
    if depth > 0 {
        let d = &c.delta[base - m..base];
        for ((t, x), y) in gplus.iter_mut().zip(d).zip(gbit) {
            *t = x.wrapping_add(*y);
        }
    } else {
        gplus.copy_from_slice(gbit);
    }
    for sn in 1..subsets {
        let d = sn * m;
        let add = if sn & mask == 0 { gbit } else { &gplus[..] };
        let (src, dst) = (&prev[d..d + m], &mut next[d..d + m]);
        for ((v, ad), t) in src.iter().zip(add).zip(dst.iter_mut()) {
            *t = v.wrapping_add(*ad);
        }
    }
}

/// Computes the bounds of all `2^r` children of a node into `bounds`. For
/// child `taken`, the per-row minimum over non-empty subsets is
///
/// ```text
/// min( min_{S subset of !taken} t_S , delta + min_{S meets taken} t_S )
/// ```
///
/// Both terms depend on the child only through a mask, so the transforms
/// `z` and `gg` below are computed once for all children.
#[allow(clippy::too_many_arguments)]
fn open(
    c: &Ctx,
    ones: &[u16],
    p0: usize,
    depth: usize,
    subsets: usize,
    equal: u32,
    residual: &[u64],
    na: &[u16],
    z: &mut [u16],
    gg: &mut [u16],
    g: &mut [u16],
    mins: &mut [u16],
    bounds: &mut [u64],
    parent: u64,
) {
    let m = c.m;
    let r = c.r;

    // z: minimum over the non-empty subsets of each mask. The empty cell is
    // set to infinity so it never wins.
    for v in z[..m].iter_mut() {
        *v = u16::MAX;
    }
    z[m..subsets * m].copy_from_slice(&na[m..subsets * m]);
    for k in 0..r {
        let bit = 1usize << k;
        for hi in (bit..subsets).filter(|q| q & bit != 0) {
            let (before, after) = z.split_at_mut(hi * m);
            let lo = (hi ^ bit) * m;
            for (t, v) in after[..m].iter_mut().zip(&before[lo..lo + m]) {
                *t = (*t).min(*v);
            }
        }
    }

    // g[k]: minimum over the subsets containing `k`. A subset meets `P` iff
    // it contains some `k` in `P`, so gg[P] = min over k in P of g[k].
    for v in g[..r * m].iter_mut() {
        *v = u16::MAX;
    }
    for s in 1..subsets {
        let d = s * m;
        for k in 0..r {
            if s >> k & 1 == 0 {
                continue;
            }
            for (t, v) in g[k * m..k * m + m].iter_mut().zip(&na[d..d + m]) {
                *t = (*t).min(*v);
            }
        }
    }
    for v in gg[..m].iter_mut() {
        *v = u16::MAX;
    }
    for p in 1..subsets {
        let k = p.trailing_zeros() as usize;
        let lo = (p ^ (1 << k)) * m;
        let (before, after) = gg.split_at_mut(p * m);
        for ((t, v), w) in after[..m]
            .iter_mut()
            .zip(&before[lo..lo + m])
            .zip(&g[k * m..k * m + m])
        {
            *t = (*v).min(*w);
        }
    }

    let all_ones = &ones[p0 * m..p0 * m + m];
    let free = &ones[(depth + 1) * m..(depth + 1) * m + m];
    let charge = residual[depth + 1];
    let full = subsets - 1;
    let delta = &c.delta[depth * m..depth * m + m];
    for taken in 0..subsets {
        if symmetry(r, equal, taken).is_none() {
            bounds[taken] = u64::MAX;
            continue;
        }
        let za = &z[(full ^ taken) * m..(full ^ taken) * m + m];
        // For `taken == 0` no subset meets the mask; the sentinel in `gg`
        // must not be shifted by `delta`.
        let mi = &mut mins[..m];
        if taken == 0 {
            mi.copy_from_slice(za);
        } else {
            let gb = &gg[taken * m..taken * m + m];
            for ((t, &a), (&b, &d)) in mi.iter_mut().zip(za).zip(gb.iter().zip(delta)) {
                *t = a.min(b.wrapping_add(d));
            }
        }
        // Two bounds on the same error, so take the max, not the sum: the
        // empty subset charged all its ones, or the fixed columns plus the
        // optimum of the free suffix.
        let mut with_empty = 0u64;
        if c.settings.no_empty {
            for (((&v, &to), &li), &rho) in mi.iter().zip(all_ones).zip(free).zip(c.rho.iter()) {
                with_empty += rho * v.min(to - li) as u64;
            }
            with_empty += charge;
        } else if charge == 0 {
            for ((&v, &to), &rho) in mi.iter().zip(all_ones).zip(c.rho.iter()) {
                with_empty += rho * v.min(to) as u64;
            }
        } else {
            let mut no_empty = 0u64;
            for (((&v, &to), &li), &rho) in mi.iter().zip(all_ones).zip(free).zip(c.rho.iter()) {
                with_empty += rho * v.min(to) as u64;
                no_empty += rho * v.min(to - li) as u64;
            }
            with_empty = with_empty.max(no_empty + charge);
        }
        // The parent's bound is valid for every child. The child's own bound
        // can be lower, since the suffix charge shrinks as columns are fixed.
        bounds[taken] = if c.settings.no_inherit {
            with_empty
        } else {
            with_empty.max(parent)
        };
    }
}

/// Keeps a leaf's solution if it improves.
fn record(c: &Ctx, part: &Shared, cost: u64, choice: &[usize]) {
    let mut patterns = vec![BitVec::zeros(c.n); c.r];
    for (t, &taken) in choice.iter().enumerate() {
        for k in 0..c.r {
            if taken >> k & 1 == 1 {
                patterns[k].set(c.order[t], true);
            }
        }
    }
    let mut g = part.best.lock().unwrap();
    if cost < g.0 {
        *g = (cost, patterns);
        part.threshold.store(cost, Ordering::Relaxed);
    }
}

/// Depth-first search from column `depth`; `mask` is the previous column's
/// choice. Returns false if cut, in which case `front` receives the minimum
/// bound over the unvisited nodes, a lower bound for the unexplored part.
#[allow(clippy::too_many_arguments)]
fn explore(
    c: &Ctx,
    part: &Shared,
    depth: usize,
    mask: usize,
    b: u64,
    equal: u32,
    residual: &[u64],
    space: &mut Space,
    choice: &mut Vec<usize>,
    counter: &mut u64,
    front: &mut u64,
    deadline: Instant,
) -> bool {
    if b >= part.threshold.load(Ordering::Relaxed) {
        return true;
    }
    if depth == c.n {
        record(c, part, b, choice);
        return true;
    }
    // Clock, stop flag and shared counter are checked every `clock_step` nodes.
    *counter += 1;
    if *counter % clock_step(c.r) == 0 {
        part.nodes.fetch_add(clock_step(c.r), Ordering::Relaxed);
        if part.stop.load(Ordering::Relaxed) || interruption::expired(deadline) {
            *front = (*front).min(b);
            return false;
        }
    }
    let subsets = 1usize << c.r;
    {
        let Space {
            levels,
            bounds,
            z,
            gg,
            g,
            gplus,
            mins,
            ..
        } = &mut *space;
        let (before, after) = levels.split_at_mut(depth + 1);
        extend_table(c, depth, mask, subsets, gplus, &before[depth], &mut after[0]);
        open(
            c,
            &c.ones_suf,
            0,
            depth,
            subsets,
            equal,
            residual,
            &after[0],
            z,
            gg,
            g,
            mins,
            &mut bounds[depth * subsets..(depth + 1) * subsets],
            b,
        );
    }
    // Child order: by mask, or by increasing bound with `sort_children`.
    let base = depth * subsets;
    {
        let Space { order, bounds, .. } = &mut *space;
        let ord = &mut order[base..base + subsets];
        for (i, o) in ord.iter_mut().enumerate() {
            *o = i as u32;
        }
        if c.settings.sort_children {
            let b = &bounds[base..base + subsets];
            ord.sort_by_key(|&p| b[p as usize]);
        }
    }
    for pos in 0..subsets {
        let taken = space.order[base + pos] as usize;
        let still_equal = match symmetry(c.r, equal, taken) {
            Some(v) => v,
            None => continue,
        };
        let bf = space.bounds[base + taken];
        choice.push(taken);
        let ok = explore(
            c,
            part,
            depth + 1,
            taken,
            bf,
            still_equal,
            residual,
            space,
            choice,
            counter,
            front,
            deadline,
        );
        choice.pop();
        if !ok {
            // Unopened siblings join the frontier (rejected ones carry `u64::MAX`).
            for q in (pos + 1)..subsets {
                let pq = space.order[base + q] as usize;
                *front = (*front).min(space.bounds[base + pq]);
            }
            return false;
        }
    }
    true
}

/// A prefix of columns, the work unit handed out to the threads.
struct Prefix {
    choice: Vec<usize>,
    equal: u32,
    bound: u64,
    /// True when the subtree has been fully explored or pruned.
    done: AtomicBool,
    /// Frontier bound left by a cut search, `u64::MAX` if never started.
    frontier: AtomicU64,
}

impl Prefix {
    fn lower_bound(&self) -> u64 {
        let f = self.frontier.load(Ordering::Relaxed);
        if f == u64::MAX {
            self.bound
        } else {
            f
        }
    }
}

/// Enumerates the prefixes of depth `depth0`, with their bound.
fn prefixes(c: &Ctx, depth0: usize, residual: &[u64]) -> Vec<Prefix> {
    let subsets = 1usize << c.r;
    let mut space = Space::new(c.m, c.r, subsets, depth0);
    let mut out = Vec::new();
    let mut choice = Vec::new();

    #[allow(clippy::too_many_arguments)]
    fn walk(
        c: &Ctx,
        depth: usize,
        depth0: usize,
        mask: usize,
        b: u64,
        equal: u32,
        residual: &[u64],
        space: &mut Space,
        choice: &mut Vec<usize>,
        out: &mut Vec<Prefix>,
    ) {
        if depth == depth0 {
            out.push(Prefix {
                choice: choice.clone(),
                equal,
                bound: b,
                done: AtomicBool::new(false),
                frontier: AtomicU64::new(u64::MAX),
            });
            return;
        }
        let subsets = 1usize << c.r;
        {
            let Space {
                levels,
                bounds,
                z,
                gg,
                g,
                gplus,
                mins,
                ..
            } = &mut *space;
            let (before, after) = levels.split_at_mut(depth + 1);
            extend_table(c, depth, mask, subsets, gplus, &before[depth], &mut after[0]);
            open(
                c,
                &c.ones_suf,
                0,
                depth,
                subsets,
                equal,
                residual,
                &after[0],
                z,
                gg,
                g,
                mins,
                &mut bounds[depth * subsets..(depth + 1) * subsets],
                b,
            );
        }
        for taken in 0..subsets {
            let still_equal = match symmetry(c.r, equal, taken) {
                Some(v) => v,
                None => continue,
            };
            let bf = space.bounds[depth * subsets + taken];
            choice.push(taken);
            walk(c, depth + 1, depth0, taken, bf, still_equal, residual, space, choice, out);
            choice.pop();
        }
    }

    walk(
        c,
        0,
        depth0,
        0,
        residual[0],
        (1u32 << c.r.saturating_sub(1)) - 1,
        residual,
        &mut space,
        &mut choice,
        &mut out,
    );
    out
}

/// Base budget of the residual pass, in cost-cell updates.
const RESIDUAL_CAP: u64 = 1 << 34;

/// Below this many nodes a block is solved on the current thread.
const PARALLEL_THRESHOLD: u64 = 1 << 14;

struct Block {
    /// Best value found by any thread; a minimum, so order-independent.
    best: AtomicU64,
    /// Pruning threshold for the current wave, refreshed only at barriers
    /// so the node count does not depend on thread timing.
    threshold: AtomicU64,
    /// Node budget left for the block, updated only at barriers.
    remaining: AtomicU64,
    /// Set when a thread runs out of budget or hits the deadline.
    failed: AtomicBool,
}

/// Moves a slice of `budget` into `rest`; false once the budget is exhausted.
#[inline]
fn take_slice(r: usize, budget: &mut u64, rest: &mut u64) -> bool {
    if *budget == 0 {
        return false;
    }
    let taken = (*budget).min(clock_step(r));
    *budget -= taken;
    *rest += taken;
    true
}

/// Exhaustive search of the subproblem on columns `p0..p1`. Returns false if
/// the budget or the deadline ran out first.
#[allow(clippy::too_many_arguments)]
fn explore_suffix(
    c: &Ctx,
    ones: &[u16],
    p0: usize,
    p1: usize,
    depth: usize,
    mask: usize,
    b: u64,
    equal: u32,
    subsets: usize,
    residual: &[u64],
    space: &mut Space,
    block: &Block,
    threshold: &mut u64,
    budget: &mut u64,
    rest: &mut u64,
    deadline: Instant,
) -> bool {
    // Prune against the task-local threshold, not `block.best`, so the node
    // count does not depend on other threads' progress.
    if b >= *threshold {
        return true;
    }
    if depth == p1 {
        *threshold = b;
        block.best.fetch_min(b, Ordering::Relaxed);
        return true;
    }
    if *rest == 0 {
        if interruption::expired(deadline) || !take_slice(c.r, budget, rest) {
            return false;
        }
    }
    *rest -= 1;
    {
        let Space {
            levels,
            bounds,
            z,
            gg,
            g,
            gplus,
            mins,
            ..
        } = &mut *space;
        let (before, after) = levels.split_at_mut(depth + 1);
        extend_table(c, depth, mask, subsets, gplus, &before[depth], &mut after[0]);
        open(
            c,
            ones,
            p0,
            depth,
            subsets,
            equal,
            residual,
            &after[0],
            z,
            gg,
            g,
            mins,
            &mut bounds[depth * subsets..(depth + 1) * subsets],
            b,
        );
    }
    for taken in 0..subsets {
        let still_equal = match symmetry(c.r, equal, taken) {
            Some(v) => v,
            None => continue,
        };
        let bf = space.bounds[depth * subsets + taken];
        if !explore_suffix(
            c,
            ones,
            p0,
            p1,
            depth + 1,
            taken,
            bf,
            still_equal,
            subsets,
            residual,
            space,
            block,
            threshold,
            budget,
            rest,
            deadline,
        ) {
            return false;
        }
    }
    true
}

/// Work unit of a block solve: a short prefix with its bound.
struct Task {
    choice: Vec<usize>,
    equal: u32,
    bound: u64,
}

/// Enumerates the tasks of a block: the nodes at the shallowest depth that
/// yields about `target` of them, stopping short of the leaves.
#[allow(clippy::too_many_arguments)]
fn tasks(
    c: &Ctx,
    ones: &[u16],
    p0: usize,
    p1: usize,
    bound0: u64,
    equal0: u32,
    subsets: usize,
    residual: &[u64],
    target: usize,
) -> Vec<Task> {
    let mut d = 0usize;
    let mut nb = 1usize;
    while d + 1 < p1 - p0 && nb < target {
        nb = nb.saturating_mul(subsets);
        d += 1;
    }
    let mut out = vec![Task {
        choice: Vec::new(),
        equal: equal0,
        bound: bound0,
    }];
    if d == 0 {
        return out;
    }
    let mut space = Space::new(c.m, c.r, subsets, p0 + d);
    for depth in p0..p0 + d {
        let mut next_level = Vec::with_capacity(out.len() * subsets);
        for t in &out {
            let mut mask = 0usize;
            for (k, &taken) in t.choice.iter().enumerate() {
                let Space { levels, gplus, .. } = &mut space;
                let (before, after) = levels.split_at_mut(p0 + k + 1);
                extend_table(c, p0 + k, mask, subsets, gplus, &before[p0 + k], &mut after[0]);
                mask = taken;
            }
            {
                let Space {
                    levels,
                    bounds,
                    z,
                    gg,
                    g,
                    gplus,
                    mins,
                    ..
                } = &mut space;
                let (before, after) = levels.split_at_mut(depth + 1);
                extend_table(c, depth, mask, subsets, gplus, &before[depth], &mut after[0]);
                open(
                    c,
                    ones,
                    p0,
                    depth,
                    subsets,
                    t.equal,
                    residual,
                    &after[0],
                    z,
                    gg,
                    g,
                    mins,
                    &mut bounds[depth * subsets..(depth + 1) * subsets],
                    t.bound,
                );
            }
            for taken in 0..subsets {
                let still_equal = match symmetry(c.r, t.equal, taken) {
                    Some(v) => v,
                    None => continue,
                };
                let mut choice = t.choice.clone();
                choice.push(taken);
                next_level.push(Task {
                    choice,
                    equal: still_equal,
                    bound: space.bounds[depth * subsets + taken],
                });
            }
        }
        out = next_level;
    }
    // Low-bound tasks take longest; start them first.
    out.sort_by_key(|t| t.bound);
    out
}

/// Solves the block of columns `p0..p1` on all threads. Returns its optimum,
/// or `None` if the budget or the deadline cut the search short.
#[allow(clippy::too_many_arguments)]
fn solve_block(
    c: &Ctx,
    ones: &[u16],
    p0: usize,
    p1: usize,
    bound0: u64,
    equal0: u32,
    subsets: usize,
    residual: &[u64],
    best0: u64,
    remaining: &mut u64,
    spaces: &mut [Space],
    deadline: Instant,
) -> Option<u64> {
    let nthreads = spaces.len();
    let tasks = tasks(c, ones, p0, p1, bound0, equal0, subsets, residual, 8 * nthreads);
    let d = tasks[0].choice.len();
    let block = Block {
        best: AtomicU64::new(best0),
        threshold: AtomicU64::new(best0),
        remaining: AtomicU64::new(*remaining),
        failed: AtomicBool::new(false),
    };
    let barrier = Barrier::new(nthreads);
    let (bl, tks, bar) = (&block, &tasks, &barrier);
    if nthreads == 1 {
        work_tasks(
            c,
            ones,
            p0,
            p1,
            d,
            subsets,
            residual,
            &mut spaces[0],
            bl,
            tks,
            0,
            1,
            bar,
            deadline,
        );
    } else {
        std::thread::scope(|scope| {
            for (tid, space) in spaces.iter_mut().enumerate() {
                scope.spawn(move || {
                    work_tasks(
                        c, ones, p0, p1, d, subsets, residual, space, bl, tks, tid, nthreads, bar, deadline,
                    )
                });
            }
        });
    }
    *remaining = block.remaining.load(Ordering::Relaxed);
    if block.failed.load(Ordering::Relaxed) {
        None
    } else {
        Some(block.best.load(Ordering::Relaxed))
    }
}

/// One thread's share of a block solve. In wave `v`, thread `tid` takes task
/// `v * nthreads + tid`; the pruning threshold and budget shares change only
/// at barriers, so the outcome does not depend on thread timing.
#[allow(clippy::too_many_arguments)]
fn work_tasks(
    c: &Ctx,
    ones: &[u16],
    p0: usize,
    p1: usize,
    d: usize,
    subsets: usize,
    residual: &[u64],
    space: &mut Space,
    bl: &Block,
    tks: &[Task],
    tid: usize,
    nthreads: usize,
    barrier: &Barrier,
    deadline: Instant,
) {
    space.clear_from(p0);
    let waves = tks.len().div_ceil(nthreads);
    for v in 0..waves {
        let q = v * nthreads + tid;
        // Equal share per remaining task; all threads read the same value.
        let tasks_left = (tks.len() - v * nthreads) as u64;
        let taken = bl.remaining.load(Ordering::Relaxed) / tasks_left.max(1);
        let mut budget = taken;
        let mut rest = 0u64;
        let mut threshold = bl.threshold.load(Ordering::Relaxed);
        if !bl.failed.load(Ordering::Relaxed) && q < tks.len() {
            let t = &tks[q];
            if t.bound < threshold {
                let mut mask = 0usize;
                for (k, &pattern) in t.choice.iter().enumerate() {
                    let Space { levels, gplus, .. } = &mut *space;
                    let (before, after) = levels.split_at_mut(p0 + k + 1);
                    extend_table(c, p0 + k, mask, subsets, gplus, &before[p0 + k], &mut after[0]);
                    mask = pattern;
                }
                if !explore_suffix(
                    c,
                    ones,
                    p0,
                    p1,
                    p0 + d,
                    mask,
                    t.bound,
                    t.equal,
                    subsets,
                    residual,
                    space,
                    bl,
                    &mut threshold,
                    &mut budget,
                    &mut rest,
                    deadline,
                ) {
                    bl.failed.store(true, Ordering::Relaxed);
                }
            }
        }
        let returned = budget + rest;
        bl.remaining
            .fetch_sub(taken.saturating_sub(returned), Ordering::Relaxed);
        barrier.wait();
        if tid == 0 {
            bl.threshold.store(bl.best.load(Ordering::Relaxed), Ordering::Relaxed);
        }
        barrier.wait();
    }
}

/// `residual[depth]` lower-bounds the rank `r` error on columns `depth..n`.
///
/// Error is additive over columns and restriction keeps the rank, so
/// `OPT_r(X) >= sum_B OPT_r(X restricted to B)` for any partition into
/// blocks. Suffixes are solved shortest first, each using the shorter ones as
/// its residual; a block that runs out of budget is closed at its last solved
/// suffix and a new one starts. Also returns the last pass's cuts.
fn residuals(
    c: &Ctx,
    subsets: usize,
    ub: u64,
    spaces: &mut [Space],
    deadline: Instant,
) -> (Vec<u64>, Vec<usize>) {
    let mut residual = vec![0u64; c.n + 1];
    let mut cuts = Vec::new();
    if c.r == 0 {
        return (residual, cuts);
    }
    let nthreads = spaces.len();
    // At most a quarter of the remaining time; the first rung, a probe that
    // finds nothing at high rank, gets a sixteenth.
    let rest = deadline.saturating_duration_since(Instant::now());
    let deadline = Instant::now() + rest / 4;
    let probe = Instant::now() + rest / 16;
    let base = (RESIDUAL_CAP / node_cost(c.r, c.m).max(1)).max(1);
    // A larger budget moves the cuts and is not always better, but every pass
    // is valid, so keep the elementwise max over a ladder of budgets.
    let mut ladder = vec![base, base * (nthreads as u64 / 4).max(1), base * nthreads as u64];
    ladder.dedup();
    let equal0 = (1u32 << c.r.saturating_sub(1)) - 1;
    for (pos, budget) in ladder.into_iter().enumerate() {
        let pass_deadline = if pos == 0 { probe } else { deadline };
        let (c2, whole) = pass(
            c,
            subsets,
            ub,
            budget,
            equal0,
            &mut residual,
            spaces,
            pass_deadline,
        );
        cuts = c2;
        // Stop if the chain completed (more budget adds nothing) or found
        // nothing (more budget will not reach width `r + 1` either).
        if whole || residual[0] == 0 || interruption::expired(deadline) {
            break;
        }
    }
    (residual, cuts)
}

/// One pass of the block chain at a given budget. Raises `residual` in place
/// and returns the cuts, and whether the chain completed without a cut.
#[allow(clippy::too_many_arguments)]
fn pass(
    c: &Ctx,
    subsets: usize,
    ub: u64,
    budget: u64,
    equal0: u32,
    residual: &mut [u64],
    spaces: &mut [Space],
    deadline: Instant,
) -> (Vec<usize>, bool) {
    let mut cuts = Vec::new();
    let mut whole = true;
    // `inner[q]` is `OPT_r` on `[q, prev_cut)`, the residual inside the block.
    let mut inner = vec![0u64; c.n + 1];
    let mut prev_cut = c.n;
    let mut ones = c.ones_suf.clone();
    let mut remaining = budget;
    let mut spent = 0u64;
    // Suffixes of at most `r` columns have optimum zero (unit patterns
    // reproduce them), so start at width `r + 1`.
    let mut p0 = prev_cut.saturating_sub(c.r);
    while p0 > 0 {
        p0 -= 1;
        // Start from the empty factorization, capped where the residual hits `ub`.
        let empty: u64 = (0..c.m).map(|i| c.rho[i] * ones[p0 * c.m + i] as u64).sum();
        let best0 = empty.min(ub.saturating_sub(residual[prev_cut]));
        let before = remaining;
        let useful_children = if spent < PARALLEL_THRESHOLD {
            1
        } else {
            spaces.len()
        };
        let block_out = solve_block(
            c,
            &ones,
            p0,
            prev_cut,
            inner[p0 + 1],
            equal0,
            subsets,
            &inner,
            best0,
            &mut remaining,
            &mut spaces[..useful_children],
            deadline,
        );
        spent = before - remaining;
        if let Some(best) = block_out {
            inner[p0] = best;
            // An earlier block or pass may already hold a higher valid bound.
            residual[p0] = residual[p0].max(best + residual[prev_cut]);
            // Once the residual reaches `ub` it prunes everything.
            if residual[p0] >= ub {
                break;
            }
            continue;
        }
        // Out of budget: close the block at its last solved suffix. Both the
        // closed block and the head left before the cut must be at least
        // `r + 1` wide, since narrower blocks are worth zero.
        whole = false;
        if interruption::expired(deadline) {
            break;
        }
        let cut = (p0 + 1).max(c.r + 1);
        if prev_cut < cut + c.r + 1 {
            break;
        }
        prev_cut = cut;
        cuts.push(prev_cut);
        inner[prev_cut] = 0;
        // Restrict the ones counts to the new block (zero past `prev_cut`).
        for depth in 0..=c.n {
            for i in 0..c.m {
                ones[depth * c.m + i] =
                    c.ones_suf[depth * c.m + i].saturating_sub(c.ones_suf[prev_cut * c.m + i]);
            }
        }
        remaining = budget;
        for q in prev_cut.saturating_sub(c.r)..=prev_cut {
            inner[q] = 0;
        }
        p0 = prev_cut.saturating_sub(c.r);
    }
    // A longer suffix cannot cost less, so propagate values toward the head.
    for p in (0..c.n).rev() {
        residual[p] = residual[p].max(residual[p + 1]);
    }
    (cuts, whole)
}

/// Runs the branch and bound at rank `pos`. `ub` is the incumbent cost and
/// the initial pruning threshold; `lb0` is a lower bound known from
/// elsewhere, used as the starting point of the target ladder.
pub fn solve(
    red: &Reduced,
    pos: usize,
    ub: u64,
    lb0: u64,
    nthreads: usize,
    deadline: Instant,
    verbose: bool,
    settings: Settings,
) -> Outcome {
    let empty = Outcome {
        lb: lb0.min(ub),
        ub,
        patterns: Vec::new(),
        complete: false,
        nodes: 0,
    };
    if pos == 0 || red.n == 0 {
        return empty;
    }
    // Costs are `u16`, and a row's cost never exceeds the total column weight.
    if red.gamma.iter().sum::<u64>() > u16::MAX as u64 {
        return empty;
    }
    // Per-thread table memory; past this the tree is out of reach anyway.
    let bytes = (red.n + 1)
        .saturating_mul(red.m)
        .saturating_mul(1usize.checked_shl(pos as u32).unwrap_or(usize::MAX))
        .saturating_mul(2);
    if bytes > 128 << 20 {
        return empty;
    }
    let c = Ctx::new(red, pos, settings);
    let subsets = 1usize << pos;
    let nthreads = nthreads.max(1);

    let t_residuals = Instant::now();
    let mut spaces: Vec<Space> = (0..nthreads)
        .map(|_| Space::new(c.m, c.r, subsets, c.n))
        .collect();
    let (residual, cuts) = residuals(&c, subsets, ub, &mut spaces, deadline);
    let residual: Vec<u64> = if c.settings.no_residuals {
        vec![0; residual.len()]
    } else {
        residual
    };
    if verbose {
        let profile: Vec<String> = (0..=c.n)
            .rev()
            .map(|p| format!("{}:{}", c.n - p, residual[p]))
            .collect();
        println!(
            "  residuals in {:.1}s (free columns:lower bound): {}",
            t_residuals.elapsed().as_secs_f64(),
            profile.join(" ")
        );
        if !cuts.is_empty() {
            let bounds: Vec<String> = cuts.iter().map(|&q| (c.n - q).to_string()).collect();
            println!(
                "  blocks of the last pass (cuts in free columns): {}",
                bounds.join(" ")
            );
        }
    }

    // Prefix depth: as deep as the prefix count and the (sequential)
    // enumeration work allow, for load balance and a tighter frontier.
    let cap = 65_536.max(8 * nthreads) as u64;
    let work = 1u64 << 32;
    let cost = node_cost(c.r, c.m);
    let mut depth0 = 1.min(c.n);
    while depth0 < c.n {
        let next_level = count_prefixes(c.r, depth0 + 1);
        if next_level > cap || next_level.saturating_mul(cost) > work {
            break;
        }
        depth0 += 1;
    }
    let mut prefs = prefixes(&c, depth0, &residual);
    // Lowest bound first, so the reported lower bound rises steadily.
    prefs.sort_by_key(|p| p.bound);
    let prefs = prefs;
    if prefs.is_empty() {
        return empty;
    }

    // `residual[0]` bounds the whole problem.
    let floor = residual[0].max(lb0);
    let mut e = Bracket {
        lb: floor.min(ub),
        ub,
        patterns: Vec::new(),
        nodes: 0,
    };

    // Proof of the incumbent first, with four fifths of the time.
    let mut hi = e.ub;
    if e.lb < e.ub && !interruption::expired(deadline) {
        let rest = deadline.saturating_duration_since(Instant::now());
        let cutoff = if c.settings.no_ladder {
            deadline
        } else {
            (Instant::now() + rest * 4 / 5).min(deadline)
        };
        if verbose {
            println!(
                "  proof: target {} (bounds [{}, {}])",
                e.ub, e.lb, e.ub
            );
        }
        let t = attempt(
            &c, &prefs, depth0, subsets, &residual, &mut spaces, e.ub, e.ub, e.lb, cutoff, verbose,
        );
        e.nodes += t.nodes;
        if let Some((cost, ms)) = t.found {
            if cost < e.ub {
                e.ub = cost;
                e.patterns = ms;
            }
        }
        e.lb = e.lb.max(t.lb.min(e.ub));
        if t.complete {
            e.lb = e.ub;
        } else {
            hi = e.ub.saturating_sub(1);
        }
    }

    if e.lb < e.ub && !c.settings.no_ladder {
        ladder(
            &c, &prefs, depth0, subsets, &residual, &mut spaces, &mut e, hi, deadline, verbose,
        );
    }

    Outcome {
        lb: e.lb.min(e.ub),
        ub: e.ub,
        patterns: e.patterns,
        complete: e.lb >= e.ub,
        nodes: e.nodes,
    }
}

/// Current `[lb, ub]` bracket and the patterns of the incumbent.
struct Bracket {
    lb: u64,
    ub: u64,
    /// Empty until the search improves on the starting solution.
    patterns: Vec<BitVec>,
    nodes: u64,
}

/// Bisection over targets between `lb` and `hi0`. An attempt at `T` that
/// completes certifies `OPT >= T`; one that is cut lowers `hi` below `T`.
/// Each attempt gets half of the remaining time.
#[allow(clippy::too_many_arguments)]
fn ladder(
    c: &Ctx,
    prefs: &[Prefix],
    depth0: usize,
    subsets: usize,
    residual: &[u64],
    spaces: &mut [Space],
    e: &mut Bracket,
    hi0: u64,
    deadline: Instant,
    verbose: bool,
) {
    let mut hi = hi0.min(e.ub);
    while e.lb < hi && !interruption::expired(deadline) {
        let rest = deadline.saturating_duration_since(Instant::now());
        if rest < std::time::Duration::from_millis(200) {
            break;
        }
        let target = e.lb + (hi - e.lb).div_ceil(2);
        let cutoff = (Instant::now() + rest / 2).min(deadline);
        if verbose {
            println!("  target {target} (bounds [{}, {}])", e.lb, e.ub);
        }
        let t = attempt(
            c, prefs, depth0, subsets, residual, spaces, target, e.ub, e.lb, cutoff, verbose,
        );
        e.nodes += t.nodes;
        if let Some((cost, ms)) = t.found {
            if cost < e.ub {
                e.ub = cost;
                e.patterns = ms;
            }
        }
        e.lb = e.lb.max(t.lb.min(e.ub));
        if t.complete {
            // Exhausted: no solution lies strictly below `target`.
            e.lb = e.lb.max(target.min(e.ub));
        } else {
            hi = target.saturating_sub(1);
        }
        hi = hi.min(e.ub);
    }
}

/// An attempt is stopped when its projected time exceeds `MARGIN` times its
/// remaining time. Wide, because hardest prefixes come first.
const MARGIN: u128 = 50;

/// Watcher thread: stops an attempt unlikely to finish in time, projecting
/// linearly the fraction of non-trivial prefixes exhausted, so the time goes
/// to lower targets instead.
fn watch(
    part: &Shared,
    prefs: &[Prefix],
    start: Instant,
    deadline: Instant,
    margin: u128,
    verbose: bool,
) {
    if margin == 0 {
        return;
    }
    let total = prefs.len() as u128;
    let patience =
        (deadline.saturating_duration_since(start) / 4).max(std::time::Duration::from_millis(500));
    // Zero exhausted prefixes is judged after `floor`; a slow nonzero rate
    // only after `patience`.
    let floor = std::time::Duration::from_secs(3).min(patience);
    let (lock, signal) = &part.last;
    let mut done_flag = lock.lock().unwrap();
    loop {
        // Check before waiting: the last thread may have signalled before
        // the lock was taken.
        if *done_flag || part.stop.load(Ordering::Relaxed) {
            return;
        }
        let (kept, _) = signal
            .wait_timeout(done_flag, std::time::Duration::from_millis(100))
            .unwrap();
        done_flag = kept;
        if *done_flag || part.stop.load(Ordering::Relaxed) {
            return;
        }
        let elapsed = start.elapsed();
        if elapsed < floor {
            continue;
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return;
        }
        let done = prefs
            .iter()
            .filter(|p| p.done.load(Ordering::Relaxed))
            .count() as u128;
        // Prefixes pruned without work say nothing about the rate.
        let free_prefixes = part.pruned.load(Ordering::Relaxed) as u128;
        let useful = total.saturating_sub(free_prefixes);
        let progressed = done.saturating_sub(free_prefixes);
        if progressed > 0 && elapsed < patience {
            continue;
        }
        // With none exhausted, assume one is about to finish. Relative to the
        // time left, so it never fires without a time limit.
        let cadence = progressed.max(1);
        let feasible =
            (useful - progressed) * elapsed.as_millis() <= margin * cadence * remaining.as_millis();
        if !feasible {
            if verbose {
                println!(
                    "  watcher: cut at {:.1}s, {progressed}/{useful} useful prefixes",
                    elapsed.as_secs_f64()
                );
            }
            part.stop.store(true, Ordering::Relaxed);
            return;
        }
    }
}

/// Result of one attempt.
struct Attempt {
    /// Valid lower bound on the optimum.
    lb: u64,
    /// Solution found strictly below the target, if any.
    found: Option<(u64, Vec<BitVec>)>,
    /// True if the tree was exhausted before the deadline.
    complete: bool,
    nodes: u64,
}

/// Searches the tree with pruning threshold `target`, recording only
/// solutions strictly below it. Exhausting the tree proves `OPT >= target`.
#[allow(clippy::too_many_arguments)]
fn attempt(
    c: &Ctx,
    prefs: &[Prefix],
    depth0: usize,
    subsets: usize,
    residual: &[u64],
    spaces: &mut [Space],
    target: u64,
    true_ub: u64,
    floor: u64,
    deadline: Instant,
    verbose: bool,
) -> Attempt {
    // Prefix bounds do not depend on the threshold; only progress is reset.
    for p in prefs {
        p.done.store(false, Ordering::Relaxed);
        p.frontier.store(u64::MAX, Ordering::Relaxed);
    }
    let nthreads = spaces.len();
    let part = Shared {
        threshold: AtomicU64::new(target),
        best: Mutex::new((target, Vec::new())),
        next_prefix: AtomicUsize::new(0),
        nodes: AtomicU64::new(0),
        stop: AtomicBool::new(false),
        active: AtomicUsize::new(nthreads),
        pruned: AtomicUsize::new(0),
        last: (Mutex::new(false), Condvar::new()),
    };

    let start = Instant::now();
    std::thread::scope(|scope| {
        if verbose {
            let (part, prefs) = (&part, &prefs);
            scope.spawn(move || log_progress(part, prefs, target, true_ub, floor, start));
        }
        {
            let (part, prefs) = (&part, &prefs);
            scope.spawn(move || watch(part, prefs, start, deadline, c.settings.margin, verbose));
        }
        for space in spaces.iter_mut() {
            let (c, part, prefs, residual) = (&c, &part, &prefs, &residual);
            scope.spawn(move || {
                let mut counter = 0u64;
                let mut front = u64::MAX;
                loop {
                    if part.stop.load(Ordering::Relaxed) {
                        // `break`, not `return`: the code after the loop
                        // decrements `active`, which the logger waits on.
                        break;
                    }
                    let p = part.next_prefix.fetch_add(1, Ordering::Relaxed);
                    if p >= prefs.len() {
                        break;
                    }
                    let pref = &prefs[p];
                    if pref.bound >= part.threshold.load(Ordering::Relaxed) {
                        pref.done.store(true, Ordering::Relaxed);
                        part.pruned.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                    let mut choice = pref.choice.clone();
                    let mut mask = 0usize;
                    for (depth, &taken) in pref.choice.iter().enumerate() {
                        let Space { levels, gplus, .. } = &mut *space;
                        let (before, after) = levels.split_at_mut(depth + 1);
                        extend_table(c, depth, mask, subsets, gplus, &before[depth], &mut after[0]);
                        mask = taken;
                    }
                    let ok = explore(
                        c,
                        part,
                        depth0,
                        mask,
                        pref.bound,
                        pref.equal,
                        residual,
                        space,
                        &mut choice,
                        &mut counter,
                        &mut front,
                        deadline,
                    );
                    if ok {
                        pref.done.store(true, Ordering::Relaxed);
                    } else {
                        pref.frontier.store(front, Ordering::Relaxed);
                        part.stop.store(true, Ordering::Relaxed);
                        break;
                    }
                }
                part.nodes
                    .fetch_add(counter % clock_step(c.r), Ordering::Relaxed);
                if part.active.fetch_sub(1, Ordering::AcqRel) == 1 {
                    // Last thread out wakes the watcher.
                    let (lock, signal) = &part.last;
                    *lock.lock().unwrap() = true;
                    signal.notify_all();
                }
            });
        }
    });

    let (cost, patterns) = part.best.lock().unwrap().clone();
    let complete = !part.stop.load(Ordering::Relaxed);
    // Explored and pruned subtrees hold nothing below `cost`; unfinished
    // prefixes contribute their frontier bound.
    let mut lb = cost;
    if !complete {
        for p in prefs {
            if !p.done.load(Ordering::Relaxed) {
                lb = lb.min(p.lower_bound());
            }
        }
    }
    Attempt {
        lb: lb.max(floor),
        found: if patterns.is_empty() {
            None
        } else {
            Some((cost, patterns))
        },
        complete,
        nodes: part.nodes.load(Ordering::Relaxed),
    }
}

/// Rebuilds the factorization from the column patterns: each row takes the
/// subset of patterns that minimizes its error, the smallest on ties.
pub fn factorize(red: &Reduced, patterns: &[BitVec]) -> Vec<(BitVec, BitVec)> {
    let (n, r) = (red.n, patterns.len());
    let mut rects: Vec<(BitVec, BitVec)> = patterns
        .iter()
        .map(|h| (BitVec::zeros(red.m), h.clone()))
        .collect();
    let mut u = BitVec::zeros(n);
    for i in 0..red.m {
        let (mut bs, mut be) = (0usize, u64::MAX);
        for s in 0..(1usize << r) {
            u.fill_zero();
            for k in 0..r {
                if s >> k & 1 == 1 {
                    u.or_in(&patterns[k]);
                }
            }
            let e: u64 = (0..n)
                .filter(|&j| red.is_known(i, j) && red.rows[i].get(j) != u.get(j))
                .map(|j| red.gamma[j])
                .sum();
            if e < be || (e == be && s.count_ones() < bs.count_ones()) {
                be = e;
                bs = s;
            }
        }
        for (k, rect) in rects.iter_mut().enumerate() {
            rect.0.set(i, bs >> k & 1 == 1);
        }
    }
    rects
}

/// Current lower bound: minimum over unfinished prefixes, within `[floor, best]`.
fn current_bound(part: &Shared, prefs: &[Prefix], best: u64, floor: u64) -> u64 {
    let mut lb = best;
    for p in prefs {
        if !p.done.load(Ordering::Relaxed) {
            lb = lb.min(p.lower_bound());
        }
    }
    let _ = part;
    lb.max(floor).min(best)
}

/// Progress log: one line per second while threads are running.
fn log_progress(part: &Shared, prefs: &[Prefix], target: u64, true_ub: u64, floor: u64, start: Instant) {
    let total = prefs.len();
    println!(
        "  {:>8}  {:>12}  {:>15}  {:>9}  {:>9}  {:>7}",
        "time", "nodes", "prefixes", "LB", "UB", "gap"
    );
    let mut last_line = String::new();
    let mut round = 0u32;
    loop {
        // Poll every 100 ms so the logger exits promptly, but print once a second.
        std::thread::sleep(std::time::Duration::from_millis(100));
        let still_active = part.active.load(Ordering::Relaxed);
        round += 1;
        if round % 10 != 0 && still_active != 0 {
            continue;
        }
        let done = prefs
            .iter()
            .filter(|p| p.done.load(Ordering::Relaxed))
            .count();
        // `best` starts at the target; UB shows the real incumbent.
        let best = part.best.lock().unwrap().0.min(target);
        let ub = true_ub.min(best);
        let lb = current_bound(part, prefs, best, floor);
        let gap = if ub == 0 {
            0.0
        } else {
            100.0 * (ub - lb.min(ub)) as f64 / ub as f64
        };
        let line = format!(
            "  {:>7.1}s  {:>12}  {:>7}/{:<7}  {:>9}  {:>9}  {:>6.1}%",
            start.elapsed().as_secs_f64(),
            part.nodes.load(Ordering::Relaxed),
            done,
            total,
            lb,
            ub,
            gap
        );
        if line[10..] != last_line {
            println!("{line}");
            last_line = line[10..].to_string();
        }
        if still_active == 0 {
            return;
        }
    }
}
