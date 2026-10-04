//! Combinatorial lower bound from isolated sets, summed over a grid of blocks.
//!
//! An isolated set is a set of ones on pairwise distinct rows and columns such
//! that every pair has a known zero at one of its two crossed positions. A
//! rectangle covering two of its elements covers that zero too, so covering
//! `k` isolated ones with `r` rectangles forces false positives or leaves ones
//! uncovered. `isolated_bound` prices the cheapest way to do so.
//!
//! A single isolated set has at most `min(m, n)` elements. Error is additive
//! over a partition of the entries, and a rank `r` factorization restricted to
//! a block is still of rank `r`, so for any grid of blocks
//! `OPT_r(X) >= sum over blocks of isolation_bound_r(block)`. The grid bound
//! sweeps block sizes and orderings, then runs a local search that moves rows
//! and columns between bands. Every grid visited is a valid bound, so the
//! maximum is too.

use crate::reduce::Reduced;
use crate::bits::BitVec;
use crate::rng::Rng;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

/// Maximum isolated set, searched over the whole matrix.
pub fn isolated_set(red: &Reduced) -> Vec<(usize, usize)> {
    let rows: Vec<usize> = (0..red.m).collect();
    let cols: Vec<usize> = (0..red.n).collect();
    isolated_set_in_block(red, &rows, &cols, 20_000)
}

/// Isolated set in the sub block `rows x cols`, as indices of `red`.
///
/// This is a stable set in the conflict graph of the block's ones, found by a
/// greedy pass in increasing degree order followed by `effort` random 1-for-1
/// swaps with recompletion.
pub fn isolated_set_in_block(
    red: &Reduced,
    rows: &[usize],
    cols: &[usize],
    effort: usize,
) -> Vec<(usize, usize)> {
    let ones: Vec<(usize, usize)> = rows
        .iter()
        .flat_map(|&i| cols.iter().map(move |&j| (i, j)))
        .filter(|&(i, j)| red.rows[i].get(j))
        .collect();
    let p = ones.len();
    if p == 0 {
        return Vec::new();
    }
    // The conflict graph takes `p * p` bits; skip it on very large blocks.
    if p > 12_000 {
        return greedy_without_graph(red, &ones);
    }
    let mut adj: Vec<BitVec> = (0..p).map(|_| BitVec::zeros(p)).collect();
    for a in 0..p {
        let (ia, ja) = ones[a];
        for b in (a + 1)..p {
            let (ib, jb) = ones[b];
            // Conflict if they share a line, or if neither crossed position is
            // a known zero: then one rectangle covers both at no cost.
            let cross_free = (!red.is_known(ia, jb) || red.rows[ia].get(jb))
                && (!red.is_known(ib, ja) || red.rows[ib].get(ja));
            if ia == ib || ja == jb || cross_free {
                adj[a].set(b, true);
                adj[b].set(a, true);
            }
        }
    }

    // Greedy by increasing degree: vertices that block the least go first.
    let mut order: Vec<usize> = (0..p).collect();
    order.sort_by_key(|&v| adj[v].count());
    let mut blocked = BitVec::zeros(p);
    let mut inside: Vec<usize> = Vec::new();
    for &v in &order {
        if blocked.get(v) {
            continue;
        }
        inside.push(v);
        blocked.set(v, true);
        blocked.or_in(&adj[v]);
    }

    // Swaps. The seed is fixed so that the bound is reproducible.
    let mut rng = Rng::new(0x9E37_79B9 ^ ((p as u64) << 20));
    let mut best_set = inside.clone();
    let mut trial: Vec<usize> = Vec::new();
    for _ in 0..effort {
        let v = rng.below(p);
        if inside.contains(&v) {
            continue;
        }
        // Only a vertex that conflicts with at most one member can enter.
        let mut hit = None;
        let mut several = false;
        for &u in &inside {
            if adj[v].get(u) {
                if hit.is_some() {
                    several = true;
                    break;
                }
                hit = Some(u);
            }
        }
        if several {
            continue;
        }
        trial.clear();
        trial.extend(inside.iter().copied().filter(|&u| Some(u) != hit));
        trial.push(v);
        blocked.fill_zero();
        for &u in &trial {
            blocked.set(u, true);
            blocked.or_in(&adj[u]);
        }
        for u in 0..p {
            if !blocked.get(u) {
                trial.push(u);
                blocked.set(u, true);
                blocked.or_in(&adj[u]);
            }
        }
        if trial.len() >= inside.len() {
            // Accept neutral swaps so the set can drift along plateaus.
            inside.clear();
            inside.extend(trial.iter().copied());
            if inside.len() > best_set.len() {
                best_set.clear();
                best_set.extend(inside.iter().copied());
            }
        }
    }
    best_set.iter().map(|&v| ones[v]).collect()
}

/// Fallback when the conflict graph is too large: a few greedy passes in
/// random orders.
fn greedy_without_graph(red: &Reduced, ones: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let isolated = |taken: &[(usize, usize)], i: usize, j: usize| {
        taken.iter()
            .all(|&(i2, j2)| {
                i != i2
                    && j != j2
                    && ((red.is_known(i, j2) && !red.rows[i].get(j2))
                        || (red.is_known(i2, j) && !red.rows[i2].get(j)))
            })
    };
    let mut best: Vec<(usize, usize)> = Vec::new();
    let mut order: Vec<usize> = (0..ones.len()).collect();
    let mut rng = Rng::new(0x9E37_79B9);
    for trial in 0..16 {
        if trial > 0 {
            for k in (1..order.len()).rev() {
                order.swap(k, rng.below(k + 1));
            }
        }
        let mut taken: Vec<(usize, usize)> = Vec::new();
        for &t in &order {
            let (i, j) = ones[t];
            if isolated(&taken, i, j) {
                taken.push((i, j));
            }
        }
        if taken.len() > best.len() {
            best = taken;
        }
    }
    best
}

/// Isolation bound on the whole matrix.
pub fn isolation_bound(red: &Reduced, r: usize) -> u64 {
    let rows: Vec<usize> = (0..red.m).collect();
    let cols: Vec<usize> = (0..red.n).collect();
    block_bound(red, r, &rows, &cols, 20_000)
}

/// Isolation bound on the sub block `rows x cols`.
pub fn block_bound(
    red: &Reduced,
    r: usize,
    rows: &[usize],
    cols: &[usize],
    effort: usize,
) -> u64 {
    if r == 0 || rows.len() <= r || cols.len() <= r {
        // A block with at most `r` rows or columns is reconstructed exactly.
        return 0;
    }
    let iso = isolated_set_in_block(red, rows, cols, effort);
    isolated_bound(red, r, &iso)
}

/// Minimum weighted error that `r` rectangles must incur on an isolated set.
///
/// Each rectangle covers a subset of the set, and every pair it covers forces
/// a false positive at a crossed position. Distinct pairs have distinct
/// crossed positions, since the elements lie on distinct rows and columns, so
/// false positives are counted once per edge of the union of cliques. The
/// cheapest choice leaves the `u` lightest ones uncovered and splits the other
/// `k - u` into `r` disjoint cliques of balanced sizes, which minimizes the
/// edge count. Each edge is priced from the cheapest pair weights, a pair
/// weighing the cheaper of its known crossed zeros.
fn isolated_bound(red: &Reduced, r: usize, iso: &[(usize, usize)]) -> u64 {
    let k = iso.len();
    if k == 0 || r == 0 {
        return 0;
    }
    let mut one_weights: Vec<u64> = iso.iter().map(|&(i, j)| red.weight(i, j)).collect();
    one_weights.sort_unstable();
    let mut cum_one = vec![0u64; k + 1];
    for t in 0..k {
        cum_one[t + 1] = cum_one[t] + one_weights[t];
    }
    let mut pair_weights: Vec<u64> = Vec::with_capacity(k * (k - 1) / 2);
    for a in 0..k {
        let (ia, ja) = iso[a];
        for b in (a + 1)..k {
            let (ib, jb) = iso[b];
            let mut p = u64::MAX;
            if red.is_known(ia, jb) && !red.rows[ia].get(jb) {
                p = p.min(red.weight(ia, jb));
            }
            if red.is_known(ib, ja) && !red.rows[ib].get(ja) {
                p = p.min(red.weight(ib, ja));
            }
            if p != u64::MAX {
                pair_weights.push(p);
            }
        }
    }
    pair_weights.sort_unstable();
    let mut cum_z = vec![0u64; pair_weights.len() + 1];
    for t in 0..pair_weights.len() {
        cum_z[t + 1] = cum_z[t] + pair_weights[t];
    }
    // Edges of a balanced partition of `q` elements into `r` cliques. It must
    // be a real partition: assuming equal sizes when `r` does not divide `q`
    // overcounts and makes the bound invalid.
    let edges = |q: usize| -> usize {
        let s = q / r;
        let rest = q % r;
        rest * ((s + 1) * s / 2) + (r - rest) * (s * s.saturating_sub(1) / 2)
    };
    let mut best = u64::MAX;
    for u in 0..=k {
        let e = edges(k - u).min(pair_weights.len());
        best = best.min(cum_one[u] + cum_z[e]);
    }
    best
}

/// Bound of one row band, with the columns of `pc` cut into chunks of `tc`.
fn band_bound(
    red: &Reduced,
    r: usize,
    band: &[usize],
    pc: &[usize],
    tc: usize,
    effort: usize,
) -> u64 {
    let mut total = 0u64;
    for col_chunk in pc.chunks(tc) {
        if col_chunk.len() > r {
            total += block_bound(red, r, band, col_chunk, effort);
        }
    }
    total
}

/// Sum of the block bounds with the rows of `pl` cut into bands of `tl`.
///
/// Each band picks its own best column split. Row bands are disjoint, so the
/// blocks still partition the entries.
#[allow(clippy::too_many_arguments)]
fn bound_one_grid(
    red: &Reduced,
    r: usize,
    tl: usize,
    pl: &[usize],
    oc: &[Vec<usize>],
    tc: &[usize],
    effort: usize,
    deadline: Instant,
) -> u64 {
    let mut total = 0u64;
    for band in pl.chunks(tl) {
        if band.len() <= r {
            continue;
        }
        // Stopping midway is safe: a sum over some of the disjoint blocks is
        // still a lower bound.
        if Instant::now() >= deadline {
            break;
        }
        let mut best_bound = 0u64;
        for pc in oc {
            for &t in tc {
                best_bound = best_bound.max(band_bound(red, r, band, pc, t, effort));
            }
        }
        total += best_bound;
    }
    total
}

/// Candidate orders of the lines before cutting them into bands: identity,
/// sorted by weight, interleaved by weight, and `random` shuffles. A block
/// full of ones is worth little, so spreading heavy lines tends to help.
fn orders(n: usize, weight: &[u64], rng: &mut Rng, random: usize) -> Vec<Vec<usize>> {
    let mut out = vec![(0..n).collect::<Vec<usize>>()];
    let mut sorted: Vec<usize> = (0..n).collect();
    sorted.sort_by_key(|&i| weight[i]);
    // Interleave the sorted order so each band mixes heavy and light lines.
    let mut spread = Vec::with_capacity(n);
    let clock_step = 8.min(n.max(1));
    for offset in 0..clock_step {
        let mut k = offset;
        while k < n {
            spread.push(sorted[k]);
            k += clock_step;
        }
    }
    out.push(sorted);
    out.push(spread);
    for _ in 0..random {
        let mut p: Vec<usize> = (0..n).collect();
        for k in (1..p.len()).rev() {
            p.swap(k, rng.below(k + 1));
        }
        out.push(p);
    }
    out
}

/// Rows cut into `p` bands and columns into `q`, with each block's bound cached.
struct Partition {
    bands: Vec<Vec<usize>>,
    cols: Vec<Vec<usize>>,
    /// `val[a * q + b]`, the bound of the block `bands[a] x cols[b]`.
    val: Vec<u64>,
}

impl Partition {
    fn total(&self) -> u64 {
        self.val.iter().sum()
    }

    fn redo_band(&mut self, red: &Reduced, r: usize, a: usize, effort: usize) {
        let q = self.cols.len();
        for b in 0..q {
            self.val[a * q + b] = block_bound(red, r, &self.bands[a], &self.cols[b], effort);
        }
    }

    fn redo_column(&mut self, red: &Reduced, r: usize, b: usize, effort: usize) {
        let q = self.cols.len();
        for a in 0..self.bands.len() {
            self.val[a * q + b] = block_bound(red, r, &self.bands[a], &self.cols[b], effort);
        }
    }
}

/// Local search over `p x q` grid partitions, starting from a random one.
///
/// A move shifts one row or column to another band and is kept unless the
/// total drops; neutral moves let the search cross plateaus. Every partition
/// visited gives a valid bound, so the best value seen is returned.
#[allow(clippy::too_many_arguments)]
fn local_search(
    red: &Reduced,
    r: usize,
    p: usize,
    q: usize,
    seed: u64,
    moves: usize,
    effort: usize,
    deadline: Instant,
) -> u64 {
    if p == 0 || q == 0 || red.m < p * (r + 1) || red.n < q * (r + 1) {
        return 0;
    }
    let mut rng = Rng::new(seed);
    let mut row_order: Vec<usize> = (0..red.m).collect();
    let mut col_order: Vec<usize> = (0..red.n).collect();
    for k in (1..row_order.len()).rev() {
        row_order.swap(k, rng.below(k + 1));
    }
    for k in (1..col_order.len()).rev() {
        col_order.swap(k, rng.below(k + 1));
    }
    // Round robin, so band sizes differ by at most one.
    let mut d = Partition {
        bands: (0..p)
            .map(|a| row_order.iter().skip(a).step_by(p).copied().collect())
            .collect(),
        cols: (0..q)
            .map(|b| col_order.iter().skip(b).step_by(q).copied().collect())
            .collect(),
        val: vec![0; p * q],
    };
    for a in 0..p {
        d.redo_band(red, r, a, effort);
    }
    let mut current = d.total();
    let mut best_set = current;
    let mut kept: Vec<u64> = Vec::with_capacity(2 * p.max(q));

    for step in 0..moves {
        if step % 64 == 0 && Instant::now() >= deadline {
            break;
        }
        let on_rows = rng.bool();
        let (na, nb) = if on_rows { (p, p) } else { (q, q) };
        let a = rng.below(na);
        let b = rng.below(nb);
        if a == b {
            continue;
        }
        let source = if on_rows { &d.bands[a] } else { &d.cols[a] };
        // A band of `r` lines or fewer is worth nothing; do not shrink to it.
        if source.len() <= r + 1 {
            continue;
        }
        let idx = rng.below(source.len());
        // Save the block values the move touches, for rollback.
        kept.clear();
        if on_rows {
            for b2 in 0..q {
                kept.push(d.val[a * q + b2]);
                kept.push(d.val[b * q + b2]);
            }
        } else {
            for a2 in 0..p {
                kept.push(d.val[a2 * q + a]);
                kept.push(d.val[a2 * q + b]);
            }
        }
        let v = if on_rows {
            let v = d.bands[a].remove(idx);
            d.bands[b].push(v);
            d.redo_band(red, r, a, effort);
            d.redo_band(red, r, b, effort);
            v
        } else {
            let v = d.cols[a].remove(idx);
            d.cols[b].push(v);
            d.redo_column(red, r, a, effort);
            d.redo_column(red, r, b, effort);
            v
        };
        let fresh = d.total();
        if fresh >= current {
            current = fresh;
            best_set = best_set.max(fresh);
        } else {
            if on_rows {
                d.bands[b].pop();
                d.bands[a].push(v);
                for (b2, pair) in kept.chunks(2).enumerate() {
                    d.val[a * q + b2] = pair[0];
                    d.val[b * q + b2] = pair[1];
                }
            } else {
                d.cols[b].pop();
                d.cols[a].push(v);
                for (a2, pair) in kept.chunks(2).enumerate() {
                    d.val[a2 * q + a] = pair[0];
                    d.val[a2 * q + b] = pair[1];
                }
            }
        }
    }
    best_set
}

/// Grid bound: the best sum of block bounds over a sweep of block sizes and
/// line orders, then, if `local` is set, over rounds of local search. Never
/// below `offset`. Runs on `nthreads` threads and returns what it has at the
/// deadline.
pub fn grid_bound(
    red: &Reduced,
    r: usize,
    nthreads: usize,
    offset: u64,
    local: bool,
    deadline: Instant,
    verbose: bool,
) -> u64 {
    let pass_start = Instant::now();
    let global = isolation_bound(red, r).max(offset);
    if r == 0 || red.m <= r || red.n <= r {
        return global;
    }
    let row_weights: Vec<u64> = (0..red.m)
        .map(|i| (0..red.n).filter(|&j| red.rows[i].get(j)).count() as u64)
        .collect();
    let col_weights: Vec<u64> = (0..red.n)
        .map(|j| (0..red.m).filter(|&i| red.rows[i].get(j)).count() as u64)
        .collect();
    let mut rng = Rng::new(0x5DEE_CE66);
    let ol = orders(red.m, &row_weights, &mut rng, 3);
    let oc = orders(red.n, &col_weights, &mut rng, 3);

    // Candidate band sizes, between `r + 1` and the full dimension.
    let sizes = |d: usize| -> Vec<usize> {
        let mut v: Vec<usize> = [r + 1, r + 3, r + 6, 2 * r, 3 * r, 4 * r, 5 * r, d]
            .into_iter()
            .filter(|&t| t > r && t <= d)
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    let tl = sizes(red.m);
    let tc = sizes(red.n);

    // One task per (row order, band size).
    let mut tasks: Vec<(usize, usize)> = Vec::new();
    for (a, _) in ol.iter().enumerate() {
        for &s in &tl {
            tasks.push((a, s));
        }
    }
    // The value and the grid that achieves it are updated together under the
    // lock; the lock-free read only filters out non-improvements.
    let threshold = AtomicU64::new(global);
    let chosen: Mutex<Option<(usize, usize)>> = Mutex::new(None);
    let next_task = AtomicU64::new(0);
    let (tasks, ol, oc, tc) = (&tasks, &ol, &oc, &tc);
    let (threshold, chosen, next_task) = (&threshold, &chosen, &next_task);
    std::thread::scope(|scope| {
        for _ in 0..nthreads.max(1) {
            scope.spawn(move || loop {
                let q = next_task.fetch_add(1, Ordering::Relaxed) as usize;
                if q >= tasks.len() || Instant::now() >= deadline {
                    return;
                }
                let (a, s) = tasks[q];
                let v = bound_one_grid(red, r, s, &ol[a], oc, tc, 200, deadline);
                if v > threshold.load(Ordering::Relaxed) {
                    let mut g = chosen.lock().unwrap();
                    if v > threshold.load(Ordering::Relaxed) {
                        threshold.store(v, Ordering::Relaxed);
                        *g = Some((a, s));
                    }
                }
            });
        }
    });
    let mut best_set = threshold.load(Ordering::Relaxed);

    // Re-evaluate the best grid with a larger swap effort, if at least as much
    // time is left as the sweep took.
    if let Some((a, s)) = *chosen.lock().unwrap() {
        if deadline.saturating_duration_since(Instant::now()) > pass_start.elapsed() {
            best_set = best_set.max(bound_one_grid(red, r, s, &ol[a], oc, tc, 20_000, deadline));
        }
    }

    if !local {
        return best_set;
    }
    let pmax = (red.m / (r + 1)).min(10);
    let qmax = (red.n / (r + 1)).min(10);
    // Upper limit on what a `p x q` grid can reach; plans are tried in
    // decreasing order of it.
    let cap_of = |p: usize, q: usize| -> i64 {
        (p.min(q) as i64) * (red.m.min(red.n) as i64) - (r as i64) * (p as i64) * (q as i64)
    };
    let threshold2 = AtomicU64::new(best_set);

    // Rounds of local search, each with new seeds and twice the moves and swap
    // effort. Stop at the deadline or after two rounds without a gain.
    let mut round = 0u32;
    let mut secs = 0u32;
    loop {
        if Instant::now() >= deadline || secs >= 2 {
            break;
        }
        let moves = 3_000usize.saturating_mul(1 << round.min(5));
        let effort = 60usize.saturating_mul(1 << round.min(5));
        let mut plans: Vec<(usize, usize, u64)> = Vec::new();
        for p in 1..=pmax {
            for q in 1..=qmax {
                if p * q <= 1 {
                    continue;
                }
                for offset in 0..2u64 {
                    plans.push((
                        p,
                        q,
                        0x1234_5678
                            ^ (p as u64) << 40
                            ^ (q as u64) << 20
                            ^ offset
                            ^ (round as u64) << 52,
                    ));
                }
            }
        }
        plans.sort_by_key(|&(p, q, _)| std::cmp::Reverse(cap_of(p, q)));
        let before = threshold2.load(Ordering::Relaxed);
        let next_level = AtomicU64::new(0);
        let (plans, s2, next_level) = (&plans, &threshold2, &next_level);
        std::thread::scope(|scope| {
            for _ in 0..nthreads.max(1) {
                scope.spawn(move || loop {
                    let q = next_level.fetch_add(1, Ordering::Relaxed) as usize;
                    if q >= plans.len() || Instant::now() >= deadline {
                        return;
                    }
                    let (np, nq, seed) = plans[q];
                    let v = local_search(red, r, np, nq, seed, moves, effort, deadline);
                    s2.fetch_max(v, Ordering::Relaxed);
                });
            }
        });
        let after = threshold2.load(Ordering::Relaxed);
        if verbose {
            println!(
                "  grid: round {round}, {moves} moves, effort {effort}: bound {after} in {:.1}s",
                pass_start.elapsed().as_secs_f64()
            );
        }
        if after > before {
            secs = 0;
        } else {
            secs += 1;
        }
        round += 1;
    }
    threshold2.load(Ordering::Relaxed)
}
