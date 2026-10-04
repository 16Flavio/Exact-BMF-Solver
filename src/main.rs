//! Command line driver for the exact solver.
//!
//! Phases, in order:
//!
//!  0. lossless reduction of the matrix (`reduce`);
//!  1. upper bound from an initial factorization (`--init`), or the empty one;
//!  2. quick grid bound (`bounds`);
//!  3. branch and bound on column patterns (`bb`), the part that proves;
//!  4. refined grid bound, if the optimum is still unproven.
//!
//! The output is always a bracket `[LB, UB]`, even after an interruption.

use bmf::{bb, bounds, interruption, io, mip, subproblem, verify, wcnf};

use bmf::reduce::Reduced;
use bmf::bits::BitVec;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Parsed command line options.
struct Args {
    input: Option<PathBuf>,
    rank: usize,
    time_limit: f64,
    init: Option<PathBuf>,
    bb_time: f64,
    no_bb: bool,
    json: bool,
    verbose: bool,
    verify: bool,
    out: Option<PathBuf>,
    nthreads: Option<usize>,
    mip: Option<PathBuf>,
    mst: Option<PathBuf>,
    wcnf: Option<PathBuf>,
    margin: u128,
    without: Vec<String>,
    reverse_order: bool,
    sort_children: bool,
}

fn usage() -> ! {
    eprintln!(
        "usage:
  bmf -i <file> -r <rank> [options]

  -i, --input <F>        binary matrix (packed digits or spaces, header m n optional)
  -r, --rank <R>         factorization rank
  -t, --time-limit <S>   total budget in seconds, default 600; 0 for no
                         limit, the optimum will be proven or Ctrl+C gives back control
      --init <F>         initial factorization (W and H, as written by -o);
                         without it the solver starts from the empty one
      --bb-time <S>      budget of the branch and bound on patterns, default 60;
                         0 for no limit
      --no-bb            skips the branch and bound, returns only the initial
                         factorization and the combinatorial bound
      --mip <F>          writes the integer program in LP format and
                         exits, for comparison with an off the shelf solver
      --mst <F>          with --mip and --init, also writes the initial
                         factorization as a MIP start (Gurobi MST format)
      --wcnf <F>         writes the problem as weighted MaxSAT (WCNF) and exits
  -o, --out <F>          writes W and H
      --threads <N>      number of threads, default: every core
      --margin <M>       watcher margin, default 50; 0 to never cut
      --without <list>   ablation: switches off residuals, ladder, inherit, empty
                         or grid (comma separated)
      --order reverse    least discriminating columns first
      --sort-children    experiment: open a node's children by increasing bound
      --json             JSON output
      --verify           consistency checks
  -v, --verbose          progress log

Ctrl+C interrupts cleanly: the solver returns the bracket [LB, UB] it reached.
"
    );
    std::process::exit(2)
}

fn parse() -> Args {
    let mut a = Args {
        input: None,
        rank: 0,
        time_limit: 600.0,
        init: None,
        bb_time: 60.0,
        no_bb: false,
        json: false,
        verbose: false,
        verify: false,
        out: None,
        nthreads: None,
        margin: bb::Settings::default().margin,
        without: Vec::new(),
        reverse_order: false,
        sort_children: false,
        mip: None,
        mst: None,
        wcnf: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(t) = it.next() {
        let mut val = |flag: &str| -> String {
            it.next().unwrap_or_else(|| {
                eprintln!("missing value after {flag}");
                usage()
            })
        };
        match t.as_str() {
            "-i" | "--input" => a.input = Some(PathBuf::from(val("-i"))),
            "-r" | "--rank" => a.rank = val("-r").parse().unwrap_or_else(|_| usage()),
            "-t" | "--time-limit" => a.time_limit = val("-t").parse().unwrap_or_else(|_| usage()),
            "--init" => a.init = Some(PathBuf::from(val("--init"))),
            "-o" | "--out" => a.out = Some(PathBuf::from(val("-o"))),
            "--threads" => a.nthreads = Some(val("--threads").parse().unwrap_or_else(|_| usage())),
            "--margin" => a.margin = val("--margin").parse().unwrap_or_else(|_| usage()),
            "--without" => {
                a.without = val("--without").split(',').map(|s| s.to_string()).collect();
                let valid = ["residuals", "ladder", "inherit", "empty", "grid"];
                if a.without.iter().any(|s| !valid.contains(&s.as_str())) {
                    usage();
                }
            }
            "--sort-children" => a.sort_children = true,
            "--order" => {
                let o = val("--order");
                if o != "reverse" && o != "direct" {
                    usage();
                }
                a.reverse_order = o == "reverse";
            }
            "--bb-time" => a.bb_time = val("--bb-time").parse().unwrap_or_else(|_| usage()),
            "--no-bb" => a.no_bb = true,
            "--mip" => a.mip = Some(PathBuf::from(val("--mip"))),
            "--mst" => a.mst = Some(PathBuf::from(val("--mst"))),
            "--wcnf" => a.wcnf = Some(PathBuf::from(val("--wcnf"))),
            "--json" => a.json = true,
            "--verify" => a.verify = true,
            "-v" | "--verbose" => a.verbose = true,
            "-h" | "--help" => usage(),
            _ if t.starts_with('-') => {
                eprintln!("unknown option: {t}");
                usage()
            }
            _ => a.input = Some(PathBuf::from(t)),
        }
    }
    a
}

/// Maps a rectangle of the original matrix to the reduced one. A reduced row or
/// column is kept only if the rectangle contains its whole group, so the result
/// never grows.
fn project(red: &Reduced, w_orig: &BitVec, h_orig: &BitVec) -> Option<(BitVec, BitVec)> {
    let (wt, ht) = if red.transposed {
        (h_orig, w_orig)
    } else {
        (w_orig, h_orig)
    };
    let mut w = BitVec::zeros(red.m);
    let mut h = BitVec::zeros(red.n);
    for i in 0..red.m {
        if red.row_src[i].iter().all(|&s| wt.get(s)) {
            w.set(i, true);
        }
    }
    for j in 0..red.n {
        if red.col_src[j].iter().all(|&s| ht.get(s)) {
            h.set(j, true);
        }
    }
    if w.count() == 0 || h.count() == 0 {
        None
    } else {
        Some((w, h))
    }
}

/// Reads the initial factorization and pads it with empty rectangles up to
/// `rank`. Exits on a malformed file or on dimensions that do not match.
fn load_init(path: &Path, raw: &io::RawMat, rank: usize) -> (Vec<Vec<bool>>, Vec<Vec<bool>>) {
    let (mut w, mut h) = io::load_factors(path).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1)
    });
    let r = h.len();
    if w.len() != raw.m || h.iter().any(|row| row.len() != raw.n) || w.iter().any(|row| row.len() != r) {
        eprintln!(
            "{}: factors do not match the {}x{} matrix",
            path.display(),
            raw.m,
            raw.n
        );
        std::process::exit(1);
    }
    if r > rank {
        eprintln!("{}: rank {r} above the requested rank {rank}", path.display());
        std::process::exit(1);
    }
    for row in w.iter_mut() {
        row.resize(rank, false);
    }
    h.resize(rank, vec![false; raw.n]);
    (w, h)
}

fn main() {
    let a = parse();
    let file = a.input.clone().unwrap_or_else(|| usage());
    if a.rank == 0 {
        eprintln!("-r <rank> is required");
        usage()
    }
    interruption::install();
    let start = Instant::now();
    // A zero budget means no limit: only a proof or Ctrl+C stops the solver.
    let deadline = if a.time_limit > 0.0 {
        start + Duration::from_secs_f64(a.time_limit)
    } else {
        interruption::unlimited()
    };

    let raw = match io::load(Path::new(&file)) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1)
        }
    };
    let red = Reduced::build(&raw);
    let missing = raw.missing();
    if !a.json {
        if missing == 0 {
            println!(
                "instance : {} ({}x{}, {} ones)",
                raw.name,
                raw.m,
                raw.n,
                raw.ones()
            );
        } else {
            println!(
                "instance : {} ({}x{}, {} ones, {} missing entries)",
                raw.name,
                raw.m,
                raw.n,
                raw.ones(),
                missing
            );
        }
        println!(
            "reduced  : {}x{}{}",
            red.m,
            red.n,
            if red.transposed { " (transposed)" } else { "" }
        );
        println!("rank     : {}", a.rank);
    }

    // Model export only: write the file and exit.
    if let Some(f) = &a.wcnf {
        if let Err(e) = std::fs::write(f, wcnf::write_wcnf(&red, a.rank)) {
            eprintln!("write {}: {e}", f.display());
            std::process::exit(1);
        }
        if !a.json {
            println!("MaxSAT written to {}", f.display());
        }
        return;
    }
    if let Some(f) = &a.mip {
        let lp = mip::write_lp(&red, a.rank);
        if let Err(e) = std::fs::write(f, lp) {
            eprintln!("write {}: {e}", f.display());
            std::process::exit(1);
        }
        if !a.json {
            println!("model written to {}", f.display());
        }
        // A MIP start from the initial factorization, for the same model.
        if let Some(g) = &a.mst {
            let init = a.init.as_ref().unwrap_or_else(|| {
                eprintln!("--mst needs --init");
                std::process::exit(2)
            });
            let (w, h) = load_init(init, &raw, a.rank);
            let (sa, sb) = mip::project_start(&red, &w, &h);
            let e_red = mip::start_error(&red, &sa, &sb);
            let e_raw = verify::naive_error(&raw, &w, &h);
            if e_red != e_raw {
                eprintln!("warning: the start has error {e_red} on the reduced matrix, {e_raw} on the original");
            }
            if let Err(e) = std::fs::write(g, mip::write_mst(&red, &sa, &sb)) {
                eprintln!("write {}: {e}", g.display());
                std::process::exit(1);
            }
            if !a.json {
                println!("start of error {e_red} written to {}", g.display());
            }
        }
        return;
    }

    // Phase 1: the initial factorization, given on the original matrix, or the
    // empty one. The reduction is lossless, so its error is also an upper
    // bound for the reduced problem.
    let t_init = Instant::now();
    let mut sol: (Vec<Vec<bool>>, Vec<Vec<bool>>) = match &a.init {
        Some(f) => load_init(f, &raw, a.rank),
        None => (vec![vec![false; a.rank]; raw.m], vec![vec![false; raw.n]; a.rank]),
    };
    let mut ub = verify::naive_error(&raw, &sol.0, &sol.1);
    let mut rects_ub: Vec<(BitVec, BitVec)> = Vec::new();
    {
        // Project the solution into the reduced frame and polish it there.
        let mut rects: Vec<(BitVec, BitVec)> = Vec::new();
        for k in 0..a.rank {
            let mut w_orig = BitVec::zeros(raw.m);
            let mut h_orig = BitVec::zeros(raw.n);
            for i in 0..raw.m {
                w_orig.set(i, sol.0[i][k]);
            }
            for j in 0..raw.n {
                h_orig.set(j, sol.1[k][j]);
            }
            match project(&red, &w_orig, &h_orig) {
                Some(q) => rects.push(q),
                None => rects.push((BitVec::zeros(red.m), BitVec::zeros(red.n))),
            }
        }
        if missing == 0 {
            // Alternate exact solves of both factors.
            if a.rank <= subproblem::MAX_ZETA_RANK {
                subproblem::ao_exact(&red, &mut rects);
            }
        } else {
            // The zeta transform does not handle missing entries, so only W is
            // recomputed exactly, with H fixed.
            let patterns: Vec<BitVec> = rects.iter().map(|(_, h)| h.clone()).collect();
            let polished = bb::factorize(&red, &patterns);
            if red.rects_error(&polished) < red.rects_error(&rects) {
                rects = polished;
            }
        }
        // Keep the polished factorization only if it improves on the given one.
        let e = red.rects_error(&rects);
        if e < ub {
            ub = e;
            sol = red.expand(&rects);
            rects_ub = rects;
        }
    }

    // The weighted error on the reduced matrix must equal the raw error.
    if a.verify || cfg!(debug_assertions) {
        if let Err(e) = verify::check_reduction(&raw, &red, &rects_ub) {
            eprintln!("VERIFICATION: {e}");
            std::process::exit(3);
        }
    }
    // The reported upper bound is always recounted from the returned solution
    // on the original matrix, so it cannot drift from what is written out.
    let raw_err = verify::naive_error(&raw, &sol.0, &sol.1);
    if missing == 0 {
        assert_eq!(
            raw_err, ub,
            "the returned solution does not match the announced error"
        );
    }
    ub = raw_err;

    if !a.json {
        println!(
            "phase 1  : UB {ub} from {} in {:.3}s",
            a.init
                .as_ref()
                .map(|f| f.display().to_string())
                .unwrap_or_else(|| "the empty factorization".to_string()),
            t_init.elapsed().as_secs_f64()
        );
    }

    // Phase 2: grid bound, the isolation bound summed over blocks.
    let nthreads = a.nthreads.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    });
    let t_bounds = Instant::now();
    // A short sweep only (at most 1 s). The expensive local search runs in
    // phase 4, and only if the branch and bound does not prove.
    let bounds_cutoff = Instant::now()
        + (deadline.saturating_duration_since(Instant::now()) / 100).min(Duration::from_secs(1));
    let no_grid = a.without.iter().any(|s| s == "grid");
    let lb_iso = if no_grid {
        0
    } else {
        bounds::grid_bound(&red, a.rank, nthreads, 0, false, bounds_cutoff, false)
    };
    if !a.json {
        println!(
            "phase 2  : grid bound {lb_iso} in {:.2}s",
            t_bounds.elapsed().as_secs_f64()
        );
    }

    let mut lb = lb_iso.min(ub);
    let mut proved = lb >= ub;
    let mut interrupted = false;
    let mut bb_nodes = 0u64;

    // Phase 3: branch and bound on column patterns.
    if !proved && !a.no_bb {
        let cutoff = if a.bb_time > 0.0 {
            (Instant::now() + Duration::from_secs_f64(a.bb_time)).min(deadline)
        } else {
            deadline
        };
        // Keep an eighth of the budget (at most 60 s) for phase 4.
        let reserve =
            (cutoff.saturating_duration_since(Instant::now()) / 8).min(Duration::from_secs(60));
        let cutoff = cutoff - reserve;
        let settings = bb::Settings {
            margin: a.margin,
            no_residuals: a.without.iter().any(|s| s == "residuals"),
            no_ladder: a.without.iter().any(|s| s == "ladder"),
            no_inherit: a.without.iter().any(|s| s == "inherit"),
            no_empty: a.without.iter().any(|s| s == "empty"),
            reverse_order: a.reverse_order,
            sort_children: a.sort_children,
        };
        let s = bb::solve(&red, a.rank, ub, lb, nthreads, cutoff, a.verbose && !a.json, settings);
        // The tree beat the EA: rebuild the factors from the column patterns
        // and check the error both on the reduced and on the original matrix.
        if !s.patterns.is_empty() && s.ub < ub {
            let rects = bb::factorize(&red, &s.patterns);
            assert_eq!(
                red.rects_error(&rects),
                s.ub,
                "the branch and bound got the error wrong"
            );
            if a.verify || cfg!(debug_assertions) {
                if let Err(msg) = verify::check_reduction(&raw, &red, &rects) {
                    eprintln!("VERIFICATION: {msg}");
                    std::process::exit(3);
                }
            }
            ub = s.ub;
            sol = red.expand(&rects);
            assert_eq!(
                verify::naive_error(&raw, &sol.0, &sol.1),
                ub,
                "inconsistent solution"
            );
        }
        lb = lb.max(s.lb).min(ub);
        proved = s.complete;
        bb_nodes = s.nodes;
        // Distinguishes an exhausted budget from a user stop.
        interrupted = !s.complete && !interruption::requested();
        if !a.json {
            if s.complete {
                println!("phase 3  : branch and bound on column patterns, certified optimum {ub}");
            } else {
                println!(
                    "phase 3  : branch and bound on column patterns stopped, LB {} after {} nodes",
                    s.lb, s.nodes
                );
            }
        }
    }

    // Phase 4: refined grid bound with the remaining time.
    if !proved && !no_grid {
        let t2 = Instant::now();
        // After a first Ctrl+C the bound still gets one minute; a second
        // Ctrl+C exits immediately.
        let cutoff2 = if interruption::requested() {
            Instant::now() + Duration::from_secs(60)
        } else {
            deadline
        };
        let lb2 = bounds::grid_bound(&red, a.rank, nthreads, lb, true, cutoff2, a.verbose && !a.json);
        if lb2 > lb {
            if !a.json {
                println!(
                    "phase 4  : refined grid bound {lb2} in {:.2}s",
                    t2.elapsed().as_secs_f64()
                );
            }
            lb = lb2.min(ub);
            proved = lb >= ub;
        }

    }

    let time_limit = start.elapsed().as_secs_f64();
    let gap = ub.saturating_sub(lb);

    if a.json {
        println!("{{");
        println!("  \"instance\": \"{}\",", raw.name);
        println!(
            "  \"m\": {}, \"n\": {}, \"rank\": {},",
            raw.m, raw.n, a.rank
        );
        println!("  \"lb\": {lb}, \"ub\": {ub}, \"gap\": {gap},");
        println!("  \"proven_optimal\": {proved},");
        println!("  \"interrupted\": {interrupted},");
        println!("  \"bb_nodes\": {bb_nodes},");
        println!("  \"interrupted_by_user\": {},", interruption::requested());
        println!("  \"time_s\": {time_limit:.3}");
        println!("}}");
    } else {
        println!("--------------------------------------------");
        println!("LB       : {lb}");
        println!("UB       : {ub}");
        println!(
            "gap      : {gap}  ({})",
            if proved {
                "OPTIMAL, proven".to_string()
            } else if interruption::requested() {
                "interrupted".to_string()
            } else if interrupted {
                "budget exhausted".to_string()
            } else {
                "not proven".to_string()
            }
        );
        if bb_nodes > 0 {
            println!("bb       : {bb_nodes} nodes");
        }
        println!("time     : {time_limit:.2}s");
        println!("--------------------------------------------");
    }

    if let Some(o) = a.out {
        if let Err(e) = std::fs::write(&o, io::dump_factors(&sol.0, &sol.1)) {
            eprintln!("write {}: {e}", o.display());
        }
    }
}
