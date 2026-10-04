#!/bin/bash
# Every measurement of the README, without an initial solution and from the
# stored factorization of solutions/. Blocks run in order and each one
# resumes: a row already in its CSV is not replayed, so the script can simply
# be relaunched after an interruption. Creating experiments/results/STOP makes
# it exit after the current run. A block can be run alone: ONLY="3 5" runs
# blocks 3 and 5.
#
# Before each block the machine speed is probed (heart r=2 from its stored
# factorization, about 10 s at full speed); while the probe is slower than
# 13 s the script waits and probes again, so a throttled machine delays the
# measurements instead of spoiling them; with NOWAIT=1 it goes on regardless.
#
# Needs: target/release/bmf; for blocks 3 and 5, gurobi_cl on PATH (or in
# GUROBI_CL), Python 3 with OR-tools and PySAT (pip install ortools python-sat).
# Budget: about 30 h for blocks 1 to 6, and up to 15 h for block 7.
# usage, from the repository root:
#   cargo build --release
#   bash experiments/run.sh
# Results go to experiments/results/*.csv.
set -u
EXE="${EXE:-target/release/bmf}"
D=experiments/results
GUROBI="${GUROBI_CL:-gurobi_cl}"
MAIN="zoo:2 zoo:5 zoo:10 heart:2 heart:5 heart:10 lymp:2 lymp:5 lymp:10 apb:2 apb:5 apb:10 \
tumor_w_missing:2 tumor_w_missing:5 tumor_w_missing:10 hepatitis_w_missing:2 hepatitis_w_missing:5 \
hepatitis_w_missing:10 audio_w_missing:2 audio_w_missing:5 audio_w_missing:10 votes_w_missing:2 \
votes_w_missing:5 votes_w_missing:10"
OPTIMA="zoo:1 zoo:2 zoo:3 zoo:4 zoo:5 heart:1 heart:2 lymp:1 lymp:2 apb:1 tumor_w_missing:2 \
tumor_w_missing:3 hepatitis_w_missing:2 audio_w_missing:2 votes_w_missing:2 votes_w_missing:3"
EXTRA="zoo:1 zoo:3 zoo:4 heart:1 lymp:1 apb:1 tumor_w_missing:3 votes_w_missing:3"
GENERIC="zoo:2 zoo:5 heart:2 lymp:2 apb:1 tumor_w_missing:2 hepatitis_w_missing:2"
LONG="zoo:6 zoo:7 apb:2 heart:3 apb:3 lymp:3 tumor_w_missing:3 hepatitis_w_missing:3 audio_w_missing:3 votes_w_missing:3"
mkdir -p "$D/tmp"

log() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$D/run.log"; }
stop() { [ -f "$D/STOP" ] && { log "STOP file found, exiting"; exit 0; }; }
block() { [ -z "${ONLY:-}" ] || [[ " $ONLY " == *" $1 "* ]]; }
probe() {
  while :; do
    stop
    t=$(bash experiments/probe.sh "$EXE" | grep -o 'in [0-9.]*s' | grep -o '[0-9.]*')
    log "probe: ${t:-failed} s"
    [ -n "$t" ] && [ "${t%.*}" -le 13 ] && return
    [ -n "${NOWAIT:-}" ] && { log "slow machine, going on (NOWAIT)"; return; }
    log "machine throttled, waiting 10 min"; sleep 600
  done
}
done_row() { [ -f "$1" ] && grep -q "^$2," "$1"; }
json() { echo "$1" | grep -o "\"$2\": [a-z0-9.]*" | sed 's/.*: //'; }
header() { [ -f "$1" ] || echo "$2" > "$1"; }
# bmf <csv> <key> <file> <rank> <args...>: one run, one row "key,lb,ub,proven,nodes,time".
bmf() {
  local out=$1 key=$2 f=$3 r=$4; shift 4
  done_row "$out" "$key" && return
  stop
  local j; j=$("$EXE" -i "$f" -r "$r" --json "$@" 2>/dev/null)
  echo "$key,$(json "$j" lb),$(json "$j" ub),$(json "$j" proven_optimal),$(json "$j" bb_nodes),$(json "$j" time_s)" >> "$out"
  log "$out $key $(json "$j" lb) $(json "$j" ub) $(json "$j" proven_optimal) $(json "$j" time_s)s"
}
# gurobi <csv> <key> <file> <rank> [start]: the compact program written by bmf,
# 120 s on 16 threads, with the start as a MIP start if given.
gurobi() {
  local out=$1 key=$2 f=$3 r=$4 s=${5:-}
  done_row "$out" "$key" && return
  stop
  local mst=() in=()
  [ -n "$s" ] && { mst=(--init "$s" --mst "$D/tmp/m.mst"); in=(InputFile=m.mst); }
  "$EXE" -i "$f" -r "$r" --mip "$D/tmp/m.lp" "${mst[@]}" > /dev/null
  local g; g=$(cd "$D/tmp" && "$GUROBI" TimeLimit=120 Threads=16 "${in[@]}" LogFile= m.lp 2>&1)
  local st ob bd tm
  st=$(echo "$g" | grep -q 'Optimal solution found' && echo optimal || echo time_limit)
  ob=$(echo "$g" | grep -o 'Best objective [-0-9.e+]*' | sed 's/Best objective //' | awk '{printf "%.0f", $1}')
  bd=$(echo "$g" | grep -o 'best bound [-0-9.e+]*' | sed 's/best bound //' | awk '{printf "%.0f", $1}')
  tm=$(echo "$g" | grep -o 'in [0-9.]* seconds' | grep -o '[0-9.]*' | tail -1)
  echo "$key,$st,$ob,$bd,$tm" >> "$out"; log "$out $key $st $ob $bd"
}
# cpsat <csv> <key> <file> <rank> [start]: 600 s, 8 workers, the start as a hint.
cpsat() {
  local out=$1 key=$2 f=$3 r=$4 s=${5:-}
  done_row "$out" "$key" && return
  stop
  local l; l=$(python experiments/cpsat.py "$f" "$r" 600 8 $s 2>/dev/null | tail -n 1)
  echo "$key,$(echo "$l" | cut -d, -f3-6)" >> "$out"; log "$out $key $l"
}
# rc2 <csv> <key> <file> <rank>: one thread, stopped after 120 s, last bound kept.
rc2() {
  local out=$1 key=$2 f=$3 r=$4
  done_row "$out" "$key" && return
  stop
  "$EXE" -i "$f" -r "$r" --wcnf "$D/tmp/m.wcnf" > /dev/null
  local o; o=$(timeout 120 python -u experiments/rc2.py "$D/tmp/m.wcnf" 2>/dev/null)
  local st=time_limit; echo "$o" | grep -q 'END OPTIMUM' && st=optimal
  local bd; bd=$(echo "$o" | grep -o 'c cost: [0-9]*' | tail -1 | grep -o '[0-9]*')
  echo "$key,$st,,$bd," >> "$out"; log "$out $key $st $bd"
}

# 1. The 24 main cells and the certified optima: five runs from the empty and
#    from the stored factorization, 60 s (600 s for the rank 3 cells).
if block 1; then
  OUT=$D/bench.csv; header "$OUT" "instance,r,start,run,lb,ub,proven,nodes,time"
  probe
  for run in 1 2 3 4 5; do
    for c in $MAIN $EXTRA; do
      f=${c%%:*}; r=${c##*:}; T=60; [ "$r" = 3 ] && T=600
      bmf "$OUT" "$f,$r,none,$run" "data/$f.txt" "$r" -t $T --bb-time 0
      bmf "$OUT" "$f,$r,start,$run" "data/$f.txt" "$r" -t $T --bb-time 0 --init "solutions/${f}_r$r.txt"
    done
    probe
  done
fi

# 2. Other routes to each certificate, from the stored factorization: one
#    thread, reversed column order, node bound alone.
if block 2; then
  OUT=$D/routes.csv; header "$OUT" "instance,r,route,lb,ub,proven,nodes,time"
  probe
  for c in $OPTIMA; do
    f=${c%%:*}; r=${c##*:}; S="solutions/${f}_r$r.txt"
    bmf "$OUT" "$f,$r,one_thread" "data/$f.txt" "$r" -t 600 --bb-time 0 --init "$S" --threads 1
    bmf "$OUT" "$f,$r,reversed" "data/$f.txt" "$r" -t 600 --bb-time 0 --init "$S" --order reverse
    bmf "$OUT" "$f,$r,node_bound" "data/$f.txt" "$r" -t 600 --bb-time 0 --init "$S" --without residuals,grid
  done
fi

# 3. Generic exact solvers: Gurobi and CP-SAT from the stored factorization on
#    every optimum, and on the comparison cells also without it, with RC2;
#    bmf on one thread on the comparison cells.
if block 3; then
  OUT=$D/external.csv; header "$OUT" "instance,r,solver,status,objective,bound,time"
  probe
  for c in $OPTIMA; do
    f=${c%%:*}; r=${c##*:}; S="solutions/${f}_r$r.txt"
    gurobi "$OUT" "$f,$r,gurobi_start" "data/$f.txt" "$r" "$S"
    cpsat "$OUT" "$f,$r,cpsat_start" "data/$f.txt" "$r" "$S"
  done
  for c in $GENERIC; do
    f=${c%%:*}; r=${c##*:}
    gurobi "$OUT" "$f,$r,gurobi_none" "data/$f.txt" "$r"
    cpsat "$OUT" "$f,$r,cpsat_none" "data/$f.txt" "$r"
    rc2 "$OUT" "$f,$r,rc2" "data/$f.txt" "$r"
  done
  OUT=$D/one_thread.csv; header "$OUT" "instance,r,start,lb,ub,proven,nodes,time"
  for c in $GENERIC; do
    f=${c%%:*}; r=${c##*:}
    bmf "$OUT" "$f,$r,none" "data/$f.txt" "$r" -t 60 --bb-time 0 --threads 1
    bmf "$OUT" "$f,$r,start" "data/$f.txt" "$r" -t 60 --bb-time 0 --threads 1 --init "solutions/${f}_r$r.txt"
  done
fi

# 4. Ablation, from the stored factorization, one run per variant.
if block 4; then
  OUT=$D/ablation.csv; header "$OUT" "instance,r,variant,lb,ub,proven,nodes,time"
  probe
  for v in residuals ladder inherit empty grid watcher; do
    for c in $MAIN; do
      f=${c%%:*}; r=${c##*:}; S="solutions/${f}_r$r.txt"
      if [ "$v" = watcher ]; then opt="--margin 0"; else opt="--without $v"; fi
      bmf "$OUT" "$f,$r,$v" "data/$f.txt" "$r" -t 60 --bb-time 0 --init "$S" $opt
    done
    probe
  done
fi

# 5. The synthetic family: bmf and CIP, four processes of four threads.
if block 5; then
  probe; stop
  python experiments/generate_synth.py
  python experiments/synth.py "$EXE" "$GUROBI" 2>&1 | tee -a "$D/run.log"
fi

# 6. Ten minutes on the cells nearest to a proof, and the watcher's margin,
#    from the stored factorization.
if block 6; then
  OUT=$D/long.csv; header "$OUT" "instance,r,run,lb,ub,proven,nodes,time"
  probe
  for run in 1 2 3; do
    for c in $LONG; do
      f=${c%%:*}; r=${c##*:}
      bmf "$OUT" "$f,$r,$run" "data/$f.txt" "$r" -t 600 --bb-time 0 --init "solutions/${f}_r$r.txt"
    done
    probe
  done
  OUT=$D/margin.csv; header "$OUT" "instance,r,margin,lb,ub,proven,nodes,time"
  for c in zoo:6 heart:3 apb:2; do
    f=${c%%:*}; r=${c##*:}
    for mg in 200 0; do
      bmf "$OUT" "$f,$r,$mg" "data/$f.txt" "$r" -t 600 --bb-time 0 --init "solutions/${f}_r$r.txt" --margin $mg
    done
  done
fi

# 7. Scaling on planted matrices, 300 s per run, from the planted factorization.
if block 7; then
  stop
  python experiments/generate_scaling.py
  python experiments/scaling.py "$EXE" 300 start 2>&1 | tee -a "$D/run.log"
fi
log "done"
