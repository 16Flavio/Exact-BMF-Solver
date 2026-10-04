#!/bin/bash
# Machine speed probe: heart r=2, started from solutions/heart_r2.txt, proves in about 10 s when the
# machine runs at full speed and in 25 to 30 s when it is throttled (battery, power
# plan, thermal throttling). Run it before a sweep, never during one: the
# solver takes all sixteen threads.
# usage, from the repository root:
#   bash experiments/probe.sh [binary]      (default target/release/bmf)
EXE="${1:-${EXE:-target/release/bmf}}"
j=$("$EXE" -i data/heart.txt -r 2 -t 60 --init solutions/heart_r2.txt --bb-time 0 --json 2>/dev/null)
tm=$(echo "$j" | grep -o '"time_s": [0-9.]*' | grep -o '[0-9.]*')
pr=$(echo "$j" | grep -o '"proven_optimal": [a-z]*' | sed 's/.*: //')
echo "heart r=2 : proven $pr in ${tm}s (full speed: about 10 s; throttled: 25 to 30 s)"
