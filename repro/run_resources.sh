#!/usr/bin/env bash
set -uo pipefail
DATA=${1:?validation folder}
THREADS=${2:-16}
WORK=${VTR_WORK:?set VTR_WORK (see repro/setup_local.sh)}
export VTR_WORK
REPO=$(cd "$(dirname "$0")/.." && pwd)
R="$REPO/repro/official/run_official.py"
LOG="$REPO/repro/logs/resources"
mkdir -p "$LOG"
P04="$WORK/venv_timm04/bin/python"; P067="$WORK/venv_timm067/bin/python"
P092="$WORK/venv_timm092/bin/python"; PB="$WORK/venv_base/bin/python"
[ -x "$WORK/venv_atc/bin/python" ] && PA="$WORK/venv_atc/bin/python" || PA="$PB"
M=deit_small_patch16_224

run() {
  local tag="$2_${3:-default}"
  for mode in gpu_throughput gpu_latency cpu_latency; do
    case $mode in
      gpu_throughput) extra="--device cuda --throughput 256";;
      gpu_latency)    extra="--device cuda --latency 50";;
      cpu_latency)    extra="--device cpu --latency 50 --threads $THREADS";;
    esac
    echo "$(date '+%F %T') start $mode $tag" | tee -a "$LOG/progress.txt"
    if "$1" "$R" --model "$M" --method "$2" --setting "${3:-}" --data "$DATA" $extra \
         --out "$REPO/repro/resources_$mode.csv" > "$LOG/${mode}_$tag.log" 2>&1; then
      echo "$(date '+%F %T') done  $mode $tag $(grep '^{' "$LOG/${mode}_$tag.log" | tail -1)" | tee -a "$LOG/progress.txt"
    else
      echo "$(date '+%F %T') FAILED $mode $tag (see $LOG/${mode}_$tag.log)" | tee -a "$LOG/progress.txt"
    fi
  done
}

run "$P04" none ""
run "$P04" diffrate 2.9;     run "$P04" diffrate 2.3
run "$P04" tome 11;          run "$P04" tome 16
run "$P04" mctf 13;          run "$P04" mctf 19
run "$P04" pitome 0.925;     run "$P04" pitome 0.875
run "$P04" ppt 50
run "$PA"  atc 0.7;          run "$PA"  atc 0.5
run "$P04" topk 0.7;         run "$P04" topk 0.5
run "$P04" evit 0.7;         run "$P04" evit 0.5
run "$P092" gtp 11;          run "$P092" gtp 16
run "$PB"  reprshift 0.2;    run "$PB"  reprshift 0.31
run "$P067" papr 0.6;        run "$P067" papr 0.45
run "$P04" ats ""
echo "$(date '+%F %T') ALL DONE" | tee -a "$LOG/progress.txt"
