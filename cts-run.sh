#!/bin/bash
# Run OpenCL CTS binaries against one driver configuration and keep the logs.
#
#   ./cts-run.sh <label> <driver> [suite...]
#     driver: system | local          (local = ./install via with-cl.sh)
#     suite:  CTS directory names, nested ones as images/clCopyImage; append
#             :arg,arg for arguments, e.g. math_brute_force:-w
#             (default: $DEFAULT_SUITES)
#   Extra env (e.g. ASAHI_PERFTEST=nooverlap) is passed through.
#
# Run it memory-capped (the image suites peak at ~3.4 GB of GPU memory, which
# the OOM killer cannot attribute to the test):
#   ./capped.sh 10G build-logs/cts.log ./cts-run.sh <label> local
#
# Output: cts-results/<label>/<suite>.log and summary.txt. Compare two runs
# with ./cts-compare.sh <labelA> <labelB>: the failure SETS must match, a pass
# count alone hides a swap.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LABEL=${1:?label}; DRIVER=${2:?driver}; shift 2
DEFAULT_SUITES="api basic buffers events atomics c11_atomics workgroups profiling mem_host_flags
  multiple_device_context non_uniform_work_group printf select relationals geometrics commonfns
  vectors half subgroups generic_address_space computeinfo device_timer compiler integer_ops"
SUITES=${*:-$DEFAULT_SUITES}
OUT=$HERE/cts-results/$LABEL
BIN=$HERE/cts-build/test_conformance
PER_SUITE_TIMEOUT=${PER_SUITE_TIMEOUT:-5400}
mkdir -p "$OUT"
case "$DRIVER" in
  system) RUN=() ;;
  local)  RUN=("$HERE/with-cl.sh") ;;
  *) echo "driver must be system or local" >&2; exit 2 ;;
esac
{
  echo "label=$LABEL driver=$DRIVER ASAHI_PERFTEST=${ASAHI_PERFTEST:-} date=$(date -Is)"
  [ "$DRIVER" = local ] && echo "mesa=$(git -C "${MESA_SRC:-$HERE/mesa}" log -1 --format=%h 2>/dev/null)"
} > "$OUT/summary.txt"
dmesg_before=$(sudo dmesg | grep -c "GPU timeout\|Fault info" || true)
for spec in $SUITES; do
  s=${spec%%:*}; args=(); [ "$spec" != "$s" ] && IFS=, read -r -a args <<< "${spec#*:}"
  name=${s//\//_}
  exe=$(ls "$BIN/$s"/test_* 2>/dev/null | grep -v '\.' | head -1)
  [ -x "$exe" ] || { echo "$s: no binary" | tee -a "$OUT/summary.txt"; continue; }
  start=$(date +%s)
  ( cd "$BIN/$s" && timeout "$PER_SUITE_TIMEOUT" "${RUN[@]}" "$exe" "${args[@]}" ) > "$OUT/$name.log" 2>&1
  rc=$?
  dt=$(( $(date +%s) - start ))
  line=$(grep -E "^(PASSED|FAILED|SKIPPED) " "$OUT/$name.log" | tail -1)
  printf '%-34s rc=%-3s %5ss  %s\n' "$spec" "$rc" "$dt" "${line:-no summary line}" | tee -a "$OUT/summary.txt"
done
dmesg_after=$(sudo dmesg | grep -c "GPU timeout\|Fault info" || true)
echo "gpu faults/timeouts during run: $((dmesg_after - dmesg_before))" | tee -a "$OUT/summary.txt"
"$HERE/scrub-logs.sh" "$OUT" >/dev/null
