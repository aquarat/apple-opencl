#!/bin/bash
# Run a command in a memory-capped systemd user scope and report its peak.
#
#   ./capped.sh <max> <logfile> <command...>        e.g. ./capped.sh 10G out.log ./test_basic
#
# GPU buffer objects on Asahi are pinned shmem: they cannot be swapped and the
# OOM killer cannot attribute them to the process that allocated them, so an
# uncapped GPU test that over-allocates kills unrelated processes (it took out
# the user's systemd manager and D-Bus on 2026-10-03). They ARE charged to the
# allocating cgroup, so MemoryMax contains them: an over-allocation then kills
# only the test. Environment is passed through (PREFIX, ASAHI_PERFTEST, ...).
set -uo pipefail
MAX=${1:?max, e.g. 8G}; LOGF=$(realpath -m "${2:?log file}"); shift 2
UNIT=capped-$$-$RANDOM
envargs=()
for v in NOKERNEL PREFIX MESA_SHADER_CACHE_DIR MESA_SHADER_CACHE_DISABLE ASAHI_PERFTEST AGX_MESA_DEBUG \
         RUSTICL_DEBUG AGX_CDM_BARRIER_MASK AGX_BO_CACHE_MB ASAHI_MESA_DEBUG OCL_ICD_VENDORS; do
  [ -n "${!v:-}" ] && envargs+=(--setenv="$v=${!v}")
done
: > "$LOGF"
start=$(date +%s)
systemd-run --user --quiet --wait --collect --unit="$UNIT" \
  -p MemoryMax="$MAX" -p MemorySwapMax=0 -p StandardOutput="append:$LOGF" -p StandardError="append:$LOGF" \
  --working-directory="$PWD" "${envargs[@]}" "$@" &
waiter=$!
peak=0
sleep 0.3
while kill -0 $waiter 2>/dev/null; do
  s=$(awk '/^Shmem:/{print int($2/1024)}' /proc/meminfo)
  [ "$s" -gt "$peak" ] && peak=$s
  sleep 0.5
done
wait $waiter; rc=$?
echo "rc=$rc  $(( $(date +%s) - start ))s  peak system Shmem ${peak} MB (cap $MAX)"
exit $rc
