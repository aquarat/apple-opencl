#!/bin/bash
# Compare the per-test failure sets of two CTS runs from cts-run.sh.
#   ./cts-compare.sh <labelA> <labelB>
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
T=$(mktemp); A=$HERE/cts-results/$1; B=$HERE/cts-results/$2
fails() { # suite log -> "suite: test" for every failed subtest
  for f in "$1"/*.log; do
    s=$(basename "$f" .log)
    grep -E "^[A-Za-z0-9_]+ FAILED$|^[A-Za-z0-9_]+\.\.\.failed|FAILED test" "$f" 2>/dev/null |
      sed "s/^/$s: /"
    grep -q -E "^(PASSED|FAILED|SKIPPED) " "$f" || echo "$s: <no summary: crashed or timed out>"
  done | sort -u
}
diff <(fails "$A") <(fails "$B") > "$T" && echo "failure sets identical ($(fails "$A" | wc -l) failures)" || {
  echo "< only in $1   > only in $2"; cat "$T"; }
rm -f "$T"
