#!/bin/bash
# Remove machine-identifying details from result logs before they are
# committed: the host name (the CTS prints "node name = ..."), the user's home
# directory, and the user name in any remaining path.
#
#   ./scrub-logs.sh [dir-or-file...]      default: cts-results results
#
# Idempotent. cts-run.sh calls it on every finished run.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE"
HOST=$(hostname -s)
HOSTFULL=$(hostname)
USERNAME=$(id -un)
targets=("$@")
[ ${#targets[@]} -eq 0 ] && targets=(cts-results results)

find "${targets[@]}" -type f \( -name '*.log' -o -name '*.txt' -o -name '*.json' \) -print0 2>/dev/null |
while IFS= read -r -d '' f; do
  sed -i \
    -e "s|$HOME|~|g" \
    -e "s|/home/$USERNAME|~|g" \
    -e "s|node name\( *\)= .*|node name\1= <host>|" \
    -e "s|\b$HOSTFULL\b|<host>|g" \
    -e "s|\b$HOST\b|<host>|g" \
    "$f"
done

left=$(grep -rlE "$HOST|/home/$USERNAME" "${targets[@]}" 2>/dev/null | wc -l)
echo "scrubbed; files still mentioning host/home: $left"
[ "$left" -eq 0 ]
