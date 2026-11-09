#!/usr/bin/env bash
# Runs a Catch2 test executable in CI and turns its failed checks into
# annotations, which can be read without logging in (plan task 5.5).
#
#   scripts/ci_run_tests.sh "<title>" <executable> [Catch2 arguments...]
#
# The run's output goes to the log as usual; its exit status is the
# executable's.
title="$1"
shift
report="$(mktemp)"
status=0
"$@" --reporter compact > "$report" 2>&1 || status=$?
cat "$report"
if [ "$status" -ne 0 ]; then
    grep -E "failed:|FAILED|error:|fatal" "$report" | head -n 10 | while IFS= read -r line; do
        echo "::error title=${title}::${line:0:600}"
    done
    echo "::error title=${title}::exit status ${status}; last line: $(tail -n 1 "$report" | cut -c1-300)"
fi
rm -f "$report"
exit "$status"
