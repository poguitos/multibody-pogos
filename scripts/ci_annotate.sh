#!/usr/bin/env bash
# Runs a command in CI and, if it fails, turns its error lines and its last
# lines into annotations, which can be read without logging in (task 6.2;
# scripts/ci_run_tests.sh does the same for the Catch2 executables).
#
#   scripts/ci_annotate.sh "<title>" <command> [arguments...]
#
# The command's output goes to the log as usual; the exit status is its own.
title="$1"
shift
log="$(mktemp)"
status=0
"$@" > "$log" 2>&1 || status=$?
cat "$log"
if [ "$status" -ne 0 ]; then
    grep -E "error|Error|FAILED|undefined|relocation" "$log" | head -n 8 | while IFS= read -r line; do
        echo "::error title=${title}::${line:0:600}"
    done
    # The last lines in one annotation, each line break written as %0A.
    last=$(tail -n 15 "$log" | cut -c1-200 | sed -e 's/%/%25/g' -e 's/\r//g' | awk '{ printf "%s%%0A", $0 }')
    # No comma in a title: commas separate a workflow command's properties.
    echo "::error title=${title} (exit status ${status}; last lines)::${last}"
fi
rm -f "$log"
exit "$status"
