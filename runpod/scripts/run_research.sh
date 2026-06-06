#!/usr/bin/env bash
set -euo pipefail

# Standalone wrapper to launch Claude Code with a chosen prompt.
# The main entrypoint.sh handles all this — this script exists for manual /
# in-container reruns: `runpod/scripts/run_research.sh research` etc.

MODE="${1:-research}"
PROMPT_FILE="/app/runpod/prompts/${MODE}.md"
RESULTS_DIR="${RESULTS_DIR:-/workspace/results/manual-$(date +%s)}"

if [ ! -f "$PROMPT_FILE" ]; then
    echo "ERROR: ${PROMPT_FILE} not found" >&2
    exit 1
fi
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "ERROR: ANTHROPIC_API_KEY not set" >&2
    exit 1
fi

mkdir -p "$RESULTS_DIR"
touch "${RESULTS_DIR}/claude.log"

exec claude --dangerously-skip-permissions \
    -p "$(cat "$PROMPT_FILE")" \
    --verbose 2>&1 | tee "${RESULTS_DIR}/claude.log"
