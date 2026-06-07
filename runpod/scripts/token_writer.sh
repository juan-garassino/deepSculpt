#!/usr/bin/env bash
set -euo pipefail

# Receive a fresh GCS access token on stdin and write it atomically to
# the control file the running entrypoint reads from.
#
# Invoked from outside the pod via:
#   runpodctl exec <pod-id> bash -c 'cat | /app/runpod/scripts/token_writer.sh' < <(echo "$NEW_TOKEN")
#
# Atomicity: write to a sibling .new file in the same dir, then rename.
# On POSIX, rename is atomic — any concurrent reader sees either the old
# token or the new one, never a partial write.

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
CONTROL_DIR="${WORKSPACE_DIR}/control"
TOKEN_FILE="${CONTROL_DIR}/gcs_token"

mkdir -p "$CONTROL_DIR"
umask 077

TOKEN="$(cat)"
if [ -z "$TOKEN" ]; then
    echo "ERROR: empty token on stdin" >&2
    exit 1
fi

printf '%s' "$TOKEN" > "${TOKEN_FILE}.new"
mv "${TOKEN_FILE}.new" "$TOKEN_FILE"

echo "=== gcs_token refreshed at $(date -Iseconds) ==="
