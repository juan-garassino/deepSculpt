#!/usr/bin/env bash
set -euo pipefail

# Drop-in wrapper around gsutil that re-reads the bearer token from the
# control file before each call. Use when you can't easily `source` the
# entrypoint helper (e.g. from a Python subprocess).
#
# Usage:
#   gcs_with_token.sh ls gs://garassino-ml-artifacts/deepsculpt/
#   gcs_with_token.sh -m rsync -r ./checkpoints gs://.../checkpoints/

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
TOKEN_FILE="${WORKSPACE_DIR}/control/gcs_token"

if [ -s "$TOKEN_FILE" ]; then
    CLOUDSDK_AUTH_ACCESS_TOKEN="$(cat "$TOKEN_FILE")"
    export CLOUDSDK_AUTH_ACCESS_TOKEN
fi

exec gsutil "$@"
