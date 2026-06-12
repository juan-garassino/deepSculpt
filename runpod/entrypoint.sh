#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# DeepSculpt RunPod entrypoint
#   MODE=train         → run deepsculpt CLI training
#   MODE=research      → launch Claude Code with prompts/research.md (one-shot)
#   MODE=improve       → launch Claude Code with prompts/improve.md (one-shot)
#   MODE=self-improve  → continuous improve loop; toggleable on/off via
#                        gs://$GCS_BUCKET/deepsculpt/control/self_improve.enabled
#                        Mirrors the 020-autoresearch pattern + autoresearch/program.md.
#
# Required env:
#   ANTHROPIC_API_KEY                 (research / improve only)
#   GCS_BUCKET                        e.g. garassino-ml-artifacts
#   GCS_PROJECT                       e.g. garassino-ml (informational; used by gcloud)
#   GCS_ACCESS_TOKEN                  short-lived OAuth2 token minted by GHA via WIF.
#                                     Pod NEVER receives a service-account JSON key.
#                                     Refreshed by .github/workflows/refresh-token.yml
#                                     every 50 min via `runpodctl exec` writing to
#                                     /workspace/control/gcs_token.
#
# Optional env:
#   RUN_ID            default: timestamp
#   TIME_BUDGET       seconds (Claude research loop will stop near this)
#   TRAIN_CMD         CLI subcommand for MODE=train (default: train-gan)
#   TRAIN_ARGS        extra CLI args (e.g. "--model-type skip --epochs 200")
#   DATA_SAMPLES      MODE=train: samples to generate if DATA_DIR is empty (default 2000)
#   VOID_DIM          MODE=train: voxel grid size for generated data (default 64)
#   SYNC_INTERVAL     seconds between periodic checkpoint pushes (default 600)
#   ITER_BUDGET       MODE=self-improve: seconds per claude iteration (default 1800)
#   COOLDOWN          MODE=self-improve: seconds between iterations (default 300)
#   POLL_INTERVAL     MODE=self-improve: seconds between toggle checks (default 60)
# ---------------------------------------------------------------------------

MODE="${MODE:-train}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
TIME_BUDGET="${TIME_BUDGET:-3600}"
SYNC_INTERVAL="${SYNC_INTERVAL:-600}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
TRAIN_CMD="${TRAIN_CMD:-train-gan}"
TRAIN_ARGS="${TRAIN_ARGS:-}"
ITER_BUDGET="${ITER_BUDGET:-1800}"
COOLDOWN="${COOLDOWN:-300}"
POLL_INTERVAL="${POLL_INTERVAL:-60}"

CKPT_DIR="${WORKSPACE_DIR}/checkpoints/${RUN_ID}"
RESULTS_DIR="${WORKSPACE_DIR}/results/${RUN_ID}"
DATA_DIR="${WORKSPACE_DIR}/data"
CONTROL_DIR="${WORKSPACE_DIR}/control"
GCS_TOKEN_FILE="${CONTROL_DIR}/gcs_token"

mkdir -p "$CKPT_DIR" "$RESULTS_DIR" "$DATA_DIR" "$CONTROL_DIR"

# ---------------------------------------------------------------------------
# Auth: short-lived bearer token from env, refreshed by GHA cron
# ---------------------------------------------------------------------------
if [ -z "${GCS_BUCKET:-}" ]; then
    echo "ERROR: GCS_BUCKET not set." >&2
    exit 1
fi

if [ -n "${GCS_ACCESS_TOKEN:-}" ]; then
    # Atomic write: rename is atomic on POSIX, refreshers may overwrite this
    # file concurrently while gsutil is reading it.
    umask 077
    printf '%s' "$GCS_ACCESS_TOKEN" > "${GCS_TOKEN_FILE}.new"
    mv "${GCS_TOKEN_FILE}.new" "$GCS_TOKEN_FILE"
    export CLOUDSDK_AUTH_ACCESS_TOKEN="$GCS_ACCESS_TOKEN"
    [ -n "${GCS_PROJECT:-}" ] && export CLOUDSDK_CORE_PROJECT="$GCS_PROJECT"
    echo "=== GCS auth: short-lived token (refresher writes to ${GCS_TOKEN_FILE}) ==="
elif [ -s "$GCS_TOKEN_FILE" ]; then
    export CLOUDSDK_AUTH_ACCESS_TOKEN="$(cat "$GCS_TOKEN_FILE")"
    [ -n "${GCS_PROJECT:-}" ] && export CLOUDSDK_CORE_PROJECT="$GCS_PROJECT"
    echo "=== GCS auth: existing token at ${GCS_TOKEN_FILE} ==="
else
    # On Vertex AI the job runs AS the runtime SA — gsutil/gcloud use ADC via
    # the metadata server, so no bearer token is needed (RunPod-only plumbing).
    echo "=== GCS auth: no bearer token — relying on ADC (Vertex AI / metadata server) ==="
fi

# Helper: refresh gsutil's view of the bearer token from the control file.
# Called before every gsutil invocation in case the refresher updated it.
gcs_reload_token() {
    if [ -s "$GCS_TOKEN_FILE" ]; then
        CLOUDSDK_AUTH_ACCESS_TOKEN="$(cat "$GCS_TOKEN_FILE")"
        export CLOUDSDK_AUTH_ACCESS_TOKEN
    fi
}
export -f gcs_reload_token

# ---------------------------------------------------------------------------
# Pull existing checkpoints + data for this RUN_ID (resume support)
# ---------------------------------------------------------------------------
GCS_ROOT="gs://${GCS_BUCKET}/deepsculpt"

echo "=== Pulling state from ${GCS_ROOT} (RUN_ID=${RUN_ID}) ==="
gcs_reload_token
gsutil -m -q rsync -r "${GCS_ROOT}/checkpoints/${RUN_ID}" "$CKPT_DIR" 2>/dev/null || \
    echo "  (no prior checkpoints for run ${RUN_ID} — fresh start)"
gcs_reload_token
gsutil -m -q rsync -r "${GCS_ROOT}/data" "$DATA_DIR" 2>/dev/null || \
    echo "  (no warm data cache)"

# ---------------------------------------------------------------------------
# Background periodic checkpoint sync (crash-safe)
# ---------------------------------------------------------------------------
push_state() {
    gcs_reload_token
    gsutil -m -q rsync -r "$CKPT_DIR" "${GCS_ROOT}/checkpoints/${RUN_ID}" || true
    gsutil -m -q rsync -r "$RESULTS_DIR" "${GCS_ROOT}/results/${RUN_ID}" || true
}

(
    while true; do
        sleep "$SYNC_INTERVAL"
        push_state
    done
) &
SYNC_PID=$!

# Always push on exit
cleanup() {
    echo "=== Final sync to GCS ==="
    kill "$SYNC_PID" 2>/dev/null || true
    push_state
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
export RUN_ID GCS_BUCKET CKPT_DIR RESULTS_DIR DATA_DIR TIME_BUDGET

echo "=== MODE=${MODE} RUN_ID=${RUN_ID} ==="

case "$MODE" in
    train)
        # Bootstrap training data if neither the GCS warm cache nor a prior
        # run populated DATA_DIR.
        if [ -z "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
            echo "=== No data in ${DATA_DIR} — generating ${DATA_SAMPLES:-2000} samples ==="
            python -m deepsculpt.main generate-data \
                --num-samples "${DATA_SAMPLES:-2000}" \
                --void-dim "${VOID_DIM:-64}" \
                --output-dir "$DATA_DIR"
            gcs_reload_token
            gsutil -m -q rsync -r "$DATA_DIR" "${GCS_ROOT}/data" || true
        fi

        echo "=== Training: deepsculpt ${TRAIN_CMD} ${TRAIN_ARGS} ==="
        python -m deepsculpt.main "$TRAIN_CMD" \
            --data-folder "$DATA_DIR" \
            --output-dir "$CKPT_DIR" \
            ${TRAIN_ARGS} 2>&1 | tee "${RESULTS_DIR}/train.log"
        ;;

    research|improve)
        if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
            echo "ERROR: ANTHROPIC_API_KEY required for MODE=${MODE}" >&2
            exit 1
        fi
        export ANTHROPIC_API_KEY

        PROMPT_FILE="/app/runpod/prompts/${MODE}.md"
        if [ ! -f "$PROMPT_FILE" ]; then
            echo "ERROR: prompt file ${PROMPT_FILE} not found" >&2
            exit 1
        fi

        # Archive the prompt with timestamp for reproducibility
        gsutil -m -q cp "$PROMPT_FILE" \
            "${GCS_ROOT}/prompts-archive/$(date +%Y%m%d-%H%M%S)-${MODE}.md" || true

        # Build the runtime prompt: prepend env context
        RUNTIME_PROMPT="/tmp/runtime_prompt.md"
        {
            echo "# Runtime context (do not ignore)"
            echo ""
            echo "- RUN_ID: ${RUN_ID}"
            echo "- CKPT_DIR: ${CKPT_DIR}"
            echo "- RESULTS_DIR: ${RESULTS_DIR}"
            echo "- DATA_DIR: ${DATA_DIR}"
            echo "- GCS_ROOT: ${GCS_ROOT}"
            echo "- TIME_BUDGET: ${TIME_BUDGET} seconds"
            echo ""
            echo "---"
            echo ""
            cat "$PROMPT_FILE"
        } > "$RUNTIME_PROMPT"

        touch "${RESULTS_DIR}/claude.log"
        echo "=== Launching Claude Code (MODE=${MODE}) ==="
        claude --dangerously-skip-permissions \
            -p "$(cat "$RUNTIME_PROMPT")" \
            --verbose 2>&1 | tee "${RESULTS_DIR}/claude.log"
        ;;

    self-improve)
        # Continuous self-improvement loop, toggleable on/off.
        if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
            echo "ERROR: ANTHROPIC_API_KEY required for MODE=self-improve" >&2
            exit 1
        fi
        export ANTHROPIC_API_KEY

        PROMPT_FILE="/app/runpod/prompts/self_improve.md"
        CONTROL_LOCAL="${WORKSPACE_DIR}/control/self_improve.enabled"
        CONTROL_GCS="${GCS_ROOT}/control/self_improve.enabled"
        mkdir -p "$(dirname "$CONTROL_LOCAL")"

        # Archive the prompt + autoresearch brief for reproducibility
        gsutil -m -q cp "$PROMPT_FILE" \
            "${GCS_ROOT}/prompts-archive/$(date +%Y%m%d-%H%M%S)-self-improve.md" || true
        gsutil -m -q cp /app/runpod/prompts/autoresearch_program.md \
            "${GCS_ROOT}/prompts-archive/$(date +%Y%m%d-%H%M%S)-autoresearch-program.md" || true

        touch "${RESULTS_DIR}/claude.log"
        start_ts=$(date +%s)
        iter=0

        echo "=== Self-improve loop start (ITER_BUDGET=${ITER_BUDGET}s, COOLDOWN=${COOLDOWN}s) ==="
        echo "=== Toggle: gsutil cp - ${CONTROL_GCS} <<< on    (or 'off') ==="

        while true; do
            # Global time budget
            if [ "$TIME_BUDGET" -gt 0 ]; then
                elapsed=$(($(date +%s) - start_ts))
                if [ "$elapsed" -ge "$TIME_BUDGET" ]; then
                    echo "=== Global TIME_BUDGET (${TIME_BUDGET}s) exhausted, exiting loop ==="
                    break
                fi
            fi

            # Refresh toggle from GCS (canonical state)
            gsutil -q cp "$CONTROL_GCS" "$CONTROL_LOCAL" 2>/dev/null || true

            state="off"
            if [ -s "$CONTROL_LOCAL" ] && grep -qi "^on" "$CONTROL_LOCAL"; then
                state="on"
            fi

            if [ "$state" != "on" ]; then
                echo "$(date -Iseconds) self-improve: OFF (iter=${iter}) — sleeping ${POLL_INTERVAL}s"
                sleep "$POLL_INTERVAL"
                continue
            fi

            iter=$((iter + 1))
            echo "=== Self-improve iter ${iter} (budget ${ITER_BUDGET}s) ==="

            # Build per-iteration prompt with live context
            RUNTIME_PROMPT="/tmp/self_improve_iter.md"
            {
                echo "# Self-improve iteration ${iter}"
                echo ""
                echo "- RUN_ID: ${RUN_ID}"
                echo "- iter: ${iter}"
                echo "- CKPT_DIR: ${CKPT_DIR}"
                echo "- RESULTS_DIR: ${RESULTS_DIR}"
                echo "- DATA_DIR: ${DATA_DIR}"
                echo "- GCS_ROOT: ${GCS_ROOT}"
                echo "- ITER_BUDGET: ${ITER_BUDGET} seconds"
                echo ""
                echo "Read \`runpod/prompts/autoresearch_program.md\` (operational manual — mirrors 020-autoresearch)"
                echo "before iterating. Apply ONE small improvement, log to \`${RESULTS_DIR}/improvements.tsv\`,"
                echo "commit, then exit. The orchestrator will relaunch you next cycle."
                echo ""
                echo "---"
                echo ""
                cat "$PROMPT_FILE"
            } > "$RUNTIME_PROMPT"

            timeout "$ITER_BUDGET" claude --dangerously-skip-permissions \
                -p "$(cat "$RUNTIME_PROMPT")" \
                --verbose 2>&1 | tee -a "${RESULTS_DIR}/claude.log" \
                || echo "=== iter ${iter} ended (timeout or error) ==="

            push_state
            echo "=== Cooldown ${COOLDOWN}s before iter $((iter + 1)) ==="
            sleep "$COOLDOWN"
        done
        ;;

    *)
        echo "ERROR: unknown MODE='${MODE}' (use train|research|improve|self-improve)" >&2
        exit 1
        ;;
esac

echo "=== DONE (MODE=${MODE} RUN_ID=${RUN_ID}) ==="
