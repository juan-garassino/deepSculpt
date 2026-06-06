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
#   GOOGLE_APPLICATION_CREDENTIALS_JSON   base64-encoded service-account JSON
#
# Optional env:
#   RUN_ID            default: timestamp
#   TIME_BUDGET       seconds (Claude research loop will stop near this)
#   TRAIN_CMD         CLI subcommand for MODE=train (default: train-gan)
#   TRAIN_ARGS        extra CLI args
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

mkdir -p "$CKPT_DIR" "$RESULTS_DIR" "$DATA_DIR"

# ---------------------------------------------------------------------------
# Auth: GCS service-account key from env
# ---------------------------------------------------------------------------
if [ -z "${GCS_BUCKET:-}" ]; then
    echo "ERROR: GCS_BUCKET not set." >&2
    exit 1
fi

if [ -n "${GOOGLE_APPLICATION_CREDENTIALS_JSON:-}" ]; then
    echo "$GOOGLE_APPLICATION_CREDENTIALS_JSON" | base64 -d > /tmp/gcp-key.json
    export GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcp-key.json
    gcloud auth activate-service-account --key-file=/tmp/gcp-key.json --quiet
elif [ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" ] && [ -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    gcloud auth activate-service-account --key-file="$GOOGLE_APPLICATION_CREDENTIALS" --quiet
else
    echo "WARN: no GCS credentials provided — gsutil will fail on private buckets." >&2
fi

# ---------------------------------------------------------------------------
# Pull existing checkpoints + data for this RUN_ID (resume support)
# ---------------------------------------------------------------------------
GCS_ROOT="gs://${GCS_BUCKET}/deepsculpt"

echo "=== Pulling state from ${GCS_ROOT} (RUN_ID=${RUN_ID}) ==="
gsutil -m -q rsync -r "${GCS_ROOT}/checkpoints/${RUN_ID}" "$CKPT_DIR" 2>/dev/null || \
    echo "  (no prior checkpoints for run ${RUN_ID} — fresh start)"
gsutil -m -q rsync -r "${GCS_ROOT}/data" "$DATA_DIR" 2>/dev/null || \
    echo "  (no warm data cache)"

# ---------------------------------------------------------------------------
# Background periodic checkpoint sync (crash-safe)
# ---------------------------------------------------------------------------
push_state() {
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
        echo "=== Training: deepsculpt ${TRAIN_CMD} ${TRAIN_ARGS} ==="
        python -m deepsculpt.main "$TRAIN_CMD" \
            --ckpt-dir "$CKPT_DIR" \
            --data-dir "$DATA_DIR" \
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
