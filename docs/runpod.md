# RunPod deploy

GPU training + autonomous Claude research/improve runs on RunPod, with all state synced to GCS.

## TL;DR

```bash
cd runpod
make build && make push                 # one-time, or after code changes
# Then on RunPod: create GPU Pod → image=ghcr.io/juan-garassino/deepsculpt-runpod:latest
# Set env vars (see below) → Start Pod.
```

## Image

Built from `runpod/Dockerfile`. Base: `nvidia/cuda:12.8.0-devel-ubuntu22.04`. Bundles:
- Python 3.10, PyTorch (cu128 wheels), `requirements.txt`
- Node.js 20 + `@anthropic-ai/claude-code` (Claude Code CLI)
- Google Cloud SDK (`gsutil`, `gcloud`)
- The full DeepSculpt source (installed editable)
- `tini` as PID 1

Registry: GitHub Container Registry, `ghcr.io/juan-garassino/deepsculpt-runpod:latest`.

## Required env at pod creation

| Var | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | required for `MODE=research`, `improve`, `self-improve`. Sourced from `~/.zshrc` → GHA repo secret. |
| `GCS_BUCKET` | required. Default: `garassino-ml-artifacts`. Bucket is **`europe-west1`**. |
| `GCS_PROJECT` | `garassino-ml`. |
| `GCS_ACCESS_TOKEN` | **Short-lived OAuth2 bearer token** minted by GHA via WIF (no SA JSON). Refreshed every 50 min by `refresh-token.yml`. |
| `MODE` | `train` \| `research` \| `improve` \| `self-improve`. Default `train`. |
| `RUN_ID` | run identifier; used in GCS paths. Default = timestamp. **Reuse to resume.** |
| `TIME_BUDGET` | seconds (research/improve modes stop near this). Default 3600. |
| `TRAIN_CMD` | CLI subcommand for `MODE=train`. Default `train-gan`. |
| `TRAIN_ARGS` | extra CLI args, e.g. `"--void-dim 64 --epochs 50 --batch-size 16"`. |
| `SYNC_INTERVAL` | background `gsutil rsync` cadence in seconds. Default 600. |

## Modes

### `MODE=train`
Runs `python -m deepsculpt.main $TRAIN_CMD $TRAIN_ARGS --ckpt-dir $CKPT_DIR --data-dir $DATA_DIR`. Output goes to `${RESULTS_DIR}/train.log`.

### `MODE=research`
Launches `claude --dangerously-skip-permissions -p "$(cat runpod/prompts/research.md)" --verbose`. Claude runs experiments autonomously, logs each run to `experiments.tsv`, and stops near `TIME_BUDGET`. The prompt is archived to `gs://$GCS_BUCKET/deepsculpt/prompts-archive/<ts>-research.md` for reproducibility.

### `MODE=improve`
Same flow, but with `runpod/prompts/improve.md` — directs Claude to drive the `ds-improve` skill: measure → diagnose → fix → re-measure. **One-shot:** runs until `TIME_BUDGET`, then exits.

### `MODE=self-improve` (continuous + toggleable)
Same idea as `improve`, but **looped**. Each iteration: cap by `ITER_BUDGET` (default 1800s), use `prompts/self_improve.md` (which references `prompts/autoresearch_program.md` — mirrored from [020-autoresearch](../../020-autoresearch/)). Between iterations: sleep `COOLDOWN` (default 300s), poll `gs://$GCS_BUCKET/deepsculpt/control/self_improve.enabled` every `POLL_INTERVAL` (60s), only relaunch if the toggle reads `on`.

Flip the toggle from anywhere with `gsutil`:

```bash
# From the runpod/ directory
make toggle-on        # enables loop; takes effect within POLL_INTERVAL
make toggle-off       # disables; current iteration finishes naturally
make toggle-status    # prints current GCS state

# Or directly from any machine with gsutil configured
echo on | gsutil cp - gs://garassino-ml-artifacts/deepsculpt/control/self_improve.enabled
```

Stop the loop globally by setting `TIME_BUDGET` > 0 — the loop exits once elapsed time exceeds it. Set `TIME_BUDGET=0` for indefinite (toggle is the only stop).

## How auth works

Pods carry **no static GCP credentials**. Per project policy:

1. `.github/workflows/deploy-runpod.yml` authenticates to GCP via **Workload Identity Federation** (`garassino-op`'s `gh-actions` pool).
2. The workflow impersonates `deepsculpt-runpod-runtime@garassino-ml.iam.gserviceaccount.com` and calls `gcloud auth print-access-token` → 1-hour token.
3. Token is injected into the pod as `GCS_ACCESS_TOKEN`. Pod writes it to `/workspace/control/gcs_token`.
4. `.github/workflows/refresh-token.yml` runs every 50 min, mints a fresh token, pushes it into all running pods via `runpodctl exec`. The pod's `gsutil` wrapper re-reads the file before each call.

This satisfies the "no SA JSON anywhere" rule — no static key in git, in GHA secrets, in RunPod env, or on disk in the image.

## Lifecycle inside the pod

```
1. entrypoint.sh
2. Read GCS_ACCESS_TOKEN from env → write to /workspace/control/gcs_token (atomic rename)
3. Export CLOUDSDK_AUTH_ACCESS_TOKEN so gsutil uses bearer auth
4. gsutil rsync gs://.../checkpoints/$RUN_ID  →  /workspace/checkpoints/$RUN_ID  (resume)
5. gsutil rsync gs://.../data                  →  /workspace/data                (warm cache)
6. Spawn background loop: every SYNC_INTERVAL sec, reload token + push back to GCS
7. trap EXIT/INT/TERM: final rsync push
8. Dispatch on $MODE
```

## Crash safety

- Background rsync runs every 10 minutes by default.
- `trap` does a final push when the container exits (success, error, kill, SIGTERM).
- **Resume by starting a new pod with the same `RUN_ID`** — step 4 pulls last saved state.

## Local smoke test

Needs `docker --gpus all` (NVIDIA GPU + nvidia-container-toolkit) AND a fresh OAuth2 token:

```bash
cd runpod
export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY"
export GCS_ACCESS_TOKEN=$(CLOUDSDK_PYTHON=/usr/local/bin/python3.12 gcloud auth print-access-token)
make build
make run-train RUN_ID=local-smoke TRAIN_ARGS="--void-dim 16 --epochs 1 --batch-size 4"
make logs
# When done:
gsutil ls gs://garassino-ml-artifacts/deepsculpt/checkpoints/local-smoke/
make stop
```

Or skip Docker entirely and trigger the CI path: `make deploy MODE=train RUN_ID=...`.

## Monitoring a live run

- **Logs**: RunPod UI → Pod → Logs tab.
- **Mid-run results** (after the first rsync interval): `gsutil cat gs://.../results/$RUN_ID/experiments.tsv`.
- **Final summary**: `${RESULTS_DIR}/summary.md` (Claude writes this in research/improve modes).

## Costs

Pod is billed per minute by RunPod. `TIME_BUDGET` lets Claude stop itself before you forget. Always set it.

## Customizing the prompts

Edit `runpod/prompts/research.md` or `runpod/prompts/improve.md`, rebuild, push. For one-off variations without rebuilding, mount over the prompt:

```bash
docker run ... -v $PWD/my_prompt.md:/app/runpod/prompts/research.md ...
```
