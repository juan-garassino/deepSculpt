# DeepSculpt on RunPod

Run DeepSculpt training, autonomous research, or self-improvement loops on a RunPod GPU pod, with checkpoints and results synced to Google Cloud Storage.

## What this directory provides

```
runpod/
├── Dockerfile               # CUDA 12.8 + PyTorch + Claude Code + gcloud
├── entrypoint.sh            # MODE dispatcher (train | research | improve | self-improve)
├── Makefile                 # build / push / deploy / run-* / sync-* / toggle-* / logs / stop
├── prompts/
│   ├── research.md          # Brief for one-shot research loop
│   ├── improve.md           # Brief for one-shot ds-improve run
│   ├── self_improve.md      # Brief for continuous self-improve iteration
│   └── autoresearch_program.md   # Operational manual mirrored from 020-autoresearch
├── scripts/
│   ├── gcs_sync.py          # local ↔ GCS rsync helper
│   ├── checkpoint_io.py     # bearer-token-aware save/load (google.oauth2 Credentials)
│   ├── gcs_with_token.sh    # gsutil wrapper that reloads the token before each call
│   ├── token_writer.sh      # atomic-rename helper; called by refresh-token.yml via runpodctl exec
│   ├── self_improve_toggle.sh   # on/off/status — flips the GCS-synced toggle object
│   └── run_research.sh      # standalone Claude Code launcher (manual reruns)
└── README.md                # this file
```

## Modes

| `MODE` | What happens |
|---|---|
| `train` | Runs `python -m deepsculpt.main $TRAIN_CMD` against `${CKPT_DIR}` / `${DATA_DIR}`. |
| `research` | One-shot Claude Code with `prompts/research.md` — runs experiments until `TIME_BUDGET`, logs to `experiments.tsv`. |
| `improve` | One-shot Claude Code with `prompts/improve.md` — drives the `ds-improve` skill, exits at `TIME_BUDGET`. |
| `self-improve` | **Continuous** Claude loop: each iteration follows `prompts/self_improve.md` (which references `prompts/autoresearch_program.md`, the [020-autoresearch](../020-autoresearch/) brief). Sleeps `COOLDOWN` between iterations. Polls a **GCS-synced toggle** every `POLL_INTERVAL` to start/stop without restarting the pod. |

In all four modes the container:
1. Pulls existing state (`gs://$GCS_BUCKET/deepsculpt/checkpoints/$RUN_ID` and `/data`) at startup.
2. Periodically rsyncs back to GCS every `$SYNC_INTERVAL` seconds (default 600).
3. Does a final rsync on `EXIT` (crash-safe via `trap`).

## Required env at pod creation

| Var | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Required for `MODE=research`, `improve`, `self-improve`. Sourced from `~/.zshrc` → GHA repo secret. |
| `GCS_BUCKET` | Required. Default value used here: `garassino-ml-artifacts`. |
| `GCS_PROJECT` | `garassino-ml`. Informational. |
| `GCS_ACCESS_TOKEN` | **Short-lived OAuth2 bearer token** minted by GHA via Workload Identity Federation. Pod never sees a service-account JSON. Refreshed every 50 min by `.github/workflows/refresh-token.yml`. |
| `MODE` | One of `train` \| `research` \| `improve` \| `self-improve`. Default `train`. |
| `RUN_ID` | Run identifier; used in GCS paths. Default = timestamp. |
| `TIME_BUDGET` | Seconds (Claude modes stop near this). Default 3600. |
| `TRAIN_CMD` | CLI subcommand for `MODE=train`. Default `train-gan`. |
| `TRAIN_ARGS` | Extra CLI args, e.g. `"--void-dim 64 --epochs 50"`. |
| `SYNC_INTERVAL` | Background rsync cadence in seconds. Default 600. |
| `ITER_BUDGET` | `MODE=self-improve` only: seconds per Claude iteration. Default 1800 (30 min). |
| `COOLDOWN` | `MODE=self-improve` only: seconds between iterations. Default 300. |
| `POLL_INTERVAL` | `MODE=self-improve` only: seconds between toggle checks. Default 60. |

## Auth (no service-account JSON anywhere)

Per project policy ([root CLAUDE.md § GCP architecture](../../../CLAUDE.md)), pods are forbidden from carrying static GCP credentials. The flow:

1. GitHub Actions `deploy-runpod.yml` authenticates to GCP via **Workload Identity Federation** through `garassino-op`'s `gh-actions` pool. No JSON key.
2. The workflow impersonates `deepsculpt-runpod-runtime@garassino-ml.iam.gserviceaccount.com` and mints a 1-hour OAuth2 access token via `gcloud auth print-access-token`.
3. The token is passed to the RunPod pod as the `GCS_ACCESS_TOKEN` env var at create-time.
4. Inside the pod, `entrypoint.sh` writes the token to `/workspace/control/gcs_token` and exports `CLOUDSDK_AUTH_ACCESS_TOKEN` for `gsutil`.
5. `.github/workflows/refresh-token.yml` runs every 50 min, mints a fresh token, and pushes it into every running pod via `runpodctl exec` → `runpod/scripts/token_writer.sh` (atomic rename).
6. The pod's `gsutil` wrapper and the Python checkpoint helper re-read from the control file before every call, picking up the refresh transparently.

## GCS layout

```
gs://garassino-ml-artifacts/deepsculpt/
├── data/<dataset_name>/...
├── checkpoints/<RUN_ID>/{generator.pt, discriminator.pt, optimizer.pt, config.yaml}
├── results/<RUN_ID>/{experiments.tsv, claude.log, samples/*.png, summary.md}
└── prompts-archive/<timestamp>-<mode>.md
```

## Deploy flow (canonical: GHA, no local Docker required)

```bash
# 1. Image is built + pushed automatically by .github/workflows/build-push.yml
#    on every push to master that touches runpod/, deepsculpt/, or deps.

# 2. Deploy a pod
gh workflow run deploy-runpod.yml \
  -f mode=research \
  -f run_id=$(date +%Y%m%d-%H%M%S) \
  -f time_budget=3600 \
  -f gpu_type='NVIDIA A100 80GB PCIe'

# 3. Monitor
#    - RunPod UI → Logs tab
#    - Or via GCS (token via local `gcloud auth login`):
gsutil cat gs://garassino-ml-artifacts/deepsculpt/results/<RUN_ID>/experiments.tsv

# 4. Resume after crash
#    Re-run deploy-runpod.yml with the same run_id — entrypoint pulls last state.
```

## Local deploy (rare — only when iterating on the entrypoint)

Manual Docker requires that you mint a token yourself:

```bash
cd runpod
export ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY                  # from your ~/.zshrc
export GCS_ACCESS_TOKEN=$(CLOUDSDK_PYTHON=/usr/local/bin/python3.12 gcloud auth print-access-token)
make build
make run-train RUN_ID=local-smoke TRAIN_ARGS="--void-dim 16 --epochs 1 --batch-size 4"
make logs
make stop
```


## Continuous self-improvement (toggleable)

`MODE=self-improve` keeps the pod alive and re-launches Claude every `COOLDOWN` seconds, capped per-iteration by `ITER_BUDGET`. The loop polls a GCS-synced toggle every `POLL_INTERVAL` — flipping it does **not** require restarting the pod:

```bash
# Start the pod in self-improve mode (toggle defaults to OFF)
cd runpod
make run-self-improve API_KEY=sk-ant-... RUN_ID=long-running

# Turn the loop on (takes effect on next poll cycle, <=POLL_INTERVAL)
make toggle-on

# Turn it off again (current iteration finishes, no new ones launch)
make toggle-off

# Check state from anywhere with gsutil access
make toggle-status
```

The toggle is a single tiny object at `gs://$GCS_BUCKET/deepsculpt/control/self_improve.enabled` containing the literal text `on` or `off`. You can flip it from any machine with `gsutil` configured — pod, laptop, CI.

Each iteration is briefed by `prompts/self_improve.md`, which references `prompts/autoresearch_program.md` (the operational manual mirrored from the [020-autoresearch](../../020-autoresearch/) reference repo). The pattern is: read program.md → one focused improvement → log to `improvements.tsv` → commit → exit. The loop relaunches next cycle.

## Crash-safety

- `entrypoint.sh` runs a background `gsutil rsync` every `SYNC_INTERVAL` seconds.
- A `trap EXIT INT TERM` pushes one final time on container shutdown.
- Re-running with the **same `RUN_ID`** pulls the most recent checkpoint and resumes.

## Prompts

`prompts/research.md` and `prompts/improve.md` are the briefs Claude Code reads at startup. Each one is also archived to `gs://$GCS_BUCKET/deepsculpt/prompts-archive/` per run, so you can reproduce or audit later.

To customize a run without changing the image, edit the prompt locally and rebuild — or mount a different prompt file via `docker run -v ./my_prompt.md:/app/runpod/prompts/research.md ...`.
