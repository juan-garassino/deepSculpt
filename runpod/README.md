# DeepSculpt on RunPod

Run DeepSculpt training, autonomous research, or self-improvement loops on a RunPod GPU pod, with checkpoints and results synced to Google Cloud Storage.

## What this directory provides

```
runpod/
├── Dockerfile           # CUDA 12.8 + PyTorch + Claude Code + gcloud
├── entrypoint.sh        # MODE dispatcher (train | research | improve)
├── Makefile             # build / push / run-* / sync-* / logs / stop
├── prompts/
│   ├── research.md      # Brief for autonomous research loop
│   └── improve.md       # Brief for ds-improve self-improvement loop
├── scripts/
│   ├── gcs_sync.py      # local ↔ GCS rsync helper
│   ├── checkpoint_io.py # backend-aware save/load (local | gcs)
│   └── run_research.sh  # standalone Claude Code launcher
└── README.md            # this file
```

## Modes

| `MODE` | What happens |
|---|---|
| `train` | Runs `python -m deepsculpt.main $TRAIN_CMD` against `${CKPT_DIR}` / `${DATA_DIR}`. |
| `research` | One-shot Claude Code with `prompts/research.md` — runs experiments until `TIME_BUDGET`, logs to `experiments.tsv`. |
| `improve` | One-shot Claude Code with `prompts/improve.md` — drives the `ds-improve` skill, exits at `TIME_BUDGET`. |
| `self-improve` | **Continuous** Claude loop: each iteration follows `prompts/self_improve.md` (which references `prompts/autoresearch_program.md`, the [020-autoresearch](../020-autoresearch/) brief). Sleeps `COOLDOWN` between iterations. Polls a **GCS-synced toggle** every `POLL_INTERVAL` to start/stop without restarting the pod. |

In all three modes the container:
1. Pulls existing state (`gs://$GCS_BUCKET/deepsculpt/checkpoints/$RUN_ID` and `/data`) at startup.
2. Periodically rsyncs back to GCS every `$SYNC_INTERVAL` seconds (default 600).
3. Does a final rsync on `EXIT` (crash-safe via `trap`).

## Required env at pod creation

| Var | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Required for `MODE=research` and `MODE=improve`. |
| `GCS_BUCKET` | Required. Default value used here: `garassino-ml-artifacts`. |
| `GOOGLE_APPLICATION_CREDENTIALS_JSON` | base64-encoded service-account JSON (decoded at runtime). |
| `MODE` | One of `train` \| `research` \| `improve`. Default `train`. |
| `RUN_ID` | Run identifier; used in GCS paths. Default = timestamp. |
| `TIME_BUDGET` | Seconds (Claude modes stop near this). Default 3600. |
| `TRAIN_CMD` | CLI subcommand for `MODE=train`. Default `train-gan`. |
| `TRAIN_ARGS` | Extra CLI args, e.g. `"--void-dim 64 --epochs 50"`. |
| `SYNC_INTERVAL` | Background rsync cadence in seconds. Default 600. |
| `ITER_BUDGET` | `MODE=self-improve` only: seconds per Claude iteration. Default 1800 (30 min). |
| `COOLDOWN` | `MODE=self-improve` only: seconds between iterations. Default 300. |
| `POLL_INTERVAL` | `MODE=self-improve` only: seconds between toggle checks. Default 60. |

## GCS layout

```
gs://garassino-ml-artifacts/deepsculpt/
├── data/<dataset_name>/...
├── checkpoints/<RUN_ID>/{generator.pt, discriminator.pt, optimizer.pt, config.yaml}
├── results/<RUN_ID>/{experiments.tsv, claude.log, samples/*.png, summary.md}
└── prompts-archive/<timestamp>-<mode>.md
```

## Deploy flow

```bash
# 1. Build + push (one-time, or whenever code changes)
export GHCR_USER=your-github-username
export GHCR_TOKEN=ghp_...           # PAT with write:packages
cd runpod
make login
make push

# 2. On RunPod (web UI):
#    - New GPU Pod → H100 or A100
#    - Container image: ghcr.io/juan-garassino/deepsculpt-runpod:latest
#    - Env vars (see table above)
#    - Start Pod — entrypoint auto-runs

# 3. Monitor
#    - RunPod UI → Logs tab
#    - Or after-the-fact via GCS:
gsutil cat gs://garassino-ml-artifacts/deepsculpt/results/<RUN_ID>/experiments.tsv

# 4. Resume after stop/crash
#    Start a new pod with the same RUN_ID — entrypoint pulls last state.
```

## Local smoke test (needs NVIDIA GPU)

```bash
cd runpod
# .env in the repo root or shell-exported
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_APPLICATION_CREDENTIALS_JSON=$(base64 -i ~/keys/gcs-sa.json)
make build
make run-train RUN_ID=local-smoke TRAIN_ARGS="--void-dim 16 --epochs 1 --batch-size 4"
make logs
# When done:
gsutil ls gs://garassino-ml-artifacts/deepsculpt/checkpoints/local-smoke/
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
