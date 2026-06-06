# GCS layout

Bucket: `gs://garassino-ml-artifacts` (set via `GCS_BUCKET` env). All DeepSculpt state lives under the `deepsculpt/` prefix.

## Tree

```
gs://garassino-ml-artifacts/deepsculpt/
├── data/
│   └── <dataset_name>/
│       ├── architectural/{train,val}/*.pt
│       └── generic/{train,val}/*.pt
├── checkpoints/
│   └── <RUN_ID>/
│       ├── generator.pt
│       ├── discriminator.pt
│       ├── optimizer.pt
│       └── config.yaml
├── results/
│   └── <RUN_ID>/
│       ├── experiments.tsv         # appended per Claude iteration
│       ├── improvements.tsv        # MODE=improve only
│       ├── claude.log              # full Claude Code stdout
│       ├── train.log               # MODE=train stdout
│       ├── samples/*.png           # rendered sample sculptures
│       └── summary.md              # written at end of Claude run
└── prompts-archive/
    └── <YYYYmmdd-HHMMSS>-<mode>.md # snapshot of the prompt used
```

## `RUN_ID` convention

- Default: `$(date +%Y%m%d-%H%M%S)` — e.g. `20260606-142855`.
- For resume: **reuse the same RUN_ID across pod restarts**. The entrypoint pulls `checkpoints/<RUN_ID>` on startup before launching the mode.
- For comparison runs: encode the change, e.g. `RUN_ID=baseline`, `RUN_ID=r1gamma05`, `RUN_ID=light-disc`.

## `experiments.tsv` columns

| Column | Meaning |
|---|---|
| `timestamp` | ISO timestamp of the row |
| `run_id` | matches `RUN_ID` env |
| `commit` | `git rev-parse --short HEAD` at the time of the iteration |
| `mode` | `gan` \| `diffusion` |
| `config` | comma-separated key=value pairs (void_dim, epochs, batch_size, etc.) |
| `occupancy_mean` | mean fraction of solid voxels across sampled outputs |
| `diversity` | pairwise mean voxel XOR distance (mode-collapse detector) |
| `fid_proxy` | cheap proxy for sample quality |
| `notes` | one-line freeform |

## `improvements.tsv` columns (MODE=improve)

| Column | Meaning |
|---|---|
| `timestamp` | ISO timestamp |
| `commit` | post-fix `git rev-parse --short HEAD` |
| `target_metric` | which metric was being improved |
| `before` | baseline value |
| `after` | post-fix value |
| `diagnosis` | one-line root cause |
| `fix` | one-line description of the code change |

## Auth

The container reads `GOOGLE_APPLICATION_CREDENTIALS_JSON` (base64-encoded service-account key), decodes to `/tmp/gcp-key.json`, and runs `gcloud auth activate-service-account`. The SA needs roles `roles/storage.objectAdmin` on the bucket (or finer-grained on the `deepsculpt/` prefix).

## Local sync (dev convenience)

```bash
# Push local checkpoints to GCS for a run
make -C runpod sync-up KIND=checkpoints RUN_ID=my-experiment

# Pull data cache from GCS to local
make -C runpod sync-down KIND=data
```
