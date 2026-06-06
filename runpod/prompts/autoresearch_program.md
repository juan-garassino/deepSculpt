# DeepSculpt Autoresearch Program

This repo supports an autonomous experiment loop for DeepSculpt training.

## Mission

Improve training behavior for DeepSculpt using bounded experiments on one GPU.

Primary targets:
1. `gan` — improve monochrome GAN training without empty-collapse
2. `diffusion` — improve diffusion training quality and stability

The active default mode is provided by the environment variable `AUTORESEARCH_MODE`.

## Setup

Before starting experiments:

1. Create a fresh branch named `autoresearch/<tag>`.
   - Use today's date for the tag, for example `autoresearch/apr15-gan`.
   - Do not reuse an existing branch.
2. Read these files:
   - `README.md`
   - `autoresearch/README.md`
   - `autoresearch/program.md`
   - `Makefile`
   - `scripts/colab_train.py`
   - `scripts/colab_train_diffusion.py`
   - `scripts/autoresearch_report.py`
3. Initialize `autoresearch/results.tsv` if it does not exist.
4. Confirm the experiment directories from the environment:
   - `AUTORESEARCH_DATA_OUT`
   - `AUTORESEARCH_RUN_OUT`
5. Start with one baseline run before making code changes.

## Constraints

What you may change:
- `deepsculpt/`
- `scripts/`
- `Makefile`
- `autoresearch/`
- small docs updates if needed

What you should avoid changing unless absolutely necessary:
- notebooks
- old boilerplate folders
- unrelated service deployment files

Do not delete datasets or previous run artifacts.

## Commands

### GAN baseline / experiments

Use:

```bash
make colab-train-mono DATA_OUT="$AUTORESEARCH_DATA_OUT" RUN_OUT="$AUTORESEARCH_RUN_OUT" DATA_MODE=reuse NUM_SAMPLES="${AUTORESEARCH_NUM_SAMPLES:-500}" EPOCHS="${AUTORESEARCH_EPOCHS:-5}" BATCH_SIZE="${AUTORESEARCH_BATCH_SIZE:-4}" VOID_DIM="${AUTORESEARCH_VOID_DIM:-32}" NOISE_DIM="${AUTORESEARCH_NOISE_DIM:-100}" NUM_INFERENCE_SAMPLES=1 NUM_WORKERS=0 R1_GAMMA="${AUTORESEARCH_R1_GAMMA:-2.0}" GAN_AUGMENT="${AUTORESEARCH_GAN_AUGMENT:-none}" GAN_AUGMENT_TARGET="${AUTORESEARCH_GAN_AUGMENT_TARGET:-0.7}" OCCUPANCY_LOSS_WEIGHT="${AUTORESEARCH_OCCUPANCY_LOSS_WEIGHT:-5.0}" OCCUPANCY_FLOOR="${AUTORESEARCH_OCCUPANCY_FLOOR:-0.01}" OCCUPANCY_TARGET_MODE="${AUTORESEARCH_OCCUPANCY_TARGET_MODE:-batch_real}" EMA_DECAY="${AUTORESEARCH_EMA_DECAY:-0.999}"
```

### Diffusion baseline / experiments

Use:

```bash
make colab-train-diffusion DATA_OUT="$AUTORESEARCH_DATA_OUT" RUN_OUT="$AUTORESEARCH_RUN_OUT" DATA_MODE=reuse NUM_SAMPLES="${AUTORESEARCH_NUM_SAMPLES:-500}" EPOCHS="${AUTORESEARCH_EPOCHS:-5}" BATCH_SIZE="${AUTORESEARCH_BATCH_SIZE:-4}" VOID_DIM="${AUTORESEARCH_VOID_DIM:-32}" TIMESTEPS="${AUTORESEARCH_TIMESTEPS:-100}" LEARNING_RATE="${AUTORESEARCH_LR:-1e-4}" WEIGHT_DECAY="${AUTORESEARCH_WEIGHT_DECAY:-0.01}" NOISE_SCHEDULE="${AUTORESEARCH_NOISE_SCHEDULE:-cosine}" NUM_INFERENCE_SAMPLES=1 NUM_INFERENCE_STEPS="${AUTORESEARCH_NUM_INFERENCE_STEPS:-50}" NUM_WORKERS=0 EMA_DECAY="${AUTORESEARCH_EMA_DECAY:-0.9999}"
```

### Run scoring

After every run, score the latest run with:

```bash
python scripts/autoresearch_report.py --mode "$AUTORESEARCH_MODE" --run-output "$AUTORESEARCH_RUN_OUT"
```

This prints JSON. Lower `score` is better.

## Results log

Use `autoresearch/results.tsv` with this header:

```tsv
commit	mode	score	status	run_dir	summary
```

Where:
- `commit` is the 7-char git hash
- `mode` is `gan` or `diffusion`
- `score` is the numeric score from `scripts/autoresearch_report.py`
- `status` is `keep`, `discard`, or `crash`
- `run_dir` is the full run directory path
- `summary` is a short description of what changed and what happened

Do not commit `results.tsv`.

## Experiment loop

Loop forever until interrupted:

1. Check the current branch and commit.
2. Choose one concrete experiment.
3. Make the code change.
4. Commit it.
5. Run the appropriate `make` command.
6. Score the run with `scripts/autoresearch_report.py`.
7. Append the result to `autoresearch/results.tsv`.
8. If the score improved, keep the commit and continue from there.
9. If the score got worse or the run crashed, revert to the previous commit and continue.

## Decision policy

### GAN

Primary goal:
- prevent empty-collapse

Strong positive signs:
- fake occupancy stays close to real occupancy
- collapse warnings disappear
- `scripts/autoresearch_report.py` returns `suggested_status=keep`

Strong negative signs:
- fake occupancy collapses toward zero
- collapse warnings dominate
- helper reports `collapsed=true`

If there is any clear sign of collapse:
- stop treating the current direction as viable
- mark the experiment as `discard`
- revert or change the code
- run a new experiment immediately

Do not keep training iterations that already show obvious collapse just because they finished cleanly.
For GAN, collapse is a hard failure signal, not a soft warning.

### Diffusion

Primary goal:
- lower diffusion loss without breaking training

Positive signs:
- lower final loss
- healthy summaries and inference outputs

## Research priorities

1. First establish a baseline in the active mode.
2. Prefer simple, defensible changes over random churn.
3. For GAN, focus on anti-collapse, occupancy behavior, and training stability.
4. For diffusion, focus on trainer defaults, EMA behavior, scheduler/sampler coherence, and stable sample quality.
5. If GAN repeatedly collapses despite bounded tuning, switch effort toward diffusion rather than wasting cycles.

## Important behavior

- Do not ask the human for permission between experiments.
- Keep experiments bounded and reviewable.
- Prefer one clean idea per commit.
- If a run crashes, inspect the traceback, make a focused fix if it is worthwhile, otherwise discard and move on.
- After every run, use `scripts/autoresearch_report.py` as the decision gate.
- If the report says `collapsed=true` for GAN, treat the run as failed immediately and move to the next code change.
