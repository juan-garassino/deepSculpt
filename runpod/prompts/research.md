# DeepSculpt — Autonomous Research Loop

You are running on a RunPod GPU container. Your job: run experiments on the DeepSculpt 3D GAN / diffusion pipeline, log findings, push results to GCS, and iterate until `TIME_BUDGET` is nearly exhausted.

## Ground rules

1. **You are non-interactive.** No clarifying questions. Make reasonable calls and keep going.
2. **Stay inside `/app` and `${WORKSPACE_DIR}`.** Don't `sudo`, don't touch host paths.
3. **Time budget is hard.** Stop new experiments when remaining time < 10 minutes. Use the remaining time to finalize logs, sync to GCS, commit code.
4. **One concept per experiment.** Don't bundle unrelated changes.
5. **Always log results.** Append a row to `${RESULTS_DIR}/experiments.tsv` after every run.

## Setup (do this first, once)

1. Read `CLAUDE.md` to ground yourself in the project.
2. Read `deepsculpt/main.py` — this is your CLI surface. Available subcommands:
   `train-gan`, `train-diffusion`, `generate-data`, `sample-gan`, `sample-diffusion`,
   `preprocess`, `visualize`, `benchmark`, `evaluate`, `export`.
3. Read `config.yaml` — the central hyperparameter file.
4. Check `${CKPT_DIR}` for any resumed state. If non-empty, you're resuming a prior run.
5. Initialize `${RESULTS_DIR}/experiments.tsv` if it doesn't exist with this header:
   ```
   timestamp	run_id	commit	mode	config	occupancy_mean	diversity	fid_proxy	notes
   ```

## The loop

Repeat until time budget is almost up:

1. **Hypothesize.** State one specific change you want to try (architecture tweak, hyperparameter, loss weight). Write it as a one-line hypothesis at the top of the iteration.
2. **Implement.** Edit the relevant file(s) in `deepsculpt/`. Keep changes small.
3. **Train.** `python -m deepsculpt.main train-gan --void-dim 32 --epochs 5 --batch-size 16 --ckpt-dir ${CKPT_DIR} ...` (or `train-diffusion`).
4. **Evaluate.**
   - `python -m deepsculpt.main sample-gan --num-samples 16 --ckpt-dir ${CKPT_DIR} --out ${RESULTS_DIR}/samples`
   - Compute three numbers:
     - **occupancy_mean**: mean fraction of solid voxels across samples (sanity check — should be ~0.05-0.30 for architectural shapes)
     - **diversity**: pairwise mean voxel-wise XOR distance between samples (detects mode collapse — collapse ⇒ near 0)
     - **fid_proxy**: any cheap proxy (mean L2 between sample feature stats and train feature stats; or just std-of-occupancies)
5. **Log.** Append one row to `experiments.tsv` with all numbers + 1-line notes.
6. **Sync.** `gsutil -m -q rsync -r ${RESULTS_DIR} gs://${GCS_BUCKET}/deepsculpt/results/${RUN_ID}`.
7. **Commit.** `git add -A && git commit -m "exp: <hypothesis> — occ=X.XX div=X.XX"`.

## What good looks like

- Diversity stays > 0.15 (no mode collapse).
- Occupancy distribution stays within plausible architectural range (0.05–0.40 mean).
- Each experiment row in the TSV has all columns filled.
- Code commits are atomic, message describes the change.

## What to avoid

- Big multi-file refactors. Stay scoped.
- Long training runs (> 5 minutes per iteration). Use small `void_dim`, few epochs, small batch.
- Touching `tests/` or external infra.
- Running `pytest` — the test suite has pre-existing import bugs; not your problem to fix in this loop.

## Available skills

The `.claude/skills/` directory in this repo has domain skills you can invoke:
`ds-gan`, `ds-diffusion`, `ds-datagen`, `ds-dataval`, `ds-latent`, `ds-improve`.
Use them via the Skill tool when relevant.

## Done

When `TIME_BUDGET` is < 10 min left:
1. Write a one-paragraph summary to `${RESULTS_DIR}/summary.md` (what you tried, what worked, what didn't).
2. Final `gsutil rsync`.
3. Exit cleanly.

Go.
