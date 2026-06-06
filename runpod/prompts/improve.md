# DeepSculpt — Autonomous Self-Improvement Loop

You are running on a RunPod GPU container. Your job: run the `ds-improve` skill against the DeepSculpt codebase — measure quality, diagnose failures, apply targeted fixes, re-measure, iterate.

## Ground rules

1. **Non-interactive.** No clarifying questions. Make calls and keep moving.
2. **Time budget is hard.** Stop when remaining time < 10 min; finalize, sync, commit.
3. **Use the `ds-improve` skill via the Skill tool** — it's the right entry point for this run.
4. **Atomic fixes.** One root cause per commit.
5. **Stay inside `/app` and `${WORKSPACE_DIR}`.**

## Setup

1. Read `CLAUDE.md`.
2. Skim `deepsculpt/main.py` and `config.yaml` for the CLI surface and hyperparameters.
3. Read available skill list under `.claude/skills/` — focus on `ds-improve`, `ds-gan`, `ds-diffusion`, `ds-dataval`.
4. Initialize `${RESULTS_DIR}/improvements.tsv` if missing:
   ```
   timestamp	commit	target_metric	before	after	diagnosis	fix
   ```

## The loop

Invoke the `ds-improve` skill. It will drive the cycle:

1. **Measure baseline** — run a small `train-gan` or `train-diffusion` job (`void_dim=32`, `epochs=3`, `batch_size=16`). Compute occupancy, diversity, FID proxy. Save baseline numbers.
2. **Diagnose** — pick the worst metric and trace the root cause (e.g. mode collapse → discriminator overwhelms generator; bad occupancy → sculptor generator bias; NaN loss → bad lr or unscaled grads).
3. **Fix** — apply the smallest targeted code change that addresses the root cause.
4. **Re-measure** — same training config as baseline. Record before/after.
5. **Log** — append to `improvements.tsv`. Commit. Sync to GCS.
6. **Pick next worst metric.** Repeat.

## Specific watch-list (from project history)

- **Mode collapse**: GAN diversity drops near zero. Likely cause: discriminator too strong → try LightDiscriminator, R1 regularization (already in code), or higher TTUR ratio.
- **Bad occupancy**: training shapes have mean occupancy 0.1-0.3; if model outputs 0.0 or 0.9, the occupancy loss weight is wrong.
- **NaN / Inf loss**: bad lr, missing grad clipping, AMP overflow. Check `mixed_precision`.
- **Diffusion samples noisy at end**: noise schedule too aggressive, or insufficient sampling steps.

## What good looks like

- Each row in `improvements.tsv` shows a measurable delta on a named metric.
- Commits have specific messages: `fix(gan): clamp R1 gamma to 0.5 — diversity 0.08 → 0.21`.
- No big refactors. No new dependencies.

## Done

When time budget < 10 min:
1. Write `${RESULTS_DIR}/summary.md` (one paragraph per fix attempted, what stuck, what didn't).
2. Final `gsutil rsync`.
3. Exit cleanly.

Go.
