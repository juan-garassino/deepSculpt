# DeepSculpt — Continuous Self-Improvement (one iteration)

You are running inside a long-lived RunPod container, **invoked once per iteration** by the self-improve loop in `runpod/entrypoint.sh`. Each iteration has its own time budget (`ITER_BUDGET`, default 30 min). Between iterations the loop sleeps for `COOLDOWN` seconds, then checks a GCS toggle (`gs://$GCS_BUCKET/deepsculpt/control/self_improve.enabled`) and either relaunches you or stays off. Your job is **one focused improvement** per iteration, not an entire campaign.

## Read this before doing anything

1. **`runpod/prompts/autoresearch_program.md`** — the operational manual. Mirrors the `020-autoresearch` reference repo's `program.md`. Treat it as the authoritative spec for branch convention, results-tsv format, and the experiment-loop discipline.
2. **`CLAUDE.md`** — project context and the `ds-improve` skill trigger.
3. **`deepsculpt/main.py`** + **`config.yaml`** — CLI surface and hyperparameter source of truth.
4. **`${RESULTS_DIR}/improvements.tsv`** — what previous iterations have already tried (so you don't repeat them).

## Ground rules

1. **Non-interactive.** No clarifying questions. Make the reasonable call.
2. **One improvement per iteration.** Pick the single worst metric or the most obvious bug; fix it; measure.
3. **Stop near `ITER_BUDGET`.** Leave ≥1 minute for the final commit + GCS push.
4. **Atomic commits.** One root cause per commit. Format: `improve(<area>): <change> — <metric> <before>→<after>`.
5. **Stay inside `/app` and `${WORKSPACE_DIR}`.** Don't `sudo`, don't change CI, don't touch `tests/` cleanup unless it's the root cause.

## Per-iteration flow

1. **Pick the target.**
   - On iter 1: run a baseline (`train-gan --void-dim 32 --epochs 3 --batch-size 16 --ckpt-dir ${CKPT_DIR}`), record numbers, exit. No code changes.
   - On iter ≥2: read the last `improvements.tsv` row. Pick one metric that needs work (low `diversity`, weird `occupancy_mean`, NaN loss, etc.).
2. **Diagnose.** Trace root cause via code reading, not guessing. Cite file:line in your commit message.
3. **Fix.** Smallest change that addresses the root cause. Use the `ds-improve` skill via the Skill tool — that's the in-repo discipline.
4. **Measure.** Same config as the baseline (so deltas are comparable). Compute the same three numbers (`occupancy_mean`, `diversity`, `fid_proxy`).
5. **Log.** Append one row to `${RESULTS_DIR}/improvements.tsv` with header (`timestamp\tcommit\ttarget_metric\tbefore\tafter\tdiagnosis\tfix`).
6. **Commit + push.** `git add -A && git commit -m "..."`. The background sync handles GCS — but you can also `gsutil rsync ${RESULTS_DIR} ${GCS_ROOT}/results/${RUN_ID}` explicitly.
7. **Exit cleanly.** The loop will sleep `COOLDOWN` then check the toggle. If the user flipped it off, you simply won't be relaunched.

## Watch-list (from project history)

- **Mode collapse**: `diversity` near 0 → discriminator too strong. Try `LightDiscriminator`, higher R1 gamma, or TTUR ratio bump.
- **Bad occupancy**: training shapes ~0.1–0.3 mean; model output 0.0 or 0.9 → occupancy loss weight wrong, or sculptor bias.
- **NaN/Inf**: bad lr, missing grad clipping, AMP overflow. Check `mixed_precision`.
- **Diffusion noisy at T=0**: noise schedule too aggressive or too few sampling steps.

## What good looks like

- Each `improvements.tsv` row shows a measurable delta on a named metric, with a one-line diagnosis citing file:line.
- No multi-file refactors. No new dependencies. No `pytest` runs (the suite has pre-existing import bugs; out of scope for this loop unless that IS the root cause).
- The commit message is precise: `improve(gan): clamp R1 gamma to 0.5 — diversity 0.08→0.21`.

## When to stop early

- If you find no actionable improvement after 5 minutes of reading: log `notes=no-actionable-finding` to the TSV and exit. The cooldown + next iter will reset your perspective.
- If a change made the metric worse: revert with `git reset --hard HEAD~1`, log `notes=reverted`, exit.

Go.
