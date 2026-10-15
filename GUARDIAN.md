# GUARDIAN

Automated security & bug-fix ledger for `deep-sculpt`. Maintained on branch
`claude/guardian`; every run appends to the Log and updates the ledger.

## Checks

| Check | Tool / rule | Scope |
|---|---|---|
| Dependency vulns | `pip-audit -r` | `services/{inference,trainer,mlflow}/requirements.txt`, `requirements.txt` |
| Risky code | `bandit -r` + grep (eval/exec, `torch.load` w/o `weights_only`, `pickle`, `yaml.load`, `shell=True`, `verify=False`, CORS, `0.0.0.0`) | `deepsculpt/`, `services/`, `scripts/` |
| Secrets | regex scan over `git log -p --all` (AWS/GCP/GitHub/Anthropic/OpenAI/Slack/Telegram/private keys/DSNs) + tracked `.env`/key files | full history (gitleaks not installable in sandbox) |
| GitHub Actions | pinning, `permissions:`, `${{ inputs/github.event }}` in `run:` | `.github/workflows/` |
| Dockerfiles | base pinning, `USER` | `services/*/Dockerfile`, `runpod/Dockerfile` |
| Tests | `pytest tests/test_shodhan.py tests/test_palette.py tests/test_latent_ops.py tests/test_factory.py tests/test_inference_security.py` | CPU torch |
| Lint | `ruff --select E9,F63,F7,F82` (syntax/undefined names) | touched files |

## Findings ledger

| id | severity | type | file | status | date |
|---|---|---|---|---|---|
| G-001 | medium | confused deputy / arbitrary GCS read | `services/inference/app/main.py` (`/visualize-dataset`) | fixed | 2026-10-15 |
| G-002 | low | bug: gcloud fallback splits JSON on commas | `services/inference/app/gcp_run_job.py` | fixed | 2026-10-15 |
| G-003 | medium | vuln deps: pillow 10.2.0, requests 2.31.0 | `services/inference/requirements.txt`, `services/trainer/requirements.txt` | fixed | 2026-10-15 |
| G-004 | high | vuln deps: mlflow 2.13.1 (many advisories), pyarrow 15.0.2, protobuf 4.25.9 | `services/*/requirements.txt` | open (needs major bump + image test) | 2026-10-15 |
| G-005 | medium | vuln deps: starlette 0.36.3 via fastapi 0.110.0; prefect 2.16.5, anyio 3.7.1 | `services/inference`, `services/trainer` | open | 2026-10-15 |
| G-006 | medium | unauthenticated control endpoints (`/train`, `/reload-model`) rely solely on Cloud Run IAM | `services/inference/app/main.py` | open (verify `--no-allow-unauthenticated`) | 2026-10-15 |
| G-007 | medium | `torch.load(weights_only=False)` on checkpoints (RCE if a checkpoint is attacker-controlled) | `deepsculpt/core/**` (~20 sites) | wontfix for now (operator-owned checkpoints; see Backlog) | 2026-10-15 |
| G-008 | low | `${{ inputs.* }}` interpolated directly in `run:` (dispatch-only, writers only) | `.github/workflows/deploy-runpod.yml`, `deploy-vertex.yml` | open (proposed patch below) | 2026-10-15 |
| G-009 | low | actions pinned by tag, not SHA | `.github/workflows/*.yml` | open | 2026-10-15 |
| G-010 | low | services Dockerfiles run as root | `services/{inference,trainer,mlflow}/Dockerfile` | open | 2026-10-15 |
| G-011 | low | bug: `/train/status/{execution_id}` can never match real execution names (they contain `/`) | `services/inference/app/main.py` | open | 2026-10-15 |
| G-012 | low | bug: test hardcoded a developer-machine `cwd` | `tests/test_shodhan.py` | fixed | 2026-10-15 |
| G-013 | low | raw exception text returned in HTTP 4xx/5xx details | `services/inference/app/main.py` | open | 2026-10-15 |

Secrets scan: no credential patterns found in history (PNG base64 false positives only).

## Metrics

| metric | before (run 1) | after (run 1) |
|---|---|---|
| open vulns crit/high/med/low | 0 / 1 / 5 / 6 | 0 / 1 / 3 / 5 |
| pip-audit advisories (inference reqs) | 136 | 101 |
| bandit (High / Medium / Low) | 0 / 33 / 81 | 0 / 33 / 81 |
| trusted tests passing | 33 / 34 (1 failing) | 42 / 42 |
| lint errors (ruff E9/F63/F7/F82, touched files) | 0 | 0 |
| coverage | not measured | not measured |

## Backlog

1. **G-004/G-005**: bump `mlflow` (2.13 -> >=2.22/3.x), `fastapi` to a release whose starlette is >=1.3, `prefect`; rebuild images and smoke `/health`, `/infer`.
2. **G-006**: confirm both Cloud Run services are deployed with `--no-allow-unauthenticated`; optionally add a shared-secret header check on `/train` and `/reload-model`.
3. **G-007**: switch `torch.load` to `weights_only=True` where only tensors/state_dicts are loaded (`core/latent/directions.py`, `codec.py` VAE state_dict, data loaders) and add `torch.serialization.add_safe_globals` for config objects in checkpoints.
4. **G-008 proposed patch** (not pushed: token lacks `workflow` scope) - in `deploy-runpod.yml`/`deploy-vertex.yml` move inputs to `env:` and reference shell vars:
   ```yaml
   env:
     INPUT_RUN_ID: ${{ inputs.run_id }}
   run: |
     if [ -z "$INPUT_RUN_ID" ]; then ... else echo "value=$INPUT_RUN_ID" >> "$GITHUB_OUTPUT"; fi
   ```
   Same for `case "${{ inputs.machine }}"` -> `case "$INPUT_MACHINE"`.
5. **G-009**: pin `actions/checkout`, `docker/*`, `google-github-actions/*` to full commit SHAs (Dependabot can keep them fresh).
6. **G-010**: add a non-root `USER` to services Dockerfiles; pin base images by digest.
7. **G-011**: use `{execution_id:path}` and validate it matches `projects/.../executions/...`.
8. Legacy suite: 11 collection errors in `tests/unit`, `tests/integration`, `tests/test_trainer.py`.

## Log

### Run 1 — 2026-10-15
- Created branch + ledger. Installed local pre-push guard (only `claude/guardian`).
- Scanned deps (pip-audit), code (bandit + grep), secrets (history regex), workflows, Dockerfiles.
- Fixed G-001 (bucket pinning + `allow_pickle=False` in `/visualize-dataset`), G-002 (gcloud
  `^~|~^` env delimiter), G-003 (pillow 12.3.0, requests 2.33.0), G-012 (portable test cwd).
- Added `tests/test_inference_security.py` (8 tests, GCP stubbed). Tests 33/34 -> 42/42.

## Learnings

- `download.pytorch.org` is blocked by the sandbox proxy; PyPI `torch` installs fine and the trusted tests also need `tqdm scikit-learn plotly psutil` beyond torch/numpy.
- `services/` is a namespace package (no `__init__.py`): helpers with no GCP imports are directly testable from `tests/`.
- pip-audit reports no severity; triage by package exposure (internet-facing inference image first).

## Ideas

1. Add a CPU CI job (GCP-native Cloud Build or GHA) running the trusted test subset + `pip-audit` on `services/*/requirements.txt`.
2. Split heavy optional imports (`plotly`, `sklearn`) out of `deepsculpt.core` import paths so the core tests need only torch+numpy.
3. Add a Dependabot config for `pip` (services requirements) and `github-actions`.
