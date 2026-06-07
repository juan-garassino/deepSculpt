# Operations

## RunPod (canonical training path)

| Action | Command |
|---|---|
| Deploy a pod (any mode) | `gh workflow run deploy-runpod.yml -f mode={train,research,improve,self-improve} -f run_id=... -f time_budget=... -f gpu_type='NVIDIA A100 80GB PCIe'` |
| Build + push image | `git push origin master` (auto via `.github/workflows/build-push.yml`) |
| Refresh GCS token in running pods | automatic (cron every 50 min via `refresh-token.yml`); manual: `gh workflow run refresh-token.yml` |
| Read live results | `gsutil cat gs://garassino-ml-artifacts/deepsculpt/results/<RUN_ID>/experiments.tsv` |
| Toggle self-improve loop | `make -C runpod toggle-{on,off,status}` (writes to `gs://.../control/self_improve.enabled`) |
| Show / destroy infra | `make -C infra/gcp {show,destroy}` |
| Telegram notify (optional) | cron `notify-telegram.yml`; no-ops until `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` secrets exist |

Full deploy guide: [`docs/runpod.md`](./runpod.md). Infra reference: [`infra/gcp/README.md`](../infra/gcp/README.md).

## Docker Compose (local dev only)

1. Build images.
```bash
make docker-build
```

2. Start services.
```bash
make docker-up
```

3. Tail logs.
```bash
make docker-logs
```

4. Stop services.
```bash
make docker-down
```

## MLflow

The MLflow server is stateless and expects:
1. `BACKEND_STORE_URI` pointing to Cloud SQL Postgres in production.
2. `ARTIFACT_ROOT` pointing to GCS.

For local dev, `docker-compose.yml` uses `sqlite:///mlflow.db` and `./mlruns`.

## Cloud Run Job Execution Status

Use:
```bash
curl http://localhost:8081/train/status/projects/<PROJECT_ID>/locations/<REGION>/jobs/<JOB_NAME>/executions/<EXECUTION_ID>
```
