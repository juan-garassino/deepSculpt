# `infra/gcp/` — DeepSculpt show-and-destroy infrastructure

Terraform stack that owns the GCP-side runtime identity for DeepSculpt's RunPod deploy.

## What lives here

| Resource | Purpose |
|---|---|
| `google_service_account.runpod_runtime` | `deepsculpt-runpod-runtime@garassino-ml.iam.gserviceaccount.com`. Impersonated by GHA via WIF. **No JSON key issued.** |
| `google_storage_bucket_iam_member.runtime_object_admin` | `roles/storage.objectAdmin` scoped to `gs://garassino-ml-artifacts/deepsculpt/*` via IAM condition. |
| `google_storage_bucket_iam_member.runtime_legacy_reader` | `roles/storage.legacyBucketReader` so `listObjects` under the prefix works. |
| `google_service_account_iam_member.wif_impersonation` | Binds the GHA workflow identity of `juan-garassino/deep_sculpt` to impersonate the SA. |
| `google_service_account_iam_member.wif_token_creator` | Grants `roles/iam.serviceAccountTokenCreator` so `gcloud auth print-access-token` works after impersonation. |

## What does NOT live here

- **The bucket itself.** `gs://garassino-ml-artifacts` is shared across ML projects and was created out-of-band (`gsutil mb`). This stack only manages IAM bindings on it.
- **The WIF pool + OIDC provider.** They live in `garassino-op` and were created during the 2026-06-07 GCP cutover.
- **Budget alerts.** Created via the GCP Console because TF for budgets needs Billing-Account-level admin — easier to manage in the UI. Threshold: €25/mo on `garassino-ml`, alerts at 40/80/100% to `juan.garassino@gmail.com`.

## Prereqs

- `terraform` ≥ 1.6
- `gcloud` authenticated as a user with `roles/owner` (or fine-grained `iam.serviceAccountAdmin` + `storage.admin` + the cross-project `iam.workloadIdentityPoolAdmin` on `garassino-op`)
- `CLOUDSDK_PYTHON=/usr/local/bin/python3.12` exported (Python 3.7 is end-of-life for gcloud)
- The shared bucket `gs://garassino-ml-artifacts` must exist (in `europe-west1`)
- The WIF pool `gh-actions` and OIDC provider `github` must exist in `garassino-op`

## Show / destroy

```bash
cd infra/gcp
make init        # one-time per checkout (configures GCS backend)
make plan        # see what would change
make show        # apply — brings resources up
make output      # prints `gh secret set` commands for the WIF outputs
make destroy     # tear down (asks for confirmation)
```

## After `make show`

Run the printed `gh secret set` commands to push the WIF outputs into the repo's GitHub Actions secrets. The two values are:

- `GCP_WIF_PROVIDER` — `projects/634336216563/locations/global/workloadIdentityPools/gh-actions/providers/github`
- `GCP_RUNTIME_SA` — `deepsculpt-runpod-runtime@garassino-ml.iam.gserviceaccount.com`

These power `.github/workflows/deploy-runpod.yml` and `refresh-token.yml`.

## Cost

Steady-state cost of this stack is **€0**. IAM bindings, SA accounts, and remote TF state are free. Storage cost is paid against `garassino-ml-artifacts` regardless of whether this stack exists.

## Show-and-destroy discipline

Per project policy, DeepSculpt is RunPod-driven — no always-on GCP compute. The only thing this stack creates that *could* cost money is the GCS access. To be doubly safe, `make destroy` removes the runtime SA, which invalidates any in-flight tokens and stops any forgotten cron from accruing storage charges.
