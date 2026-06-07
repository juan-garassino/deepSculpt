# Workload Identity Federation binding.
#
# The pool itself (`gh-actions`) and the OIDC provider (`github`) live in
# `garassino-op` and are managed there (created during 2026-06-07 cutover).
# Here we only grant our project's runtime SA to be impersonated by the
# GitHub Actions workflow identity of `juan-garassino/deep_sculpt`.

# Allow the GHA workflow identity for this specific repo to impersonate the SA.
resource "google_service_account_iam_member" "wif_impersonation" {
  service_account_id = google_service_account.runpod_runtime.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/projects/${var.op_project_number}/locations/global/workloadIdentityPools/${var.wif_pool_id}/attribute.repository/${var.github_repo}"
}

# Also grant the GHA token the right to mint access tokens for this SA
# (this is what `gcloud auth print-access-token` needs after impersonation).
resource "google_service_account_iam_member" "wif_token_creator" {
  service_account_id = google_service_account.runpod_runtime.name
  role               = "roles/iam.serviceAccountTokenCreator"
  member             = "principalSet://iam.googleapis.com/projects/${var.op_project_number}/locations/global/workloadIdentityPools/${var.wif_pool_id}/attribute.repository/${var.github_repo}"
}
