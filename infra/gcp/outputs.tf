output "runtime_sa_email" {
  description = "Set GHA repo secret GCP_RUNTIME_SA to this value."
  value       = google_service_account.runpod_runtime.email
}

output "wif_provider" {
  description = "Set GHA repo secret GCP_WIF_PROVIDER to this value."
  value       = "projects/${var.op_project_number}/locations/global/workloadIdentityPools/${var.wif_pool_id}/providers/${var.wif_provider_id}"
}

output "bucket_url" {
  value = "gs://${data.google_storage_bucket.artifacts.name}/${var.bucket_prefix}/"
}

output "gh_secret_setup" {
  description = "Copy-paste these gh commands after `terraform apply`."
  value       = <<EOT
gh secret set GCP_WIF_PROVIDER --body "projects/${var.op_project_number}/locations/global/workloadIdentityPools/${var.wif_pool_id}/providers/${var.wif_provider_id}"
gh secret set GCP_RUNTIME_SA   --body "${google_service_account.runpod_runtime.email}"
EOT
}
