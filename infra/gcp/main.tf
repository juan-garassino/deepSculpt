terraform {
  required_version = ">= 1.6"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }

  backend "gcs" {
    bucket = "garassino-op-tf-state"
    prefix = "deepsculpt"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Runtime service account that GHA workflows impersonate via WIF.
# Pod never sees this SA's JSON; pod only sees the short-lived bearer token
# the workflow mints by impersonating it.
resource "google_service_account" "runpod_runtime" {
  project      = var.project_id
  account_id   = var.runtime_sa_id
  display_name = "DeepSculpt RunPod runtime (GHA-impersonated)"
  description  = "Impersonated by github.com/${var.github_repo} via WIF in ${var.op_project_id}. No JSON key issued."
}

# Bucket assumed pre-existing (created out-of-band; europe-west1, STANDARD).
# This block only manages the lifecycle policy and the IAM binding.
data "google_storage_bucket" "artifacts" {
  name = var.bucket_name
}

# Object-level admin scoped to this project's prefix.
# Using condition expression: only objects whose name starts with deepsculpt/.
resource "google_storage_bucket_iam_member" "runtime_object_admin" {
  bucket = data.google_storage_bucket.artifacts.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.runpod_runtime.email}"

  condition {
    title       = "deepsculpt prefix only"
    description = "Restrict to gs://${var.bucket_name}/${var.bucket_prefix}/* objects."
    expression  = "resource.name.startsWith(\"projects/_/buckets/${var.bucket_name}/objects/${var.bucket_prefix}/\")"
  }
}

# Bucket-level read on the bucket itself (so listObjects works under the prefix).
resource "google_storage_bucket_iam_member" "runtime_legacy_reader" {
  bucket = data.google_storage_bucket.artifacts.name
  role   = "roles/storage.legacyBucketReader"
  member = "serviceAccount:${google_service_account.runpod_runtime.email}"
}

# Lifecycle on prompts-archive/ — cheap to keep but no need to retain forever.
# Applied at bucket level (affects all prefixes that match), so we use an
# age + matches_prefix filter to scope only to deepsculpt/prompts-archive/.
resource "google_storage_bucket" "lifecycle_only" {
  count                       = 0 # placeholder — bucket is not managed here, lifecycle set out-of-band
  name                        = var.bucket_name
  location                    = var.region
  uniform_bucket_level_access = true
}
