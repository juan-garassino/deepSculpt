# Vertex AI custom-training path.
#
# Vertex requires the training container to live in Artifact Registry, so we
# keep a single docker repo here and mirror the GHCR image into it from
# build-push.yml. The training job runs AS the runtime SA — ADC via the
# metadata server, so no bearer-token plumbing or refresh cron is needed
# (unlike the RunPod path).

resource "google_project_service" "aiplatform" {
  project            = var.project_id
  service            = "aiplatform.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "artifactregistry" {
  project            = var.project_id
  service            = "artifactregistry.googleapis.com"
  disable_on_destroy = false
}

resource "google_artifact_registry_repository" "ml_images" {
  project       = var.project_id
  location      = var.region
  repository_id = "ml-images"
  description   = "Training images for garassino-ml Vertex jobs (mirrored from GHCR)"
  format        = "DOCKER"

  depends_on = [google_project_service.artifactregistry]
}

# build-push.yml (as the runtime SA via WIF) pushes the mirrored image.
resource "google_artifact_registry_repository_iam_member" "runtime_ar_writer" {
  project    = var.project_id
  location   = google_artifact_registry_repository.ml_images.location
  repository = google_artifact_registry_repository.ml_images.name
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.runpod_runtime.email}"
}

# Vertex AI custom-code service agent pulls the image at job start.
resource "google_artifact_registry_repository_iam_member" "vertex_agent_ar_reader" {
  project    = var.project_id
  location   = google_artifact_registry_repository.ml_images.location
  repository = google_artifact_registry_repository.ml_images.name
  role       = "roles/artifactregistry.reader"
  member     = "serviceAccount:service-${var.project_number}@gcp-sa-aiplatform-cc.iam.gserviceaccount.com"

  depends_on = [google_project_service.aiplatform]
}

# deploy-vertex.yml (as the runtime SA via WIF) submits custom jobs.
resource "google_project_iam_member" "runtime_aiplatform_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.runpod_runtime.email}"
}

# The submitter must be allowed to actAs the SA the job runs as — which is
# itself here (submitting SA == job SA).
resource "google_service_account_iam_member" "runtime_self_act_as" {
  service_account_id = google_service_account.runpod_runtime.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.runpod_runtime.email}"
}
