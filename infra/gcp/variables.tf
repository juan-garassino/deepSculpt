variable "project_id" {
  description = "GCP project that owns the runtime SA + bucket."
  type        = string
  default     = "garassino-ml"
}

variable "project_number" {
  description = "Numeric ID of project_id (used in some IAM resource names)."
  type        = string
  default     = "920423386248"
}

variable "op_project_id" {
  description = "Control-plane project that hosts the WIF pool."
  type        = string
  default     = "garassino-op"
}

variable "op_project_number" {
  description = "Numeric ID of op_project_id."
  type        = string
  default     = "634336216563"
}

variable "region" {
  description = "Default region for new resources."
  type        = string
  default     = "europe-west1"
}

variable "bucket_name" {
  description = "Shared ML artifacts bucket. Already created out-of-band."
  type        = string
  default     = "garassino-ml-artifacts"
}

variable "bucket_prefix" {
  description = "Prefix this project owns within the shared bucket."
  type        = string
  default     = "deepsculpt"
}

variable "github_repo" {
  description = "owner/repo for the WIF binding."
  type        = string
  default     = "juan-garassino/deep_sculpt"
}

variable "wif_pool_id" {
  description = "WIF pool name in op_project_id."
  type        = string
  default     = "gh-actions"
}

variable "wif_provider_id" {
  description = "OIDC provider name inside the pool."
  type        = string
  default     = "github"
}

variable "runtime_sa_id" {
  description = "SA account ID (left of @)."
  type        = string
  default     = "deepsculpt-runpod-runtime"
}

variable "budget_alert_email" {
  description = "Where budget alerts go."
  type        = string
  default     = "juan.garassino@gmail.com"
}

variable "monthly_budget_eur" {
  description = "Hard cap reminder; alerts at 40/80/100% of this."
  type        = number
  default     = 25
}
