# DeepSculpt GCP — show-and-destroy infrastructure
#
# Per project policy:
#   - No always-on resources. `make show` brings everything up; `make destroy` tears it down.
#   - State is remote at gs://garassino-op-tf-state/deepsculpt/.
#   - Bucket gs://garassino-ml-artifacts is shared across ML projects and managed out-of-band;
#     this stack only manages the SA, WIF binding, and IAM membership on the bucket prefix.

CLOUDSDK_PYTHON ?= /usr/local/bin/python3.12

.PHONY: help init plan show destroy fmt lint apply destroy-confirm output

help:
	@echo "make init      — terraform init (configures GCS backend)"
	@echo "make fmt       — terraform fmt"
	@echo "make plan      — terraform plan"
	@echo "make show      — terraform apply (SHOW: bring resources up)"
	@echo "make destroy   — terraform destroy (tear resources down)"
	@echo "make output    — show GHA secret values to copy into gh secret set"

init:
	CLOUDSDK_PYTHON=$(CLOUDSDK_PYTHON) terraform init

fmt:
	terraform fmt -recursive

lint:
	terraform fmt -check -recursive
	terraform validate

plan:
	CLOUDSDK_PYTHON=$(CLOUDSDK_PYTHON) terraform plan

show: apply

apply:
	CLOUDSDK_PYTHON=$(CLOUDSDK_PYTHON) terraform apply

destroy:
	@echo "This will tear down the SA + IAM bindings."
	@echo "Bucket itself is shared and will NOT be deleted."
	@read -p "Type 'destroy' to confirm: " ans && [ "$$ans" = "destroy" ]
	CLOUDSDK_PYTHON=$(CLOUDSDK_PYTHON) terraform destroy

output:
	@CLOUDSDK_PYTHON=$(CLOUDSDK_PYTHON) terraform output -raw gh_secret_setup
