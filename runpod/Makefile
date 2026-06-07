# DeepSculpt — RunPod deployment
#
# Usage:
#   make build                              Build the Docker image
#   make login                              Login to GHCR (uses GHCR_USER + GHCR_TOKEN)
#   make push                               Push image to ghcr.io
#   make run-train  API_KEY=...             Run training mode locally (needs GPU)
#   make run-research API_KEY=...           Run Claude research mode (one-shot)
#   make run-improve  API_KEY=...           Run Claude self-improve mode (one-shot)
#   make run-self-improve API_KEY=...       Continuous self-improve loop (toggleable)
#   make toggle-on                          Enable the self-improve loop (GCS-synced)
#   make toggle-off                         Disable the self-improve loop
#   make toggle-status                      Print current toggle state
#   make logs                               Tail container logs
#   make stop                               Stop + remove the running container
#   make sync-up   KIND=checkpoints RUN_ID=...    Push local /workspace/<kind> → GCS
#   make sync-down KIND=data                       Pull GCS → local /workspace/<kind>
#   make results  RUN_ID=...                Print experiments.tsv from GCS
#
# Required env (.env or shell):
#   ANTHROPIC_API_KEY
#   GCS_BUCKET                              default: garassino-ml-artifacts
#   GHCR_USER, GHCR_TOKEN                   for manual `make push`
#                                           (CI uses GITHUB_TOKEN automatically)
#
# Auth model: pods receive a short-lived GCS_ACCESS_TOKEN minted by GHA via
# WIF. No service-account JSON keys anywhere. See ../docs/runpod.md.

IMAGE_NAME   ?= deepsculpt-runpod
REGISTRY     ?= ghcr.io/juan-garassino
IMAGE        := $(REGISTRY)/$(IMAGE_NAME)
CONTAINER    ?= deepsculpt
GCS_BUCKET   ?= garassino-ml-artifacts
API_KEY      ?=
RUN_ID       ?= $(shell date +%Y%m%d-%H%M%S)
TIME_BUDGET  ?= 3600
ITER_BUDGET  ?= 1800
COOLDOWN     ?= 300

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

.PHONY: build
build:
	cd .. && docker build -f runpod/Dockerfile -t $(IMAGE_NAME) .

# ---------------------------------------------------------------------------
# Registry (GitHub Container Registry)
# ---------------------------------------------------------------------------

.PHONY: login
login:
ifndef GHCR_TOKEN
	$(error GHCR_TOKEN is required. export GHCR_TOKEN=ghp_...)
endif
ifndef GHCR_USER
	$(error GHCR_USER is required. export GHCR_USER=your-github-username)
endif
	echo "$(GHCR_TOKEN)" | docker login ghcr.io -u "$(GHCR_USER)" --password-stdin

.PHONY: push
push: build
	docker tag $(IMAGE_NAME) $(IMAGE):latest
	docker push $(IMAGE):latest
	@echo ""
	@echo "=== Pushed $(IMAGE):latest ==="
	@echo ""
	@echo "RunPod setup:"
	@echo "  1. Create GPU Pod (H100 or A100 recommended)"
	@echo "  2. Container image: $(IMAGE):latest"
	@echo "  3. Env vars:"
	@echo "       ANTHROPIC_API_KEY=sk-ant-..."
	@echo "       GCS_BUCKET=$(GCS_BUCKET)"
	@echo "       GCS_ACCESS_TOKEN=<minted by gcloud auth print-access-token>"
	@echo "       MODE=train|research|improve|self-improve"
	@echo "       RUN_ID=<your-run-id>"
	@echo "       TIME_BUDGET=$(TIME_BUDGET)"
	@echo ""
	@echo "Or skip all this and use the GHA workflow: gh workflow run deploy-runpod.yml"

# ---------------------------------------------------------------------------
# Run locally (needs NVIDIA GPU + docker --gpus). Local runs assume you've
# minted a token yourself via `gcloud auth print-access-token` and exported
# it as GCS_ACCESS_TOKEN. CI does this automatically.
# ---------------------------------------------------------------------------

define DOCKER_RUN_BASE
docker run -d \
	--gpus all \
	--name $(CONTAINER) \
	-e ANTHROPIC_API_KEY=$(API_KEY) \
	-e GCS_BUCKET=$(GCS_BUCKET) \
	-e GCS_PROJECT=garassino-ml \
	-e GCS_ACCESS_TOKEN=$$GCS_ACCESS_TOKEN \
	-e RUN_ID=$(RUN_ID) \
	-e TIME_BUDGET=$(TIME_BUDGET) \
	-v $(PWD)/workspace:/workspace
endef

.PHONY: run-train
run-train:
	$(DOCKER_RUN_BASE) -e MODE=train $(IMAGE_NAME)
	@echo "=== Started $(CONTAINER) (MODE=train, RUN_ID=$(RUN_ID)) ==="

.PHONY: run-research
run-research:
ifndef API_KEY
	$(error API_KEY is required for research mode. make run-research API_KEY=sk-ant-...)
endif
	$(DOCKER_RUN_BASE) -e MODE=research $(IMAGE_NAME)
	@echo "=== Started $(CONTAINER) (MODE=research, RUN_ID=$(RUN_ID)) ==="

.PHONY: run-improve
run-improve:
ifndef API_KEY
	$(error API_KEY is required for improve mode. make run-improve API_KEY=sk-ant-...)
endif
	$(DOCKER_RUN_BASE) -e MODE=improve $(IMAGE_NAME)
	@echo "=== Started $(CONTAINER) (MODE=improve, RUN_ID=$(RUN_ID)) ==="

.PHONY: run-self-improve
run-self-improve:
ifndef API_KEY
	$(error API_KEY is required for self-improve mode. make run-self-improve API_KEY=sk-ant-...)
endif
	$(DOCKER_RUN_BASE) \
		-e MODE=self-improve \
		-e ITER_BUDGET=$(ITER_BUDGET) \
		-e COOLDOWN=$(COOLDOWN) \
		$(IMAGE_NAME)
	@echo "=== Started $(CONTAINER) (MODE=self-improve, RUN_ID=$(RUN_ID)) ==="
	@echo "=== Loop is OFF by default. Enable with: make toggle-on ==="

# ---------------------------------------------------------------------------
# Remote deploy via GHA (the canonical path — no local Docker required)
# ---------------------------------------------------------------------------

.PHONY: deploy
deploy:
	@if [ -z "$(MODE)" ]; then echo "Usage: make deploy MODE=train|research|improve|self-improve [RUN_ID=...] [TIME_BUDGET=...]"; exit 1; fi
	gh workflow run deploy-runpod.yml \
		-f mode=$(MODE) \
		-f run_id=$(RUN_ID) \
		-f time_budget=$(TIME_BUDGET) \
		-f iter_budget=$(ITER_BUDGET) \
		-f cooldown=$(COOLDOWN)
	@echo "=== Dispatched deploy-runpod.yml. Watch: gh run watch ==="

# ---------------------------------------------------------------------------
# Self-improve toggle (GCS-synced; works whether pod is local or on RunPod)
# ---------------------------------------------------------------------------

.PHONY: toggle-on toggle-off toggle-status
toggle-on:
	GCS_BUCKET=$(GCS_BUCKET) bash scripts/self_improve_toggle.sh on

toggle-off:
	GCS_BUCKET=$(GCS_BUCKET) bash scripts/self_improve_toggle.sh off

toggle-status:
	@GCS_BUCKET=$(GCS_BUCKET) bash scripts/self_improve_toggle.sh status

# ---------------------------------------------------------------------------
# Monitor / stop
# ---------------------------------------------------------------------------

.PHONY: logs
logs:
	docker logs -f $(CONTAINER)

.PHONY: stop
stop:
	-docker stop $(CONTAINER)
	-docker rm $(CONTAINER)
	@echo "=== Container stopped and removed ==="

# ---------------------------------------------------------------------------
# GCS sync (local dev convenience)
# ---------------------------------------------------------------------------

KIND ?= checkpoints

.PHONY: sync-up
sync-up:
	WORKSPACE_DIR=$(PWD)/workspace GCS_BUCKET=$(GCS_BUCKET) RUN_ID=$(RUN_ID) \
		python scripts/gcs_sync.py push $(KIND)

.PHONY: sync-down
sync-down:
	WORKSPACE_DIR=$(PWD)/workspace GCS_BUCKET=$(GCS_BUCKET) RUN_ID=$(RUN_ID) \
		python scripts/gcs_sync.py pull $(KIND)

.PHONY: results
results:
	gsutil cat gs://$(GCS_BUCKET)/deepsculpt/results/$(RUN_ID)/experiments.tsv \
		2>/dev/null || echo "No experiments.tsv yet for RUN_ID=$(RUN_ID)"
