# ----------------------------------
#      TRAINING DEFAULTS
# ----------------------------------

DATA_OUT ?= ./colab_data
RUN_OUT ?= ./colab_runs
DATA_MODE ?= reuse
STRUCTURE_PRESET ?= architectural
GRID_COUNT ?= 1
GRID_STEP ?= 4
NUM_SAMPLES ?= 5000
EPOCHS ?= 200
BATCH_SIZE ?= 4
VOID_DIM ?= 32
NOISE_DIM ?= 100
TIMESTEPS ?= 100
LEARNING_RATE ?= 1e-4
WEIGHT_DECAY ?= 0.01
NOISE_SCHEDULE ?= cosine
BETA_START ?= 0.0001
BETA_END ?= 0.02
NUM_INFERENCE_SAMPLES ?= 1
NUM_INFERENCE_STEPS ?= 50
DIFFUSION_SAMPLER ?= ddim
GUIDANCE_SCALE ?= 1.0
NUM_WORKERS ?= 0
DISCRIMINATOR_TYPE ?= light
GEN_CHANNELS ?= 256
R1_GAMMA ?= 10.0
R1_INTERVAL ?= 16
GAN_AUGMENT ?= none
GAN_AUGMENT_P ?= 0.0
GAN_AUGMENT_TARGET ?= 0.7
OCCUPANCY_LOSS_WEIGHT ?= 20.0
OCCUPANCY_FLOOR ?= 0.02
OCCUPANCY_TARGET_MODE ?= batch_real
EMA_DECAY ?= 0.999
TTUR_RATIO ?= 0.1

# ----------------------------------
#      GPU PRESETS
# ----------------------------------
# Usage: make colab-train-mono GPU=t4
#        make colab-train-mono GPU=a100
#        make colab-train-mono GPU=h100

ifeq ($(GPU),t4)
  # Tesla T4 (16GB) — Colab free/pro
  VOID_DIM := 32
  BATCH_SIZE := 4
  NUM_SAMPLES := 5000
  EPOCHS := 200
  NUM_WORKERS := 2
endif

ifeq ($(GPU),a100)
  # A100 (40-80GB) — Colab pro+, cloud
  VOID_DIM := 64
  BATCH_SIZE := 8
  NUM_SAMPLES := 10000
  EPOCHS := 300
  NUM_WORKERS := 4
endif

ifeq ($(GPU),h100)
  # H100 (80GB) — maximum quality
  VOID_DIM := 128
  BATCH_SIZE := 4
  NUM_SAMPLES := 10000
  EPOCHS := 300
  NUM_WORKERS := 4
endif

.PHONY: colab-train-mono colab-train-color colab-train-diffusion colab-train-diffusion-color

colab-train-mono:
	python scripts/colab_train.py \
		--data-output "$(DATA_OUT)" \
		--run-output "$(RUN_OUT)" \
		--data-mode "$(DATA_MODE)" \
		--structure-preset "$(STRUCTURE_PRESET)" \
		--grid-count "$(GRID_COUNT)" \
		--grid-step "$(GRID_STEP)" \
		--num-samples "$(NUM_SAMPLES)" \
		--epochs "$(EPOCHS)" \
		--batch-size "$(BATCH_SIZE)" \
		--void-dim "$(VOID_DIM)" \
		--noise-dim "$(NOISE_DIM)" \
		--num-inference-samples "$(NUM_INFERENCE_SAMPLES)" \
		--num-workers "$(NUM_WORKERS)" \
		--discriminator-type "$(DISCRIMINATOR_TYPE)" \
		--r1-gamma "$(R1_GAMMA)" \
		--r1-interval "$(R1_INTERVAL)" \
		--augment "$(GAN_AUGMENT)" \
		--augment-p "$(GAN_AUGMENT_P)" \
		--augment-target "$(GAN_AUGMENT_TARGET)" \
		--occupancy-loss-weight "$(OCCUPANCY_LOSS_WEIGHT)" \
		--occupancy-floor "$(OCCUPANCY_FLOOR)" \
		--occupancy-target-mode "$(OCCUPANCY_TARGET_MODE)" \
		--ema-decay "$(EMA_DECAY)" \
		--ttur-ratio "$(TTUR_RATIO)" \
		--gen-channels "$(GEN_CHANNELS)"

colab-train-color:
	python scripts/colab_train.py \
		--data-output "$(DATA_OUT)" \
		--run-output "$(RUN_OUT)" \
		--data-mode "$(DATA_MODE)" \
		--structure-preset "$(STRUCTURE_PRESET)" \
		--grid-count "$(GRID_COUNT)" \
		--grid-step "$(GRID_STEP)" \
		--num-samples "$(NUM_SAMPLES)" \
		--epochs "$(EPOCHS)" \
		--batch-size "$(BATCH_SIZE)" \
		--void-dim "$(VOID_DIM)" \
		--noise-dim "$(NOISE_DIM)" \
		--num-inference-samples "$(NUM_INFERENCE_SAMPLES)" \
		--num-workers "$(NUM_WORKERS)" \
		--discriminator-type "$(DISCRIMINATOR_TYPE)" \
		--r1-gamma "$(R1_GAMMA)" \
		--r1-interval "$(R1_INTERVAL)" \
		--augment "$(GAN_AUGMENT)" \
		--augment-p "$(GAN_AUGMENT_P)" \
		--augment-target "$(GAN_AUGMENT_TARGET)" \
		--occupancy-loss-weight "$(OCCUPANCY_LOSS_WEIGHT)" \
		--occupancy-floor "$(OCCUPANCY_FLOOR)" \
		--occupancy-target-mode "$(OCCUPANCY_TARGET_MODE)" \
		--ema-decay "$(EMA_DECAY)" \
		--ttur-ratio "$(TTUR_RATIO)" \
		--gen-channels "$(GEN_CHANNELS)" \
		--color

colab-train-diffusion:
	python scripts/colab_train_diffusion.py \
		--data-output "$(DATA_OUT)" \
		--run-output "$(RUN_OUT)" \
		--data-mode "$(DATA_MODE)" \
		--num-samples "$(NUM_SAMPLES)" \
		--epochs "$(EPOCHS)" \
		--batch-size "$(BATCH_SIZE)" \
		--void-dim "$(VOID_DIM)" \
		--timesteps "$(TIMESTEPS)" \
		--learning-rate "$(LEARNING_RATE)" \
		--weight-decay "$(WEIGHT_DECAY)" \
		--noise-schedule "$(NOISE_SCHEDULE)" \
		--beta-start "$(BETA_START)" \
		--beta-end "$(BETA_END)" \
		--num-inference-samples "$(NUM_INFERENCE_SAMPLES)" \
		--num-inference-steps "$(NUM_INFERENCE_STEPS)" \
		--sampler "$(DIFFUSION_SAMPLER)" \
		--guidance-scale "$(GUIDANCE_SCALE)" \
		--num-workers "$(NUM_WORKERS)" \
		--ema-decay "$(EMA_DECAY)"

colab-train-diffusion-color:
	python scripts/colab_train_diffusion.py \
		--data-output "$(DATA_OUT)" \
		--run-output "$(RUN_OUT)" \
		--data-mode "$(DATA_MODE)" \
		--num-samples "$(NUM_SAMPLES)" \
		--epochs "$(EPOCHS)" \
		--batch-size "$(BATCH_SIZE)" \
		--void-dim "$(VOID_DIM)" \
		--timesteps "$(TIMESTEPS)" \
		--learning-rate "$(LEARNING_RATE)" \
		--weight-decay "$(WEIGHT_DECAY)" \
		--noise-schedule "$(NOISE_SCHEDULE)" \
		--beta-start "$(BETA_START)" \
		--beta-end "$(BETA_END)" \
		--num-inference-samples "$(NUM_INFERENCE_SAMPLES)" \
		--num-inference-steps "$(NUM_INFERENCE_STEPS)" \
		--sampler "$(DIFFUSION_SAMPLER)" \
		--guidance-scale "$(GUIDANCE_SCALE)" \
		--num-workers "$(NUM_WORKERS)" \
		--ema-decay "$(EMA_DECAY)" \
		--color

# ----------------------------------
#          INSTALL & TEST
# ----------------------------------

install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* deepSculpt/*.py

black:
	@black scripts/* deepSculpt/*/*.py deepSculpt/*.py deepSculpt/*/*/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*/__pycache__ */*/*/__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr deepSculpt-*.dist-info
	@rm -fr deepSculpt.egg-info
	@rm -fr results
	@rm -fr data
	@rm -fr notebooks/results
	@rm -fr notebooks/*.npy

install:
	@pip install . -U

all: clean install test black check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      MODEL
# ----------------------------------

app-id= 1639938746408345

app-secret= eb85d63732bdf12fd935d4a3437f5740

redirect-uri=

code=

access_token=IGQVJYQTNGcHc2akpNcVdtY1NjQmgxU2V1V014bVREdDBTMFFVR3hrWms3XzVBU25QajJTWElzSmVHWTFHU2c0bFZAmQllrWkFtYkoxcFpwVnBzcEU2VGU0STRSYl9tazdfVzRJUVFmdkh0ejdhczFubAZDZD

# PACKAGE RUNS
run_test:
	curl -X POST \
    https://api.instagram.com/oauth/${access_token} client_id=${app-id} client_secret=${app-secret} \
		grant_type=${authorization_code} redirect_uri=${redirect-uri} code=${code}

# PACKAGE RUNS
run_trainer:
	python -m deepSculpt.trainer.trainer

# PACKAGE RUNS
run_sculptor:
	python -m deepSculpt.sculptor.sculptor

# PACKAGE RUNS
run_collector:
	python -m deepSculpt.collector.collector

# PACKAGE RUNS
run_curator:
	python -m deepSculpt.curator.curator

# PACKAGE RUNS
run_emissary:
	python -m deepSculpt.emissary.emissary

# PACKAGE RUNS
run_manager:
	python -m deepSculpt.manager.manager

# PACKAGE RUNS
run_all: clean black run_collector run_trainer


# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------

PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)

# ----------------------------------
#      TRAIN MODEL
# ----------------------------------

# project id - replace with your GCP project id
PROJECT_ID=deepsculpt

# bucket
BUCKET_NAME=

# training folder
BUCKET_TRAINING_FOLDER=data

# training params, choose your region from https://cloud.google.com/storage/docs/locations#available_locations
REGION=europe-west1

# app environment
PYTHON_VERSION=3.7

FRAMEWORK=scikit-learn

RUNTIME_VERSION=2.2

# package params
PACKAGE_NAME=deepSculpt

FILENAME=trainer

JOB_DIR=gs://deepsculpt

MACHINE=config.yaml

MACHINE_GPU=config-gpu.yaml

set_project:
	@gcloud config set project ${PROJECT_ID}

create_bucket:
	@gsutil mb -l ${REGION} -p ${PROJECT_ID} gs://${os.environ.get('BUCKET_NAME')}

##### Job - - - - - - - - - - - - - - - - - - - - - - - - -

JOB_NAME=deepsculpt_$(shell date +'%Y%m%d_%H%M%S')

gcp_submit_training:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
		--job-dir gs://${os.environ.get('BUCKET_NAME')}/${BUCKET_TRAINING_FOLDER} \
		--package-path ${PACKAGE_NAME} \
		--module-name ${PACKAGE_NAME}.${FILENAME} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${REGION} \
		--config ${MACHINE} \
		--job-dir ${JOB_DIR} \
		--stream-logs

gcp_submit_training_gpu:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
		--job-dir gs://${os.environ.get('BUCKET_NAME')}/${BUCKET_TRAINING_FOLDER} \
		--package-path ${PACKAGE_NAME} \
		--module-name ${PACKAGE_NAME}.${FILENAME} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${REGION} \
		--config ${MACHINE_GPU} \
		--stream-logs

# ----------------------------------
#      RUN API
# ----------------------------------

api_run:
	python app.py


# PACKAGE ACTIONS
actions_reinstall:
	@pip uninstall -y taxifare || :
	@pip install -e .

# PACKAGE ACTIONS
actions_clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr deepSculpt-*.dist-info
	@rm -fr deepSculpt.egg-info

# PACKAGE RUNS
run_api:
	uvicorn deepCab.api.fast:app --reload

# TESTS
default:
	@echo 'tests are only executed locally for this challenge'
# TESTS
test_api:
	test_api_root test_api_predict
# TESTS
test_api_root:
	TEST_ENV=development pytest tests/api -k 'test_root' --asyncio-mode=strict -W "ignore"
# TESTS
test_api_predict:
	TEST_ENV=development pytest tests/api -k 'test_predict' --asyncio-mode=strict -W "ignore"

# GOOGLE SETUP
gcpsetup_00_login:
	@gcloud auth login --cred-file=${GOOGLE_APPLICATION_CREDENTIALS}

# GOOGLE SETUP
gcpsetup_01_setproject:
	@gcloud config set project ${GCP_PROJECT_ID}

# GOOGLE SETUP
gcpsetup_02_setcredentials:
	@gcloud projects get-iam-policy ${GCP_PROJECT_ID} \
--flatten="bindings[].members" \
--format='table(bindings.role)' \
--filter="bindings.members:${SERVICE_ACCOUNT_EMAIL}"
# GOOGLE SETUP
gcpsetup_03_fullsetup:
	gcp_login set_credentials set_project

# GOOGLE BUCKET
gcpbucket_00_create:
	@gsutil mb -l ${GCR_REGION} -p ${GCP_PROJECT_ID} gs://${BUCKET_NAME}

# GOOGLE INSTANCE
gcpinstance_00_create:
	gcloud compute instances create ${INSTANCE} --image-project=${IMAGE_PROJECT} --image-family=${IMAGE_FAMILY}

# GOOGLE INSTANCE
gcpinstance_01_start:
	gcloud compute instances start ${INSTANCE} --project ${GCP_PROJECT_ID} --zone ${GCE_ZONE}

# GOOGLE INSTANCE
gcpinstance_02_connect:
	gcloud beta compute ssh ${INSTANCE} --project ${GCP_PROJECT_ID} --zone ${GCE_ZONE}

# GOOGLE INSTANCE
gcpinstance_03_stop:
	gcloud compute instances stop ${INSTANCE} --project ${GCP_PROJECT_ID} --zone ${GCE_ZONE}

# GOOGLE INSTANCE GITHUB
gcpinstance_04_copyssh:
	gcloud compute scp ~/.ssh/id_ed25519 ${INSTANCE}:~/.ssh/

# GOOGLE INSTANCE GOOGLE CREDENTIALS
gcpinstance_05_copyjson:
	gcloud compute scp ${GOOGLE_APPLICATION_CREDENTIALS} ${INSTANCE}:~/.ssh/
	gcloud compute ssh ${INSTANCE} --command "echo 'export GOOGLE_APPLICATION_CREDENTIALS=~/.ssh/$(basename ${GOOGLE_APPLICATION_CREDENTIALS})' >> ~/.zshrc"
# DOCKER
docker_00_buildimage:
	sudo docker build -t ${GCR_MULTI_REGION}/${GCP_PROJECT_ID}/${DOCKER_IMAGE_NAME} .

# DOCKER
docker_01_imagelist:
	export IMAGES=$(docker images -aq)

# DOCKER
docker_02_deleteimages:
	docker rmi -f ${IMAGES}

# DOCKER RUN INTERACTIVE
docker_01_runshell:
	docker run -it --env-file .env ${GCR_MULTI_REGION}/${GCP_PROJECT_ID}/${DOCKER_IMAGE_NAME} sh

# DOCKER RUN LOCALLY
docker_02_runimage:
	sudo docker run -e PORT=8000 -p 8080:8000 --env-file .env ${GCR_MULTI_REGION}/${GCP_PROJECT_ID}/${DOCKER_IMAGE_NAME}

# DOCKER
docker_03_pushtoregistry:
	docker push ${GCR_MULTI_REGION}/${GCP_PROJECT_ID}/${DOCKER_IMAGE_NAME}

# DOCKER
docker_04_deploycontainer:
	gcloud run deploy --image ${GCR_MULTI_REGION}/${GCP_PROJECT_ID}/${DOCKER_IMAGE_NAME} --platform managed --region ${GCR_REGION}

# bonus

GCP_cloud_config:
	gcloud run services describe ${DOCKER_IMAGE_NAME} --format export > service.yaml

GCP_update_config:
	gcloud run services replace service.yaml

prefect_server_start:
	prefect server start --postgres-port 5433 --ui-port 8088

prefect_agent_start:
	prefect agent local start

prefect_project_create:
	prefect create project ${PREFECT_FLOW_NAME}

################### DATA SOURCES ACTIONS ################

# BIG QUERY ERASE
delete_new_source:
	-bq rm -f ${DATASET}.train_new.csv
	-bq rm -f ${DATASET}.val_new.csv
	-rm ~/.lewagon/mlops/data/raw/train_new.csv
	-rm ~/.lewagon/mlops/data/raw/val_new.csv

# BIG QUERY RESET RESOURCES
reset_sources_env:
	-bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.val_processed_${DATASET_SIZE} ${GS_DIR}/processed/val_processed_${DATASET_SIZE}.csv

# BIG QUERY SHOW RESOURCES
show_sources_env:
	-ls -laR ~/.lewagon/mlops/data | grep ${DATASET_SIZE}
	-bq ls ${DATASET}
	-bq show ${DATASET}.train_${DATASET_SIZE}
	-bq show ${DATASET}.val_${DATASET_SIZE}
	-bq show ${DATASET}.train_processed_${DATASET_SIZE}
	-bq show ${DATASET}.val_processed_${DATASET_SIZE}

# ----------------------------------
#      SERVICES
# ----------------------------------

mlflow_local:
	@mlflow server --host 0.0.0.0 --port 8080 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

inference_local:
	@uvicorn services.inference.app.main:app --host 0.0.0.0 --port 8080

trainer_local:
	@python -m services.trainer.train_entrypoint

# ----------------------------------
#      DOCKER COMPOSE (SERVICES)
# ----------------------------------

docker-build:
	@docker compose build

docker-up:
	@docker compose up -d

docker-down:
	@docker compose down

docker-logs:
	@docker compose logs -f --tail=200

docker-restart:
	@docker compose down
	@docker compose up -d

docker-ps:
	@docker compose ps

docker-local:
	@test -f .env.docker || (echo ".env.docker not found"; exit 1)
	@docker compose up --build
