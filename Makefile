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
	@rm -fr results/checkpoints/*
	@rm -fr results/snapshots/*
	@rm -fr deepSculpt/results/*.svg
	@rm -fr deepSculpt/results/*.png
	@rm -fr deepSculpt/data/*.npy

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

run_test:
	curl -X POST \
    https://api.instagram.com/oauth/${access_token} client_id=${app-id} client_secret=${app-secret} \
		grant_type=${authorization_code} redirect_uri=${redirect-uri} code=${code}

run_trainer:
	python -m deepSculpt.trainer.trainer

run_sculptor:
	python -m deepSculpt.sculptor.sculptor

run_curator:
	python -m deepSculpt.curator.curator

run_emissary:
	python -m deepSculpt.emissary.emissary

run_manager:
	python -m deepSculpt.manager.manager

run_sampler:
	python -m deepSculpt.manager.tools.sampling


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
