import json
import os
import subprocess
from typing import Any, Dict

import google.auth
from google.auth.transport.requests import Request
import requests


_ENV_DELIM = "~|~"


def _get_access_token() -> str:
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    credentials.refresh(Request())
    return credentials.token


def run_job_via_api(project_id: str, region: str, job_name: str, train_params: Dict[str, Any]) -> Dict[str, Any]:
    token = _get_access_token()
    url = (
        f"https://run.googleapis.com/v2/projects/{project_id}/locations/{region}/jobs/{job_name}:run"
    )
    payload = {
        "overrides": {
            "containerOverrides": [
                {
                    "env": [
                        {
                            "name": "TRAIN_REQUEST_JSON",
                            "value": json.dumps(train_params),
                        }
                    ]
                }
            ]
        }
    }
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def run_job_via_gcloud(project_id: str, region: str, job_name: str, train_params: Dict[str, Any]) -> Dict[str, Any]:
    env_json = json.dumps(train_params)
    if _ENV_DELIM in env_json:
        raise ValueError("train params may not contain the env delimiter")
    cmd = [
        "gcloud",
        "run",
        "jobs",
        "execute",
        job_name,
        f"--project={project_id}",
        f"--region={region}",
        # gcloud splits --set-env-vars on ','; JSON is full of commas, so
        # declare an alternate delimiter (^DELIM^) that cannot occur in it.
        f"--set-env-vars=^{_ENV_DELIM}^TRAIN_REQUEST_JSON={env_json}",
        "--format=json",
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def trigger_training_job(
    project_id: str,
    region: str,
    job_name: str,
    train_params: Dict[str, Any],
    allow_gcloud_fallback: bool,
) -> Dict[str, Any]:
    try:
        return run_job_via_api(project_id, region, job_name, train_params)
    except Exception:
        if not allow_gcloud_fallback:
            raise
        return run_job_via_gcloud(project_id, region, job_name, train_params)
