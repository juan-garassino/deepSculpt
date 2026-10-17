"""Guardian regression tests for the inference service (no GCP, no network).

External deps (google.auth, requests) are stubbed so these run in the plain
torch test env.
"""

from __future__ import annotations

import json
import sys
import types

import pytest


def _stub_gcp_modules(monkeypatch):
    google = types.ModuleType("google")
    auth = types.ModuleType("google.auth")
    auth.default = lambda scopes=None: (None, None)
    transport = types.ModuleType("google.auth.transport")
    treq = types.ModuleType("google.auth.transport.requests")
    treq.Request = object
    google.auth = auth
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.auth", auth)
    monkeypatch.setitem(sys.modules, "google.auth.transport", transport)
    monkeypatch.setitem(sys.modules, "google.auth.transport.requests", treq)
    if "requests" not in sys.modules:
        monkeypatch.setitem(sys.modules, "requests", types.ModuleType("requests"))
    sys.modules.pop("services.inference.app.gcp_run_job", None)


# --------------------------------------------------------------------------
# /visualize-dataset must only read from the service's own bucket
# --------------------------------------------------------------------------

from services.inference.app.uri_utils import parse_allowed_gcs_uri  # noqa: E402


def test_dataset_uri_in_allowed_bucket_is_parsed():
    assert parse_allowed_gcs_uri("gs://my-bucket/data/x.npy", "my-bucket") == (
        "my-bucket",
        "data/x.npy",
    )


@pytest.mark.parametrize(
    "uri",
    [
        "gs://other-bucket/secret.npy",  # confused-deputy read of a foreign bucket
        "gs://my-bucket-evil/x.npy",  # prefix trick
        "gs://my-bucket",  # no object path (used to raise ValueError -> 500)
        "gs://my-bucket/",  # empty object
        "s3://my-bucket/x.npy",  # wrong scheme
        "my-bucket/x.npy",  # no scheme
    ],
)
def test_dataset_uri_outside_allowed_bucket_rejected(uri):
    with pytest.raises(ValueError):
        parse_allowed_gcs_uri(uri, "my-bucket")


# --------------------------------------------------------------------------
# gcloud fallback must not let commas in the JSON split --set-env-vars
# --------------------------------------------------------------------------


def test_gcloud_fallback_env_survives_commas(monkeypatch):
    _stub_gcp_modules(monkeypatch)
    from services.inference.app import gcp_run_job

    captured = {}

    class _Result:
        stdout = "{}"

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        assert "shell" not in kwargs or kwargs["shell"] is False
        return _Result()

    monkeypatch.setattr(gcp_run_job.subprocess, "run", fake_run)
    params = {"epochs": 2, "batch_size": 4}
    gcp_run_job.run_job_via_gcloud("proj", "europe-west1", "job", params)

    flag = next(a for a in captured["cmd"] if a.startswith("--set-env-vars="))
    value = flag[len("--set-env-vars="):]
    # gcloud splits on ',' unless an alternate delimiter is declared as ^D^
    assert value.startswith("^"), value
    delim = value[1 : value.index("^", 1)]
    body = value[value.index("^", 1) + 1 :]
    pairs = body.split(delim)
    assert len(pairs) == 1
    key, _, raw = pairs[0].partition("=")
    assert key == "TRAIN_REQUEST_JSON"
    assert json.loads(raw) == params


# --------------------------------------------------------------------------
# G-011: /train/status must accept real execution names and nothing else
# --------------------------------------------------------------------------

from services.inference.app.uri_utils import normalize_execution_name  # noqa: E402

_FULL = "projects/p1/locations/europe-west1/jobs/deepsculpt-train/executions/deepsculpt-train-abc12"


def test_execution_full_name_accepted():
    assert normalize_execution_name(_FULL, "p1", "europe-west1", "deepsculpt-train") == _FULL


def test_execution_short_id_expanded():
    assert (
        normalize_execution_name("deepsculpt-train-abc12", "p1", "europe-west1", "deepsculpt-train")
        == _FULL
    )


@pytest.mark.parametrize(
    "bad",
    [
        "projects/other/locations/europe-west1/jobs/deepsculpt-train/executions/x",  # foreign project
        "projects/p1/locations/europe-west1/jobs/other-job/executions/x",  # foreign job
        "projects/p1/locations/europe-west1/services/api",  # different resource type
        "projects/p1/locations/europe-west1/jobs/deepsculpt-train/executions/../../x",
        "../secrets",
        "a?b=c",
        "",
    ],
)
def test_execution_name_rejects_other_resources(bad):
    with pytest.raises(ValueError):
        normalize_execution_name(bad, "p1", "europe-west1", "deepsculpt-train")


# --------------------------------------------------------------------------
# Request size bounds (cheap DoS guard on the public JSON body)
# --------------------------------------------------------------------------


def test_inference_request_vectors_are_bounded():
    pydantic = pytest.importorskip("pydantic")
    from services.inference.app.schemas import InferenceRequest

    with pytest.raises(pydantic.ValidationError):
        InferenceRequest(input_vector=[0.0] * 100_000)
    with pytest.raises(pydantic.ValidationError):
        InferenceRequest(conditioning=[0.0] * 100_000)
    assert InferenceRequest(input_vector=[0.0] * 128).input_vector is not None
