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
