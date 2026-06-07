"""Backend-aware checkpoint save/load.

DEEPSCULPT_BACKEND=local → torch.save / torch.load against a local path.
DEEPSCULPT_BACKEND=gcs   → same, but the path can be `gs://...` and the file
                          is uploaded/downloaded via google-cloud-storage.

Designed to be a drop-in replacement for torch.save / torch.load calls in
training code that needs to be GCS-aware.

Usage:
    from runpod.scripts.checkpoint_io import save, load

    save(model.state_dict(), "checkpoints/run42/generator.pt")
    state = load("checkpoints/run42/generator.pt")
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import torch

BACKEND = os.environ.get("DEEPSCULPT_BACKEND", "local").lower()
BUCKET = os.environ.get("GCS_BUCKET", "")
WORKSPACE = Path(os.environ.get("WORKSPACE_DIR", "/workspace"))
TOKEN_FILE = WORKSPACE / "control" / "gcs_token"


def _gcs_client():
    """Build a storage.Client backed by the short-lived bearer token written
    by entrypoint.sh / refresh-token.yml. Re-reads the file every call so a
    refresh between calls is picked up immediately.
    """
    from google.cloud import storage
    from google.oauth2.credentials import Credentials

    if TOKEN_FILE.exists() and TOKEN_FILE.stat().st_size > 0:
        token = TOKEN_FILE.read_text().strip()
        creds = Credentials(token=token)
        project = os.environ.get("GCS_PROJECT") or None
        return storage.Client(credentials=creds, project=project)
    # No token file → fall back to ADC. Will fail in pod, but tests / local
    # dev with `gcloud auth application-default login` still work.
    return storage.Client()


def _is_gs(path: str | Path) -> bool:
    return str(path).startswith("gs://")


def _parse_gs(path: str) -> tuple[str, str]:
    rest = path[len("gs://"):]
    bucket, _, blob = rest.partition("/")
    return bucket, blob


def _resolve(path: str | Path) -> str:
    """Normalize a path string.

    - If already `gs://...`, return as-is.
    - If BACKEND=gcs and path is relative, prefix with bucket prefix.
    - Otherwise treat as local; if relative, anchor under WORKSPACE.
    """
    s = str(path)
    if _is_gs(s):
        return s
    if BACKEND == "gcs" and not os.path.isabs(s):
        if not BUCKET:
            raise RuntimeError("DEEPSCULPT_BACKEND=gcs but GCS_BUCKET is unset")
        return f"gs://{BUCKET}/deepsculpt/{s}"
    if not os.path.isabs(s):
        return str(WORKSPACE / s)
    return s


def save(obj: Any, path: str | Path) -> None:
    """Save a torch object to local fs or GCS."""
    resolved = _resolve(path)
    if _is_gs(resolved):
        buf = io.BytesIO()
        torch.save(obj, buf)
        buf.seek(0)
        bucket_name, blob_name = _parse_gs(resolved)
        client = _gcs_client()
        blob = client.bucket(bucket_name).blob(blob_name)
        blob.upload_from_file(buf, content_type="application/octet-stream")
    else:
        Path(resolved).parent.mkdir(parents=True, exist_ok=True)
        torch.save(obj, resolved)


def load(path: str | Path, map_location: Any = None) -> Any:
    """Load a torch object from local fs or GCS."""
    resolved = _resolve(path)
    if _is_gs(resolved):
        bucket_name, blob_name = _parse_gs(resolved)
        client = _gcs_client()
        blob = client.bucket(bucket_name).blob(blob_name)
        buf = io.BytesIO()
        blob.download_to_file(buf)
        buf.seek(0)
        return torch.load(buf, map_location=map_location)
    return torch.load(resolved, map_location=map_location)


def exists(path: str | Path) -> bool:
    resolved = _resolve(path)
    if _is_gs(resolved):
        bucket_name, blob_name = _parse_gs(resolved)
        client = _gcs_client()
        return client.bucket(bucket_name).blob(blob_name).exists()
    return Path(resolved).exists()
