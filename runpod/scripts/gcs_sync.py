"""Bidirectional sync helpers between /workspace and gs://<bucket>/deepsculpt/...

Usage:
    python -m runpod.scripts.gcs_sync push checkpoints
    python -m runpod.scripts.gcs_sync pull data
    python -m runpod.scripts.gcs_sync push results --run-id 20260606-104530

Falls back to shelling out to `gsutil` (already in the image). The Python
google-cloud-storage client is imported lazily for callers that want it.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(os.environ.get("WORKSPACE_DIR", "/workspace"))
BUCKET = os.environ.get("GCS_BUCKET", "")
PREFIX = "deepsculpt"


def gcs_root() -> str:
    if not BUCKET:
        raise SystemExit("GCS_BUCKET env var is required")
    return f"gs://{BUCKET}/{PREFIX}"


def _rsync(src: str, dst: str) -> int:
    cmd = ["gsutil", "-m", "-q", "rsync", "-r", src, dst]
    print(f"  $ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd)


def push(kind: str, run_id: str | None = None) -> None:
    """Local → GCS."""
    if kind in ("checkpoints", "results"):
        if not run_id:
            raise SystemExit(f"--run-id required for {kind}")
        src = WORKSPACE / kind / run_id
        dst = f"{gcs_root()}/{kind}/{run_id}"
    elif kind == "data":
        src = WORKSPACE / "data"
        dst = f"{gcs_root()}/data"
    else:
        raise SystemExit(f"unknown kind: {kind}")
    src.mkdir(parents=True, exist_ok=True)
    sys.exit(_rsync(str(src), dst))


def pull(kind: str, run_id: str | None = None) -> None:
    """GCS → local."""
    if kind in ("checkpoints", "results"):
        if not run_id:
            raise SystemExit(f"--run-id required for {kind}")
        src = f"{gcs_root()}/{kind}/{run_id}"
        dst = WORKSPACE / kind / run_id
    elif kind == "data":
        src = f"{gcs_root()}/data"
        dst = WORKSPACE / "data"
    else:
        raise SystemExit(f"unknown kind: {kind}")
    dst.mkdir(parents=True, exist_ok=True)
    sys.exit(_rsync(src, str(dst)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("direction", choices=["push", "pull"])
    p.add_argument("kind", choices=["checkpoints", "results", "data"])
    p.add_argument("--run-id", default=os.environ.get("RUN_ID"))
    args = p.parse_args()

    (push if args.direction == "push" else pull)(args.kind, args.run_id)


if __name__ == "__main__":
    main()
