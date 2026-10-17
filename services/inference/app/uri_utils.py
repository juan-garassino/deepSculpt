"""Dependency-free URI helpers for the inference API."""

from typing import Tuple


def parse_allowed_gcs_uri(uri: str, allowed_bucket: str) -> Tuple[str, str]:
    """Parse ``gs://bucket/object`` and require ``bucket == allowed_bucket``.

    The service account behind the API can usually read more than the
    models bucket, so user-supplied URIs must be pinned to it (otherwise
    any caller can make the service read arbitrary buckets).
    Raises ``ValueError`` on any malformed or foreign URI.
    """
    if not isinstance(uri, str) or not uri.startswith("gs://"):
        raise ValueError("gcs_uri must start with gs://")
    bucket, sep, blob = uri[len("gs://"):].partition("/")
    if not sep or not bucket or not blob:
        raise ValueError("gcs_uri must be gs://<bucket>/<object>")
    if not allowed_bucket or bucket != allowed_bucket:
        raise ValueError("gcs_uri must point to the configured models bucket")
    return bucket, blob


_SEGMENT_OK = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-_")


def _is_segment(value: str) -> bool:
    return bool(value) and len(value) <= 128 and set(value.lower()) <= _SEGMENT_OK


def normalize_execution_name(execution_id: str, project: str, region: str, job: str) -> str:
    """Return the full Cloud Run execution resource name for ``execution_id``.

    Accepts either the full ``projects/<p>/locations/<r>/jobs/<j>/executions/<id>``
    name (what ``/train`` returns) or the bare execution id, and requires the
    project/region/job to match the configured training job. The result is
    interpolated into a Run API URL sent with the service's own bearer token,
    so anything else (other resources, ``..``, query strings) is rejected.
    """
    prefix = f"projects/{project}/locations/{region}/jobs/{job}/executions/"
    if isinstance(execution_id, str) and execution_id.startswith(prefix):
        short = execution_id[len(prefix):]
    else:
        short = execution_id
    if not isinstance(short, str) or not _is_segment(short):
        raise ValueError("execution_id must be an execution of the configured training job")
    return prefix + short
