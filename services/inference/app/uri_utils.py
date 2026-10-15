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
