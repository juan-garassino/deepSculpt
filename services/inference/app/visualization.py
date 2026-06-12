# NOTE: the render helpers here are intentionally duplicated from
# deepsculpt/core/visualization/volume_export.py (the canonical copy) —
# this service image installs only its own requirements.txt, not the
# deepsculpt package. Fix bugs in core first, then mirror here.
import os
import tempfile
from typing import Iterable, Tuple

import imageio.v2 as imageio
import numpy as np
from google.cloud import storage
from PIL import Image
from skimage import measure


def _normalize_slice(slice_2d: np.ndarray) -> np.ndarray:
    slice_min = float(slice_2d.min())
    slice_max = float(slice_2d.max())
    if slice_max - slice_min < 1e-6:
        return np.zeros_like(slice_2d)
    return (slice_2d - slice_min) / (slice_max - slice_min)


def _get_mid_slice(tensor: np.ndarray) -> np.ndarray:
    sample = tensor[0]
    if sample.ndim == 4:
        # shape: (C, D, H, W)
        channels, depth = sample.shape[0], sample.shape[1]
        mid = depth // 2
        if channels == 1:
            return _normalize_slice(sample[0, mid])
        # Use first 3 channels as RGB
        rgb = sample[:3, mid]
        rgb = np.stack([_normalize_slice(ch) for ch in rgb], axis=-1)
        return rgb
    # shape: (D, H, W)
    mid = sample.shape[0] // 2
    return _normalize_slice(sample[mid])


def save_middle_slice_png(tensor: np.ndarray, output_path: str) -> None:
    slice_2d = _get_mid_slice(tensor)
    if slice_2d.ndim == 2:
        image = Image.fromarray((slice_2d * 255).astype(np.uint8))
    else:
        image = Image.fromarray((slice_2d * 255).astype(np.uint8), mode="RGB")
    image.save(output_path)


def save_gif_from_volumes(volumes: Iterable[np.ndarray], output_path: str) -> None:
    frames = []
    for volume in volumes:
        slice_2d = _get_mid_slice(volume)
        frames.append((slice_2d * 255).astype(np.uint8))

    imageio.mimsave(output_path, frames, duration=0.2)


def save_gif_from_volume_slices(volume: np.ndarray, output_path: str) -> None:
    frames = []
    sample = volume[0]
    if sample.ndim == 4:
        channels, depth = sample.shape[0], sample.shape[1]
        for idx in range(depth):
            if channels == 1:
                slice_2d = _normalize_slice(sample[0, idx])
                frames.append((slice_2d * 255).astype(np.uint8))
            else:
                rgb = sample[:3, idx]
                rgb = np.stack([_normalize_slice(ch) for ch in rgb], axis=-1)
                frames.append((rgb * 255).astype(np.uint8))
    else:
        for idx in range(sample.shape[0]):
            slice_2d = _normalize_slice(sample[idx])
            frames.append((slice_2d * 255).astype(np.uint8))
    imageio.mimsave(output_path, frames, duration=0.1)


def _write_obj(verts: np.ndarray, faces: np.ndarray, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as handle:
        for v in verts:
            handle.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for f in faces:
            f1, f2, f3 = f + 1
            handle.write(f"f {f1} {f2} {f3}\n")


def _write_stl(verts: np.ndarray, faces: np.ndarray, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("solid deepsculpt\n")
        for f in faces:
            v1, v2, v3 = verts[f[0]], verts[f[1]], verts[f[2]]
            handle.write("  facet normal 0 0 0\n")
            handle.write("    outer loop\n")
            handle.write(f"      vertex {v1[0]} {v1[1]} {v1[2]}\n")
            handle.write(f"      vertex {v2[0]} {v2[1]} {v2[2]}\n")
            handle.write(f"      vertex {v3[0]} {v3[1]} {v3[2]}\n")
            handle.write("    endloop\n")
            handle.write("  endfacet\n")
        handle.write("endsolid deepsculpt\n")


def save_mesh(tensor: np.ndarray, output_path: str, fmt: str) -> None:
    sample = tensor[0]
    if sample.ndim == 4:
        sample = sample[0]
    threshold = float(sample.mean())
    verts, faces, _, _ = measure.marching_cubes(sample, level=threshold)
    if fmt == "obj":
        _write_obj(verts, faces, output_path)
    elif fmt == "stl":
        _write_stl(verts, faces, output_path)
    else:
        raise ValueError(f"Unsupported mesh format: {fmt}")


def upload_visualization(local_path: str, gcs_uri: str) -> str:
    bucket_name, blob_path = gcs_uri.replace("gs://", "").split("/", 1)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)
    return gcs_uri


def make_and_upload_png(output: np.ndarray, gcs_uri: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        local_path = tmp.name
    try:
        save_middle_slice_png(output, local_path)
        return upload_visualization(local_path, gcs_uri)
    finally:
        try:
            os.remove(local_path)
        except OSError:
            pass


def make_and_upload_gif(volumes: Iterable[np.ndarray], gcs_uri: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
        local_path = tmp.name
    try:
        save_gif_from_volumes(volumes, local_path)
        return upload_visualization(local_path, gcs_uri)
    finally:
        try:
            os.remove(local_path)
        except OSError:
            pass


def make_and_upload_gif_from_slices(volume: np.ndarray, gcs_uri: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
        local_path = tmp.name
    try:
        save_gif_from_volume_slices(volume, local_path)
        return upload_visualization(local_path, gcs_uri)
    finally:
        try:
            os.remove(local_path)
        except OSError:
            pass


def make_and_upload_mesh(output: np.ndarray, gcs_uri: str, fmt: str) -> str:
    suffix = ".obj" if fmt == "obj" else ".stl"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        local_path = tmp.name
    try:
        save_mesh(output, local_path, fmt)
        return upload_visualization(local_path, gcs_uri)
    finally:
        try:
            os.remove(local_path)
        except OSError:
            pass
