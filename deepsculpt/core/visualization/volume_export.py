"""
Pure render/export helpers for 3D volumes (no cloud dependencies).

Slice/GIF/mesh writers are lifted from services/inference/app/visualization.py
(which is GCS-coupled and must not be imported from core); the contact sheet
and voxel-GIF renderers are new, built for latent walks and traversal grids.

Volumes are numpy arrays shaped (N, C, D, H, W), (C, D, H, W) or (D, H, W).
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Sequence

import imageio.v2 as imageio
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def _to_volume(arr: np.ndarray) -> np.ndarray:
    """Squeeze any input shape down to a single (D, H, W) scalar volume."""
    v = np.asarray(arr)
    while v.ndim > 3:
        v = v[0]
    if v.ndim != 3:
        raise ValueError(f"expected a 3D volume after squeezing, got shape {arr.shape}")
    return v


def _normalize_slice(slice_2d: np.ndarray) -> np.ndarray:
    lo, hi = float(slice_2d.min()), float(slice_2d.max())
    if hi - lo < 1e-6:
        return np.zeros_like(slice_2d)
    return (slice_2d - lo) / (hi - lo)


def _mid_slice(arr: np.ndarray) -> np.ndarray:
    vol = _to_volume(arr)
    return _normalize_slice(vol[vol.shape[0] // 2])


def save_middle_slice_png(volume: np.ndarray, output_path: str) -> None:
    img = (_mid_slice(volume) * 255).astype(np.uint8)
    Image.fromarray(img).save(output_path)


def save_gif_from_volumes(
    volumes: Iterable[np.ndarray],
    output_path: str,
    fps: float = 8.0,
    mode: str = "slice",
    threshold: float = 0.5,
) -> None:
    """Animate a sequence of volumes (one frame per volume).

    mode="slice": middle z-slice per frame (fast — fine for old machines).
    mode="voxel": full 3D voxel render per frame (slow, needs matplotlib).
    """
    if mode == "voxel":
        frames = [_voxel_frame(_to_volume(v), threshold) for v in volumes]
    else:
        frames = [(_mid_slice(v) * 255).astype(np.uint8) for v in volumes]
    imageio.mimsave(output_path, frames, duration=1.0 / fps)
    logger.info("Wrote %d-frame GIF to %s (mode=%s)", len(frames), output_path, mode)


def _voxel_frame(volume: np.ndarray, threshold: float) -> np.ndarray:
    """Render one 3D voxel view to an RGB array (matplotlib, headless)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(4, 4), dpi=80)
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(volume > threshold, facecolors="#3b6ea5", edgecolor="k", linewidth=0.1)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def render_contact_sheet(
    volumes: Sequence[np.ndarray],
    output_path: str,
    rows: int,
    cols: int,
    titles: Optional[Sequence[str]] = None,
    threshold: float = 0.5,
    mode: str = "slice",
) -> None:
    """Grid of renders — rows×cols, e.g. one row per traversed dimension."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(2.2 * cols, 2.4 * rows), dpi=90)
    for i, vol in enumerate(volumes[: rows * cols]):
        if mode == "voxel":
            ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
            ax.voxels(_to_volume(vol) > threshold, facecolors="#3b6ea5",
                      edgecolor="k", linewidth=0.1)
        else:
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.imshow(_mid_slice(vol), cmap="gray", vmin=0, vmax=1)
        ax.set_axis_off()
        if titles is not None and i < len(titles):
            ax.set_title(titles[i], fontsize=7)
    fig.tight_layout(pad=0.3)
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Wrote %dx%d contact sheet to %s", rows, cols, output_path)


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


def save_mesh(volume: np.ndarray, output_path: str, fmt: str = "obj",
              threshold: float = 0.5) -> None:
    """Marching-cubes mesh export (obj/stl). Requires scikit-image."""
    try:
        from skimage import measure
    except ImportError as e:
        raise ImportError(
            "Mesh export needs scikit-image: pip install scikit-image"
        ) from e

    vol = _to_volume(volume).astype(np.float32)
    if not (vol.min() < threshold < vol.max()):
        raise ValueError(
            f"threshold {threshold} outside volume range "
            f"[{vol.min():.3f}, {vol.max():.3f}] — no isosurface to extract"
        )
    verts, faces, _, _ = measure.marching_cubes(vol, level=threshold)
    if fmt == "obj":
        _write_obj(verts, faces, output_path)
    elif fmt == "stl":
        _write_stl(verts, faces, output_path)
    else:
        raise ValueError(f"Unsupported mesh format: {fmt}")
    logger.info("Wrote %s mesh (%d verts) to %s", fmt, len(verts), output_path)
