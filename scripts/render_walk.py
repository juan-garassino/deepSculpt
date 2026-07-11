"""Render a latent-walk volume sequence (walk_volumes.pt) into a polished GIF.

latent-walk already saves every generated volume to walk_volumes.pt, so
aesthetics can be iterated here without re-running the generator. Renders
each volume via marching cubes -> Poly3DCollection (orders of magnitude
faster than ax.voxels at 64^3) with depth-shaded faces and a slowly
orbiting camera.

Usage:
  python scripts/render_walk.py <walk_volumes.pt> [--out walk_hq.gif]
      [--threshold 0.5] [--fps 12] [--size 720] [--orbit 90]
      [--elev 22] [--azim0 -60] [--color storm] [--boomerang]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

PALETTES = {
    # dark-to-light ramps applied along the height (z) axis
    "storm": ["#1b2a41", "#3b6ea5", "#9db8d9", "#f2f5fa"],
    "clay": ["#4a2c2a", "#a0522d", "#d9a066", "#f5e6d3"],
    "mono": ["#111111", "#555555", "#aaaaaa", "#f0f0f0"],
}


def load_volumes(path: Path) -> np.ndarray:
    vols = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(vols, torch.Tensor):
        vols = vols.float().numpy()
    vols = np.asarray(vols)
    return vols.reshape(vols.shape[0], *vols.shape[-3:])


def mesh_frame(ax, vol: np.ndarray, threshold: float, cmap, dim: int) -> None:
    from skimage import measure

    if not (vol.min() < threshold < vol.max()):
        return  # empty frame: leave axes blank rather than crash
    verts, faces, _, _ = measure.marching_cubes(vol, level=threshold)
    tri = verts[faces]
    # shade by face height (z) for a cheap directional-light feel
    zc = tri[:, :, 2].mean(axis=1)
    zn = (zc - zc.min()) / max(zc.max() - zc.min(), 1e-6)
    colors = cmap(0.25 + 0.7 * zn)
    pc = Poly3DCollection(tri, facecolors=colors, edgecolors="none")
    ax.add_collection3d(pc)
    ax.set_xlim(0, dim)
    ax.set_ylim(0, dim)
    ax.set_zlim(0, dim)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("volumes", type=Path)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--fps", type=float, default=12.0)
    p.add_argument("--size", type=int, default=720, help="output px (square)")
    p.add_argument("--orbit", type=float, default=90.0,
                   help="total camera azimuth sweep over the walk (degrees)")
    p.add_argument("--elev", type=float, default=22.0)
    p.add_argument("--azim0", type=float, default=-60.0)
    p.add_argument("--color", choices=sorted(PALETTES), default="storm")
    p.add_argument("--boomerang", action="store_true",
                   help="append the reversed sequence for a seamless loop")
    args = p.parse_args()

    vols = load_volumes(args.volumes)
    dim = vols.shape[-1]
    cmap = LinearSegmentedColormap.from_list(args.color, PALETTES[args.color])
    out = args.out or args.volumes.with_name("walk_hq.gif")

    dpi = 100
    fig = plt.figure(figsize=(args.size / dpi, args.size / dpi), dpi=dpi)
    frames = []
    n = len(vols)
    for i, vol in enumerate(vols):
        fig.clf()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_axis_off()
        ax.set_facecolor("white")
        ax.set_box_aspect((1, 1, 1))
        mesh_frame(ax, vol, args.threshold, cmap, dim)
        ax.view_init(elev=args.elev,
                     azim=args.azim0 + args.orbit * (i / max(n - 1, 1)))
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(frame)
        print(f"frame {i + 1}/{n}", flush=True)
    plt.close(fig)

    if args.boomerang:
        frames = frames + frames[-2:0:-1]

    import imageio

    imageio.mimsave(out, frames, duration=1.0 / args.fps, loop=0)
    print(f"wrote {out} ({len(frames)} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
