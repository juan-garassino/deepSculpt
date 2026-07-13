"""Render training-run snapshots from GCS into a progression contact sheet.

Pulls snapshots/epoch_*.pt (generated-sample tensors saved once per epoch by
the trainers) for a RUN_ID from gs://garassino-ml-artifacts/deepsculpt/
checkpoints/<run_id>/, renders max-projections of up to 4 samples per epoch,
and writes results/run_monitor/<run_id>_progression.png.

Usage: python scripts/render_run_snapshots.py <run_id> [--max-epochs 8]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

GCS_ROOT = "gs://garassino-ml-artifacts/deepsculpt/checkpoints"


def sync_snapshots(run_id: str, dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    # snapshots live under <run_id>/<train_run_dir>/snapshots/epoch_NNN.pt
    ls = subprocess.run(
        ["gsutil", "ls", f"{GCS_ROOT}/{run_id}/**/snapshots/epoch_*.pt"],
        capture_output=True, text=True,
    )
    uris = [u for u in ls.stdout.splitlines() if u.endswith(".pt")]
    if not uris:
        return []
    subprocess.run(["gsutil", "-m", "-q", "cp", *uris, str(dest)], check=True)
    return sorted(dest.glob("epoch_*.pt"))


def render(run_id: str, snaps: list[Path], out: Path, samples_per_epoch: int = 4) -> None:
    rows = len(snaps)
    fig, axes = plt.subplots(rows, samples_per_epoch, figsize=(2.4 * samples_per_epoch, 2.4 * rows), squeeze=False)
    for r, snap in enumerate(snaps):
        vols = torch.load(snap, map_location="cpu", weights_only=False).float().numpy()
        if vols.ndim == 5 and vols.shape[1] == 4:   # RGBA: alpha channel for this occupancy grid
            vols = vols[:, 0]
        vols = vols.reshape(vols.shape[0], *vols.shape[-3:])
        for c in range(samples_per_epoch):
            ax = axes[r][c]
            ax.set_xticks([]); ax.set_yticks([])
            if c >= len(vols):
                ax.axis("off"); continue
            v = vols[c]
            occ = float((np.abs(v) > 0.5).mean())
            ax.imshow((np.abs(v) > 0.5).max(axis=0), cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xlabel(f"occ {occ:.3f}", fontsize=7)
        axes[r][0].set_ylabel(snap.stem, fontsize=8)
    fig.suptitle(f"{run_id} — generated samples per epoch (max projection)", fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"saved -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--max-epochs", type=int, default=8, help="render at most N epochs, evenly spaced")
    args = ap.parse_args()

    dest = Path("/tmp/ds_snaps") / args.run_id
    snaps = sync_snapshots(args.run_id, dest)
    if not snaps:
        print(f"no snapshots in GCS yet for {args.run_id}")
        return 1
    if len(snaps) > args.max_epochs:
        idx = np.linspace(0, len(snaps) - 1, args.max_epochs).round().astype(int)
        snaps = [snaps[i] for i in sorted(set(idx))]
    render(args.run_id, snaps, Path("results/run_monitor") / f"{args.run_id}_progression.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
