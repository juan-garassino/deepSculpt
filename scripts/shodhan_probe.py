"""Shodhan dataset probe: gates + contact sheet + interactive 3D HTML.

Usage:
    python scripts/shodhan_probe.py --n 200 --out results/shodhan_probe
Gates (spec): masked pairwise IoU < 0.45; per-voxel frequency < 0.5 outside
the slab/L-band template. Renders let the user visually approve the grammar
on REAL 64^3 samples before full generation.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from deepsculpt.core.data.generation import shodhan as sh


def probe_gates(n: int = 200, seed_start: int = 0) -> dict:
    vols, temps = [], []
    for i in range(n):
        v, _ = sh.generate_structure_with_params(seed_start + i)
        vols.append(v > 0)
        temps.append((v == sh.SLAB) | (v == sh.EDGE))
    vols = np.stack(vols)
    template = np.stack(temps).mean(0) > 0.95        # near-constant slab/L voxels
    freq = vols.mean(0)
    max_freq_out = float(freq[~template].max())

    ious = []
    idx = np.random.default_rng(0).permutation(n)[:min(n, 60)]
    for a, b in combinations(idx, 2):
        va, vb = vols[a] & ~template, vols[b] & ~template
        union = (va | vb).sum()
        ious.append((va & vb).sum() / union if union else 0.0)
    iou = float(np.mean(ious))

    return {
        "n": n,
        "pairwise_iou_masked": iou,
        "max_freq_outside_template": max_freq_out,
        "occupancy_mean": float(vols.mean()),
        "pass": iou < 0.45 and max_freq_out < 0.5,
    }


def render_reports(n_render: int, out_dir: Path, seed_start: int = 0) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(seed_start, seed_start + n_render))
    fig, axes = plt.subplots(len(seeds), 4, figsize=(11, 2.6 * len(seeds)), squeeze=False)
    vols = []
    for r, s in enumerate(seeds):
        v, _ = sh.generate_structure_with_params(s)
        vols.append(v)
        b = v > 0
        for c in range(3):
            axes[r][c].imshow(b.max(axis=c), cmap="gray_r", vmin=0, vmax=1)
            axes[r][c].set_xticks([]); axes[r][c].set_yticks([])
        axes[r][3].imshow(b[:, :, 32], cmap="gray_r", vmin=0, vmax=1)
        axes[r][3].set_xticks([]); axes[r][3].set_yticks([])
        axes[r][0].set_ylabel(f"s{s} occ {b.mean():.3f}", fontsize=8)
    fig.suptitle("shodhan v2 probe — projections + mid slice")
    fig.tight_layout()
    fig.savefig(out_dir / "contact_sheet.png", dpi=110)
    plt.close(fig)

    palette = {sh.COL: "#5a5a5a", sh.SLAB: "#c8c4bc", sh.SCREEN: "#e4dccb",
               sh.PIPE_R: "#c0392b", sh.PIPE_B: "#2471a3", sh.PIPE_Y: "#f1c40f",
               sh.VOL: "#a8a29a", sh.EDGE: "#f7f6f2", sh.WALL_R: "#b5493a",
               sh.WALL_B: "#3a6bb5", sh.WALL_Y: "#e0b839", sh.WALL_G: "#6b8e4e"}
    figp = go.Figure()
    n3 = min(3, len(vols))
    for i in range(n3):
        v = vols[i]
        x, y, z = np.where(v > 0)
        cols = [palette.get(int(k), "#999") for k in v[x, y, z]]
        figp.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="markers",
                                    marker=dict(size=2, symbol="square", color=cols),
                                    name=f"seed {seeds[i]}", visible=(i == 0)))
    figp.update_layout(
        updatemenus=[dict(buttons=[dict(label=f"seed {seeds[i]}", method="update",
                                        args=[{"visible": [j == i for j in range(n3)]}])
                                   for i in range(n3)], x=0.02, y=0.98)],
        scene=dict(aspectmode="cube", xaxis=dict(visible=False),
                   yaxis=dict(visible=False), zaxis=dict(visible=False)),
        title="shodhan v2 probe — drag to rotate", height=800)
    figp.write_html(out_dir / "probe_3d.html", include_plotlyjs="cdn")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n-render", type=int, default=8)
    ap.add_argument("--out", default="results/shodhan_probe")
    args = ap.parse_args()
    out = Path(args.out)
    report = probe_gates(args.n)
    render_reports(args.n_render, out)
    (out / "gates_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print(f"renders -> {out}/contact_sheet.png, {out}/probe_3d.html")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
