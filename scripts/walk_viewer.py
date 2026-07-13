"""Pack a walk/trajectory volume sequence into a self-contained three.js viewer.

The GIF renders flatten the sculpture into one camera path; this emits a
single HTML file with orbit controls + a step slider/play button so the walk
can actually be inspected (zoom into a colonnade, orbit a facade, scrub the
denoising). Voxels are drawn as instanced cubes — 30k voxels per frame is
trivial for WebGL where ax.voxels chokes at 64^3.

Data layout: per frame, exposed voxels only (interior voxels can never be
seen), packed as int16 xyz + int8 class, base64-embedded. A 162-frame 64^3
walk lands around 10-20 MB of HTML.

Usage:
  python scripts/walk_viewer.py walk_volumes.pt --out walk_viewer.html
      [--stride 1] [--threshold 0.5] [--title "gan-shodhan-color-001 @ epoch 8"]
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import torch

# Keep in sync with render_walk.py CLASS_PALETTE.
CLASS_PALETTE = {
    0: "#e8e4dc",   # mono / unknown — warm concrete
    1: "#d9d4cc", 2: "#efe9df", 3: "#c8c2b8", 4: "#c0392b",
    5: "#2e6da4", 6: "#d4a017", 7: "#b9b2a6", 8: "#8f887c",
    9: "#c0392b", 10: "#2e6da4", 11: "#d4a017", 12: "#3d8b5f",
}


def load_volumes(path: Path) -> tuple[np.ndarray, bool]:
    vols = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(vols, torch.Tensor):
        vols = vols.numpy()
    vols = np.asarray(vols)
    if vols.ndim == 5 and vols.shape[1] == 4:
        return vols.astype(np.float32), "rgba"
    is_class = np.issubdtype(vols.dtype, np.integer)
    return vols.reshape(vols.shape[0], *vols.shape[-3:]), ("class" if is_class else "mono")


def _exposed_mask(occ: np.ndarray) -> np.ndarray:
    interior = np.ones_like(occ)
    for axis in range(3):
        for shift in (1, -1):
            n = np.zeros_like(occ)
            src = [slice(None)] * 3
            dst = [slice(None)] * 3
            if shift == 1:
                dst[axis], src[axis] = slice(0, -1), slice(1, None)
            else:
                dst[axis], src[axis] = slice(1, None), slice(0, -1)
            n[tuple(dst)] = occ[tuple(src)]
            interior &= n
    return occ & ~interior


def exposed_voxels(vol: np.ndarray, threshold: float, kind: str) -> np.ndarray:
    """int16 rows for exposed voxels. class/mono: (x,y,z,class); rgba: (x,y,z,r,g,b)."""
    if kind == "rgba":
        alpha, rgb = vol[0], vol[1:]                 # (D,H,W), (3,D,H,W)
        visible = _exposed_mask(alpha > threshold)
        idx = np.argwhere(visible).astype(np.int16)
        cols = (rgb[:, visible].T.clip(0, 1) * 255).astype(np.int16)  # (M,3)
        return np.concatenate([idx, cols], axis=1)
    occ = vol > 0 if kind == "class" else vol > threshold
    visible = _exposed_mask(occ)
    idx = np.argwhere(visible).astype(np.int16)
    cls = (np.clip(vol[visible].astype(np.int16), 0, 12) if kind == "class"
           else np.zeros(len(idx), dtype=np.int16))
    return np.concatenate([idx, cls[:, None]], axis=1)


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  body { margin: 0; background: #fafafa; font-family: -apple-system, sans-serif; overflow: hidden; }
  #bar { position: fixed; left: 0; right: 0; bottom: 0; padding: 10px 16px;
         background: rgba(255,255,255,.92); display: flex; gap: 12px; align-items: center; }
  #bar input[type=range] { flex: 1; }
  #label { min-width: 90px; text-align: right; font-variant-numeric: tabular-nums; color: #333; }
  #title { position: fixed; top: 10px; left: 16px; color: #444; font-size: 14px; }
  button { font-size: 16px; padding: 2px 12px; }
</style>
</head>
<body>
<div id="title">__TITLE__</div>
<div id="bar">
  <button id="play">&#9654;</button>
  <input type="range" id="step" min="0" max="__MAXSTEP__" value="0" step="1">
  <span id="label">step 1/__NFRAMES__</span>
</div>
<script type="importmap">
{ "imports": {
  "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
  "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
} }
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const DIM = __DIM__;
const PALETTE = __PALETTE__;
const FRAMES_B64 = __FRAMES__;
const STRIDE = __STRIDE__;      // int16 per voxel: 4 (x,y,z,class) or 6 (x,y,z,r,g,b)
const RGBA = __RGBA__;
const frameCache = new Array(FRAMES_B64.length);
function frame(i) {  // lazy decode: 50+ MB of base64 would stall startup
  if (!frameCache[i]) {
    const bin = atob(FRAMES_B64[i]);
    const bytes = new Uint8Array(bin.length);
    for (let j = 0; j < bin.length; j++) bytes[j] = bin.charCodeAt(j);
    frameCache[i] = new Int16Array(bytes.buffer);
  }
  return frameCache[i];
}
const nFrames = FRAMES_B64.length;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xfafafa);
const camera = new THREE.PerspectiveCamera(40, innerWidth / innerHeight, 1, 2000);
camera.position.set(DIM * 1.6, DIM * 1.25, DIM * 1.6);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(devicePixelRatio);
document.body.appendChild(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, DIM * 0.05, 0);
controls.enableDamping = true;

scene.add(new THREE.AmbientLight(0xffffff, 0.75));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(1.5, 3, 2);
scene.add(sun);
const fill = new THREE.DirectionalLight(0xbfd4ff, 0.5);
fill.position.set(-2, 1, -1.5);
scene.add(fill);

const geo = new THREE.BoxGeometry(1, 1, 1);
const mat = new THREE.MeshLambertMaterial();
// bytes/base64 = len*3/4; int16 = 2 bytes; voxels = that / (2*STRIDE)
const maxCount = Math.max(...FRAMES_B64.map(b => Math.floor(b.length * 3 / 4 / (2 * STRIDE)))) + 8;
const mesh = new THREE.InstancedMesh(geo, mat, maxCount);
mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(maxCount * 3), 3);
scene.add(mesh);

const colors = PALETTE.map(hex => new THREE.Color(hex));
const c = new THREE.Color();
const m = new THREE.Matrix4();
function show(i) {
  const f = frame(i);
  const n = f.length / STRIDE;
  for (let v = 0; v < n; v++) {
    const o = v * STRIDE;
    // volume (x,y,z) -> scene (x, up=z, y), centered
    m.makeTranslation(f[o] - DIM/2 + .5, f[o+2] - DIM/2 + .5, f[o+1] - DIM/2 + .5);
    mesh.setMatrixAt(v, m);
    if (RGBA) { c.setRGB(f[o+3]/255, f[o+4]/255, f[o+5]/255); mesh.setColorAt(v, c); }
    else mesh.setColorAt(v, colors[f[o+3]] || colors[0]);
  }
  mesh.count = n;
  mesh.instanceMatrix.needsUpdate = true;
  mesh.instanceColor.needsUpdate = true;
  document.getElementById('label').textContent = `step ${i+1}/${nFrames}`;
}

const slider = document.getElementById('step');
slider.addEventListener('input', () => { playing = false; btn.textContent = '\\u25B6'; show(+slider.value); });
const btn = document.getElementById('play');
let playing = false, last = 0;
btn.addEventListener('click', () => { playing = !playing; btn.textContent = playing ? '\\u23F8' : '\\u25B6'; });

show(0);
renderer.setAnimationLoop((t) => {
  if (playing && t - last > 120) {
    last = t;
    slider.value = (+slider.value + 1) % nFrames;
    show(+slider.value);
  }
  controls.update();
  renderer.render(scene, camera);
});
addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});
</script>
</body>
</html>
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("volumes", type=Path)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--stride", type=int, default=1, help="keep every Nth frame")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--title", default=None)
    args = p.parse_args()

    vols, kind = load_volumes(args.volumes)
    vols = vols[:: args.stride]
    dim = vols.shape[-1]
    rgba = kind == "rgba"
    stride = 6 if rgba else 4

    packed = []
    for i, vol in enumerate(vols):
        vox = exposed_voxels(vol, args.threshold, kind)
        packed.append(base64.b64encode(vox.astype("<i2").tobytes()).decode())
        print(f"frame {i + 1}/{len(vols)}: {len(vox)} voxels", flush=True)

    palette = [CLASS_PALETTE[k] for k in range(13)]
    out = args.out or args.volumes.with_suffix(".html")
    html = (HTML_TEMPLATE
            .replace("__TITLE__", args.title or args.volumes.stem)
            .replace("__MAXSTEP__", str(len(packed) - 1))
            .replace("__NFRAMES__", str(len(packed)))
            .replace("__DIM__", str(dim))
            .replace("__STRIDE__", str(stride))
            .replace("__RGBA__", "true" if rgba else "false")
            .replace("__PALETTE__", json.dumps(palette))
            .replace("__FRAMES__", json.dumps(packed)))
    Path(out).write_text(html)
    print(f"wrote {out} ({Path(out).stat().st_size / 1e6:.1f} MB, {len(packed)} frames)")


if __name__ == "__main__":
    main()
