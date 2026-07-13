"""Trainer for the 3D KL-autoencoder (latent-diffusion Stage B).

Binary mode: BCE-on-logits + tiny KL. RGBA colour mode (palette_cfg set):
channel 0 (alpha) keeps the proven BCE recipe; channels 1-3 (RGB) get a masked
MSE (occupied voxels only — empty-voxel colour is meaningless and must not emit
gradient). No BCE pos_weight, ever — it would move the alpha decision threshold
off 0.5 and break every threshold-0.5 consumer downstream.

Gate before any latent diffusion run: alpha IoU@0.5 >= 0.95 AND (binary)
|occupancy error| < 0.01 or (RGBA) masked RGB MAE < 0.05.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base_trainer import BaseTrainer, TrainingConfig


class AutoencoderTrainer(BaseTrainer):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        config: TrainingConfig,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        kl_weight: float = 1e-6,
        palette_cfg: Optional[Any] = None,   # PaletteConfig => RGBA colour mode
        rgb_weight: float = 1.0,
    ):
        super().__init__(model, optimizer, config, scheduler, device)
        self.kl_weight = kl_weight
        self.palette_cfg = palette_cfg
        self.rgb_weight = rgb_weight
        self.logger = logging.getLogger(__name__)
        self.last_epoch_metrics: Dict[str, float] = {}
        # Fixed held-out volumes for recon snapshots, set by the CLI wiring.
        self._snapshot_batch: Optional[torch.Tensor] = None

    def _volumes(self, batch: Any) -> torch.Tensor:
        """Target volume: [B,1,D,H,W] occupancy, or [B,4,D,H,W] RGBA in colour
        mode (built from the batch's class colours + sample index)."""
        if self.palette_cfg is not None:
            from deepsculpt.core.data.transforms.palette import build_rgba

            if not (isinstance(batch, dict) and "colors" in batch and "index" in batch):
                raise ValueError("RGBA autoencoder needs 'colors' and 'index' in the batch")
            structure = batch["structure"].to(self.device)
            colors = batch["colors"].to(self.device)
            index = batch["index"].to(self.device)
            return build_rgba(structure, colors, index, self.palette_cfg)
        x = batch["structure"] if isinstance(batch, dict) else batch
        if x.dim() == 4:
            x = x.unsqueeze(1)
        return x.float().to(self.device)

    def _losses(self, logits: torch.Tensor, x: torch.Tensor):
        """Returns (recon_loss, extra_metrics). Alpha BCE always; masked RGB MSE
        when the target carries colour channels."""
        alpha_logits = logits[:, 0:1]
        alpha = x[:, 0:1]
        bce = F.binary_cross_entropy_with_logits(alpha_logits, alpha)
        extra = {"bce": bce.item()}
        recon = bce
        if x.shape[1] == 4:
            rgb_pred = torch.sigmoid(logits[:, 1:4])
            mask = (alpha > 0.5).float()  # [B,1,D,H,W] broadcast over 3 channels
            denom = mask.sum().clamp_min(1.0) * 3.0
            mse = (((rgb_pred - x[:, 1:4]) ** 2) * mask).sum() / denom
            recon = bce + self.rgb_weight * mse
            extra["rgb_mse"] = mse.item()
        return recon, extra

    def _iou(self, logits: torch.Tensor, x: torch.Tensor) -> float:
        pred = logits[:, 0:1] >= 0          # sigmoid(alpha_logit) >= 0.5
        truth = x[:, 0:1] >= 0.5
        inter = (pred & truth).sum().item()
        union = (pred | truth).sum().item()
        return inter / max(union, 1)

    def train_step(self, batch: Any) -> Dict[str, float]:
        x = self._volumes(batch)
        self.optimizer.zero_grad()
        logits, mu, logvar = self.model(x)
        recon, extra = self._losses(logits, x)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon + self.kl_weight * kl
        loss.backward()
        if self.config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
        self.optimizer.step()

        with torch.no_grad():
            iou = self._iou(logits, x)
        return {"loss": loss.item(), "kl": kl.item(), "iou": iou, **extra}

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        sums: Dict[str, float] = {}
        n = 0
        for batch in dataloader:
            step = self.train_step(batch)
            for k, v in step.items():
                sums[k] = sums.get(k, 0.0) + v
            n += 1
        metrics = {k: v / max(n, 1) for k, v in sums.items()}
        self.last_epoch_metrics = metrics
        return metrics

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        inter = union = 0
        occ_pred = occ_true = rgb_mae = recon_sum = 0.0
        rgb_batches = batches = 0
        for batch in dataloader:
            x = self._volumes(batch)
            mu, _ = self.model.encode(x)
            logits = self.model.decode(mu, return_logits=True)
            recon, _ = self._losses(logits, x)
            recon_sum += recon.item()
            pred = logits[:, 0:1] >= 0
            truth = x[:, 0:1] >= 0.5
            inter += (pred & truth).sum().item()
            union += (pred | truth).sum().item()
            occ_pred += pred.float().mean().item()
            occ_true += truth.float().mean().item()
            if x.shape[1] == 4:
                rgb_pred = torch.sigmoid(logits[:, 1:4])
                mask = (x[:, 0:1] > 0.5).float()
                denom = mask.sum().clamp_min(1.0) * 3.0
                rgb_mae += ((rgb_pred - x[:, 1:4]).abs() * mask).sum().item() / denom.item()
                rgb_batches += 1
            batches += 1
        b = max(batches, 1)
        out = {
            "loss": recon_sum / b,               # mirrors training recon for is_best
            "iou": inter / max(union, 1),
            "occupancy_error": abs(occ_pred - occ_true) / b,
        }
        if rgb_batches:
            out["rgb_mae"] = rgb_mae / rgb_batches
        return out

    def _after_epoch(self, epoch: int, train_metrics: Dict[str, float],
                     val_metrics: Optional[Dict[str, float]] = None,
                     is_best: bool = False) -> None:
        if self._snapshot_batch is None or not self.config.snapshot_dir:
            return
        from pathlib import Path

        snap_dir = Path(self.config.snapshot_dir)
        snap_dir.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            self.model.eval()
            x = self._snapshot_batch.to(self.device)
            mu, _ = self.model.encode(x)
            recon = self.model.decode(mu)  # probs (RGBA or mono)
            self.model.train()
        # Originals then reconstructions. Mono -> [2N,D,H,W]; RGBA keeps the
        # channel dim -> [2N,4,D,H,W], both rendered by render_walk --style voxel.
        stack = torch.cat([x, recon]).cpu()
        if stack.shape[1] == 1:
            stack = stack.squeeze(1)
        torch.save(stack.half(), snap_dir / f"recon_epoch_{epoch + 1:03d}.pt")
