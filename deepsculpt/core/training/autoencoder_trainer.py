"""Trainer for the 3D KL-autoencoder (latent-diffusion Stage B).

BCE-on-logits + tiny KL. No BCE pos_weight, ever — it would move the optimal
decision threshold away from 0.5 and silently break every downstream consumer
(volume_export, snapshot stats, the IoU gate) that thresholds at 0.5.
Held-out IoU@0.5 is the quality gate before any latent diffusion run:
IoU >= 0.95 and |occupancy error| < 0.01.
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
    ):
        super().__init__(model, optimizer, config, scheduler, device)
        self.kl_weight = kl_weight
        self.logger = logging.getLogger(__name__)
        self.last_epoch_metrics: Dict[str, float] = {}
        # Fixed held-out volumes for recon snapshots, set by the CLI wiring.
        self._snapshot_batch: Optional[torch.Tensor] = None

    def _volumes(self, batch: Any) -> torch.Tensor:
        x = batch["structure"] if isinstance(batch, dict) else batch
        if x.dim() == 4:
            x = x.unsqueeze(1)
        return x.float().to(self.device)

    def train_step(self, batch: Any) -> Dict[str, float]:
        x = self._volumes(batch)
        self.optimizer.zero_grad()
        logits, mu, logvar = self.model(x)
        bce = F.binary_cross_entropy_with_logits(logits, x)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = bce + self.kl_weight * kl
        loss.backward()
        if self.config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
        self.optimizer.step()

        with torch.no_grad():
            pred = logits >= 0  # sigmoid(logits) >= 0.5
            truth = x >= 0.5
            inter = (pred & truth).sum().item()
            union = (pred | truth).sum().item()
        return {
            "loss": loss.item(),
            "bce": bce.item(),
            "kl": kl.item(),
            "iou": inter / max(union, 1),
        }

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
        bce_sum = inter = union = 0
        occ_pred = occ_true = 0.0
        batches = 0
        for batch in dataloader:
            x = self._volumes(batch)
            mu, _ = self.model.encode(x)
            logits = self.model.decode(mu, return_logits=True)
            bce_sum += F.binary_cross_entropy_with_logits(logits, x).item()
            pred = logits >= 0
            truth = x >= 0.5
            inter += (pred & truth).sum().item()
            union += (pred | truth).sum().item()
            occ_pred += pred.float().mean().item()
            occ_true += truth.float().mean().item()
            batches += 1
        b = max(batches, 1)
        return {
            "loss": bce_sum / b,
            "iou": inter / max(union, 1),
            "occupancy_error": abs(occ_pred - occ_true) / b,
        }

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
            recon = self.model.decode(mu)  # probs
            self.model.train()
        # originals then reconstructions — renders as 2N frames with the
        # existing walk tooling (render_walk.py --style voxel).
        stack = torch.cat([x, recon]).squeeze(1).half().cpu()
        torch.save(stack, snap_dir / f"recon_epoch_{epoch + 1:03d}.pt")
