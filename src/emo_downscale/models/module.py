from typing import Dict

import torch
from torch import nn
import lightning.pytorch as pl
import torch.nn.functional as F


class DownscaleLightningModule(pl.LightningModule):
    def __init__(self, model: nn.Module, optimizer_cfg: Dict, scheduler_cfg: Dict):
        super().__init__()
        self.model = model
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage: str):
        x, y = batch  # x: (B, C_in, H, W), y: (B, C_out, H, W)
        y_hat = self(x)
        loss = F.l1_loss(y_hat, y)

        with torch.no_grad():
            mse = F.mse_loss(y_hat, y)
            rmse = torch.sqrt(mse)

        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_rmse", rmse, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")

    def configure_optimizers(self):
        opt_name = self.optimizer_cfg.name
        lr = self.optimizer_cfg.lr
        wd = self.optimizer_cfg.weight_decay

        if opt_name == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        else:
            raise ValueError(f"Unknown optimizer {opt_name}")

        if self.scheduler_cfg.name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "monitor": "val_rmse",
                },
            }

        return optimizer
