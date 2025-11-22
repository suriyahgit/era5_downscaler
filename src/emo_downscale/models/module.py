from typing import Dict, Any

import torch
from torch import nn
import lightning.pytorch as pl
import torch.nn.functional as F
from emo_downscale.logging_utils import get_logger
logger = get_logger("models.module")


class DownscaleLightningModule(pl.LightningModule):
    def __init__(self, model: nn.Module, optimizer_cfg: Dict[str, Any], scheduler_cfg: Dict[str, Any]):
        super().__init__()
        self.model = model
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage: str):
        # x: (B, C_in, H, W), y: (B, C_out, H, W)
        x, y = batch
        y_hat = self(x)
        loss = F.l1_loss(y_hat, y)

        with torch.no_grad():
            mse = F.mse_loss(y_hat, y)
            rmse = torch.sqrt(mse)

        # log on epoch (Lightning will aggregate over steps)
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
        opt_name = self.optimizer_cfg["name"]
        lr = self.optimizer_cfg["lr"]
        wd = self.optimizer_cfg.get("weight_decay", 0.0)

        if opt_name == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        else:
            raise ValueError(f"Unknown optimizer {opt_name}")

        sched_name = self.scheduler_cfg.get("name")

        if sched_name == "cosine":
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

        # no scheduler
        return optimizer