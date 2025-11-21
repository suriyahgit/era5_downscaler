import argparse
from typing import Any, Dict

import lightning.pytorch as pl
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import torch

from emo_downscale.config import load_config
from emo_downscale.data.datamodule import DownscaleDataModule
from emo_downscale.models.registry import create_model
from emo_downscale.models.module import DownscaleLightningModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ERA5 → EMO1 downscaling trainer")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g. configs/emo1_unet.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg: Dict[str, Any] = load_config(args.config)

    pl.seed_everything(cfg.get("seed", 42), workers=True)

    # ---- MLflow logger ----
    mlflow_cfg = cfg["mlflow"]
    logger = MLFlowLogger(
        experiment_name=mlflow_cfg["experiment_name"],
        tracking_uri=mlflow_cfg["tracking_uri"],
        run_name=cfg["run_name"] + mlflow_cfg.get("run_name_suffix", ""),
    )

    # ---- DataModule ----
    datamodule = DownscaleDataModule(cfg)

    # ---- Model ----
    model_cfg = cfg["model"]
    backbone = create_model(model_cfg["name"], **model_cfg["params"])

    lit_model = DownscaleLightningModule(
        model=backbone,
        optimizer_cfg=cfg["trainer"]["optimizer"],
        scheduler_cfg=cfg["trainer"]["scheduler"],
    )

    # ---- Callbacks ----
    trainer_cfg = cfg["trainer"]
    ckpt_cb = ModelCheckpoint(
        monitor=trainer_cfg["checkpoint"]["monitor"],
        mode=trainer_cfg["checkpoint"]["mode"],
        save_top_k=1,
        filename="{epoch:02d}-{val_rmse:.4f}",
    )

    es_cb = EarlyStopping(
        monitor=trainer_cfg["early_stopping"]["monitor"],
        patience=trainer_cfg["early_stopping"]["patience"],
        mode=trainer_cfg["early_stopping"]["mode"],
    )

    trainer = pl.Trainer(
        max_epochs=trainer_cfg["max_epochs"],
        accelerator=trainer_cfg["accelerator"],
        devices=trainer_cfg["devices"],
        precision=trainer_cfg["precision"],
        logger=logger,
        callbacks=[ckpt_cb, es_cb],
        gradient_clip_val=trainer_cfg["gradient_clip_val"],
        log_every_n_steps=50,
    )

    trainer.fit(lit_model, datamodule=datamodule)


if __name__ == "__main__":
    main()
