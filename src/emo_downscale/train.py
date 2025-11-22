# src/emo_downscale/train.py

import argparse
from typing import Any, Dict
import os
import logging
from datetime import datetime
import traceback

import lightning.pytorch as pl
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import torch

from emo_downscale.config import load_config
from emo_downscale.data.datamodule import DownscaleDataModule
from emo_downscale.models.registry import create_model
from emo_downscale.models.module import DownscaleLightningModule


def setup_logging(run_name: str) -> logging.Logger:
    os.makedirs("logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("logs", f"{run_name}_{ts}.log")

    # Root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode="w"),
        ],
        force=True,  # override any previous basicConfig
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_path}")
    return logger


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

    # Load config first (to get run_name)
    cfg: Dict[str, Any] = load_config(args.config)
    run_name = cfg.get("run_name", "downscale_run")

    logger1 = setup_logging(run_name)
    logger1.info("Loaded configuration.")
    logger1.info(f"Using config file: {args.config}")

    try:
        pl.seed_everything(cfg.get("seed", 42), workers=True)
        logger1.info("Random seed set.")

        # ---- MLflow logger ----
        mlflow_cfg = cfg["mlflow"]
        logger = MLFlowLogger(
            experiment_name=mlflow_cfg["experiment_name"],
            tracking_uri=mlflow_cfg["tracking_uri"],
            run_name=cfg["run_name"] + mlflow_cfg.get("run_name_suffix", ""),
        )
        logger1.info("Initialized MLflow logger.")

        # ---- DataModule ----
        datamodule = DownscaleDataModule(cfg)
        logger1.info("Initialized data module.")

        # ---- Model ----
        model_cfg = cfg["model"]
        backbone = create_model(model_cfg["name"], **model_cfg["params"])
        logger1.info("Model created.")

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

        logger1.info("Model summary:")
        logger1.info(str(backbone))

        trainer.fit(lit_model, datamodule=datamodule)
        logger1.info("Training finished successfully.")

    except Exception as e:
        # Make sure crashes are logged
        logger1.error("Fatal error during training!", exc_info=True)
        # Optional: also log traceback as string for readability
        tb = traceback.format_exc()
        logger1.error(f"Traceback:\n{tb}")
        raise  # keep non-zero exit code


if __name__ == "__main__":
    main()
