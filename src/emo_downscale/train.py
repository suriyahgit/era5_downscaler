# src/emo_downscale/train.py
import logging
from emo_downscale.logging_utils import setup_global_logger, get_logger
setup_global_logger("startup")  # temporary, will be replaced in main()

import argparse
from typing import Any, Dict

import lightning.pytorch as pl
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from emo_downscale.config import load_config
from emo_downscale.data.datamodule import DownscaleDataModule
from emo_downscale.models.registry import create_model
from emo_downscale.models.module import DownscaleLightningModule
import os
from dask.distributed import Client, LocalCluster
import dask

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
    # 16 cores → 16 workers × 1 thread each
    cluster = LocalCluster(
        n_workers=16,
        threads_per_worker=1,
        memory_limit="6GB",          # 16 * 6GB ≈ 96GB < 100GB
        worker_dashboard_address=False,
        diagnostics_port=None,
    )
    client = Client(cluster)

    # Make sure everyone (including openEO) uses this cluster
    dask.config.set(scheduler="distributed")
    args = parse_args()

        # ---- Load config FIRST so run_name exists ----
    cfg: Dict[str, Any] = load_config(args.config)
    run_name = cfg.get("run_name", "downscale_run")

    # ---- Initialize project-wide logger (root + project) ----
    setup_global_logger(run_name)
    log = logging.getLogger("emo_downscale.train")
    log.debug("Logging reinitialized inside main().")
    log.info("Loaded configuration.")
    log.info(f"Using config file: {args.config}")


    try:
        pl.seed_everything(cfg.get("seed", 42), workers=True)
        log.info("Random seed set.")

        # ---- MLflow logger ----
        mlflow_cfg = cfg["mlflow"]
        mlflow_logger = MLFlowLogger(
            experiment_name=mlflow_cfg["experiment_name"],
            tracking_uri=mlflow_cfg["tracking_uri"],
            run_name=cfg["run_name"] + mlflow_cfg.get("run_name_suffix", ""),
        )
        log.info("Initialized MLflow logger.")

        # ---- DataModule ----
        datamodule = DownscaleDataModule(cfg)
        log.info("Initialized data module.")

        # ---- Model ----
        model_cfg = cfg["model"]
        backbone = create_model(model_cfg["name"], **model_cfg["params"])
        log.info("Model created.")

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
            logger=mlflow_logger,
            callbacks=[ckpt_cb, es_cb],
            gradient_clip_val=trainer_cfg["gradient_clip_val"],
            log_every_n_steps=50,
        )

        log.info("Model summary:")
        log.info(str(backbone))

        trainer.fit(lit_model, datamodule=datamodule)
        log.info("Training finished successfully.")

    except Exception:
        # Any crash bubbled up from datamodule / loader / model will be logged
        log.exception("Fatal error during training!")
        raise


if __name__ == "__main__":
    main()
