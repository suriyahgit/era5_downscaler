from typing import Any, Dict, Optional

import numpy as np
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from emo_downscale.data.openeo_loader import load_era5_emo1_cubes_from_cache_only
from emo_downscale.data.datasets import LazyPatchDataset
from emo_downscale.logging_utils import get_logger

logger = get_logger("datamodule")


class DownscaleDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        tcfg = cfg["trainer"]
        self.batch_size = tcfg["batch_size"]
        self.num_workers = tcfg["num_workers"]
        # NEW: optional prefetch_factor under trainer; default to 2 if missing
        self.prefetch_factor = tcfg.get("prefetch_factor", 2)

        self._train_ds = None
        self._val_ds = None
        self._test_ds = None

    def setup(self, stage: Optional[str] = None):
        logger.debug("=== ENTER setup() ===")
        data_cfg = self.cfg["data"]
        logger.debug(f"Data config: {data_cfg}")

        # 1. STRICT: load from cached Zarr only (no openEO fallback)
        preds_da, targs_da = load_era5_emo1_cubes_from_cache_only(data_cfg)
        logger.info("Loaded predictors/targets from cached Zarr (cache-only mode).")

        # preds_da / targs_da are already (time, bands, lat, lon); transpose is cheap/no-op
        preds_da = preds_da.transpose("time", "bands", "lat", "lon")
        targs_da = targs_da.transpose("time", "bands", "lat", "lon")


        # === Rechunk for training: patch-aligned + time=1 ===
        patch_cfg = data_cfg["patch"]
        patch_h = patch_cfg["size_y"]
        patch_w = patch_cfg["size_x"]

        train_chunks = {
            "time": data_cfg.get("train_chunk_time", 1),
            "bands": -1,
            "lat": patch_h,
            "lon": patch_w,
        }

        preds_da = preds_da.chunk(train_chunks)
        targs_da = targs_da.chunk(train_chunks)
        logger.info(f"Rechunked predictors/targets for training: {train_chunks}")

        years = preds_da["time"].dt.year.values

        split = data_cfg["split"]

        train_mask = (years >= split["train_years"][0]) & (
            years <= split["train_years"][1]
        )
        val_mask = (years >= split["val_years"][0]) & (years <= split["val_years"][1])
        test_mask = (years >= split["test_years"][0]) & (
            years <= split["test_years"][1]
        )

        train_preds_da = preds_da.isel(time=train_mask)
        train_targs_da = targs_da.isel(time=train_mask)
        val_preds_da = preds_da.isel(time=val_mask)
        val_targs_da = targs_da.isel(time=val_mask)
        test_preds_da = preds_da.isel(time=test_mask)
        test_targs_da = targs_da.isel(time=test_mask)

        patch_cfg = data_cfg["patch"]
        patch_size = (patch_cfg["size_y"], patch_cfg["size_x"])
        stride = (patch_cfg["stride_y"], patch_cfg["stride_x"])

        self._train_ds = LazyPatchDataset(
            train_preds_da, train_targs_da, patch_size, stride
        )
        self._val_ds = LazyPatchDataset(val_preds_da, val_targs_da, patch_size, stride)
        self._test_ds = LazyPatchDataset(
            test_preds_da, test_targs_da, patch_size, stride
        )

        logger.info(
            f"Datasets built: "
            f"train={len(self._train_ds)}, val={len(self._val_ds)}, test={len(self._test_ds)} patches."
        )

    def _loader_common_kwargs(self) -> Dict[str, Any]:
        """Common DataLoader kwargs with safe handling for num_workers=0."""
        kwargs: Dict[str, Any] = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": True,
        }
        # prefetch_factor & persistent_workers only make sense with workers > 0
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
            kwargs["persistent_workers"] = True
        return kwargs

    def train_dataloader(self):
        return DataLoader(
            self._train_ds,
            shuffle=True,
            **self._loader_common_kwargs(),
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_ds,
            shuffle=False,
            **self._loader_common_kwargs(),
        )

    def test_dataloader(self):
        return DataLoader(
            self._test_ds,
            shuffle=False,
            **self._loader_common_kwargs(),
        )
