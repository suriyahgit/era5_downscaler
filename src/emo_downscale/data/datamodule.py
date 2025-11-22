from typing import Any, Dict, Optional

import numpy as np
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from emo_downscale.data.openeo_loader import load_era5_emo1_cubes
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

        self._train_ds = None
        self._val_ds = None
        self._test_ds = None

    def setup(self, stage: Optional[str] = None):
        data_cfg = self.cfg["data"]

        # 1. load xarray cubes via openeo-processes-dask
        predictors_ds, target_ds = load_era5_emo1_cubes(data_cfg)
        logger.info("Dask Graph Computed Successfully!")

        # standardize to DataArray: (time, C, Y, X)
        preds_da = predictors_ds.transpose("time", "bands", "lat", "lon")
        targs_da = target_ds.transpose("time", "bands", "lat", "lon")
    
        years = predictors_ds["time"].dt.year.values
        split = data_cfg["split"]
    
        train_mask = (years >= split["train_years"][0]) & (years <= split["train_years"][1])
        val_mask   = (years >= split["val_years"][0])   & (years <= split["val_years"][1])
        test_mask  = (years >= split["test_years"][0])  & (years <= split["test_years"][1])
    
        train_preds_da = preds_da.isel(time=train_mask)
        train_targs_da = targs_da.isel(time=train_mask)
        val_preds_da   = preds_da.isel(time=val_mask)
        val_targs_da   = targs_da.isel(time=val_mask)
        test_preds_da  = preds_da.isel(time=test_mask)
        test_targs_da  = targs_da.isel(time=test_mask)
    
        patch_cfg = data_cfg["patch"]
        patch_size = (patch_cfg["size_y"], patch_cfg["size_x"])
        stride     = (patch_cfg["stride_y"], patch_cfg["stride_x"])
    
        self._train_ds = LazyPatchDataset(train_preds_da, train_targs_da, patch_size, stride)
        self._val_ds   = LazyPatchDataset(val_preds_da,   val_targs_da,   patch_size, stride)
        self._test_ds  = LazyPatchDataset(test_preds_da,  test_targs_da,  patch_size, stride)

    def train_dataloader(self):
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self._test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
