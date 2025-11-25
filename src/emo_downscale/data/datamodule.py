# src/emo_downscale/data/datamodule.py

from typing import Any, Dict, Optional

import numpy as np
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from dask.distributed import get_client

from emo_downscale.data.openeo_loader import load_era5_emo1_cubes_from_cache_only
from emo_downscale.data.datasets import LazyPatchDataset
from emo_downscale.logging_utils import get_logger

logger = get_logger("datamodule")


class DownscaleDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for streaming patch-based training data from cached Zarr.

    Design goals:
    - Use ONLY cached Zarr (no openEO computation paths).
    - Let Dask handle IO/parallelism, PyTorch just iterates.
    - Split by year (train/val/test) BEFORE any heavy rechunk/persist.
    - Rechunk to patch-aligned chunks per split (train/val/test).
    - Optionally persist TRAIN only, and only when the temporal extent is small
      enough, to avoid blowing up memory for long runs.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg

        training_cfg = cfg.get("training", {})
        self.batch_size: int = int(training_cfg.get("batch_size", 8))
        self.num_workers: int = int(training_cfg.get("num_workers", 0) or 0)
        self.pin_memory: bool = bool(training_cfg.get("pin_memory", True))
        self.drop_last: bool = bool(training_cfg.get("drop_last", True))

        # Internal datasets
        self._train_ds: Optional[LazyPatchDataset] = None
        self._val_ds: Optional[LazyPatchDataset] = None
        self._test_ds: Optional[LazyPatchDataset] = None

        self._is_setup: bool = False

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def prepare_data(self) -> None:
        """
        No-op: data is already prepared into cached Zarr by `prepare_zarr.py`.

        All heavy lifting (downloading, openEO processing, Zarr writing) is done
        outside of the training script. Here we only *open* and stream.
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Build LazyPatchDataset instances for train/val/test.

        This method:
        1. Loads predictors/targets from cached Zarr (cache-only path).
        2. Ensures canonical dimension order (time, bands, lat, lon).
        3. Splits by year into train/val/test.
        4. Rechunks per split (patch-aligned).
        5. Optionally persists TRAIN only (if configured and small enough).
        6. Wraps each split in LazyPatchDataset for patch extraction.
        """
        if self._is_setup:
            # Lightning may call setup() multiple times; avoid rebuilding
            logger.debug("DownscaleDataModule.setup() called again; skipping.")
            return

        data_cfg: Dict[str, Any] = self.cfg["data"]

        # ------------------------------------------------------------------
        # 1) Load from cache-only loader (no openEO path)
        # ------------------------------------------------------------------
        preds_da, targs_da = load_era5_emo1_cubes_from_cache_only(data_cfg)
        if preds_da is None or targs_da is None:
            raise RuntimeError(
                "[DataModule] Expected cached Zarr via "
                "`load_era5_emo1_cubes_from_cache_only`, but got None."
            )

        # Enforce canonical dimension order
        preds_da = preds_da.transpose("time", "bands", "lat", "lon")
        targs_da = targs_da.transpose("time", "bands", "lat", "lon")

        logger.info(
            "[DataModule] Loaded predictors/targets from cached Zarr:\n"
            f"  predictors: {preds_da.sizes}\n"
            f"  targets:    {targs_da.sizes}"
        )

        # ------------------------------------------------------------------
        # 2) Split by year BEFORE any rechunk/persist
        # ------------------------------------------------------------------
        years = preds_da["time"].dt.year.values  # np.ndarray
        split = data_cfg["split"]

        train_mask = (years >= split["train_years"][0]) & (
            years <= split["train_years"][1]
        )
        val_mask = (years >= split["val_years"][0]) & (
            years <= split["val_years"][1]
        )
        test_mask = (years >= split["test_years"][0]) & (
            years <= split["test_years"][1]
        )

        train_preds_da = preds_da.isel(time=train_mask)
        train_targs_da = targs_da.isel(time=train_mask)
        val_preds_da = preds_da.isel(time=val_mask)
        val_targs_da = targs_da.isel(time=val_mask)
        test_preds_da = preds_da.isel(time=test_mask)
        test_targs_da = targs_da.isel(time=test_mask)

        logger.info(
            "[DataModule] Time split (n_time): "
            f"train={train_preds_da.sizes['time']}, "
            f"val={val_preds_da.sizes['time']}, "
            f"test={test_preds_da.sizes['time']}"
        )

        # ------------------------------------------------------------------
        # 3) Rechunk per split with patch-aligned lat/lon
        # ------------------------------------------------------------------
        patch_cfg = data_cfg["patch"]
        patch_h = int(patch_cfg["size_y"])
        patch_w = int(patch_cfg["size_x"])

        # Allow configurable time chunking; fall back to reasonable defaults
        train_chunk_time = int(data_cfg.get("train_chunk_time", 1))
        val_chunk_time = int(data_cfg.get("val_chunk_time", max(4, train_chunk_time)))
        test_chunk_time = int(data_cfg.get("test_chunk_time", val_chunk_time))

        train_chunks = {
            "time": train_chunk_time,
            "bands": -1,
            "lat": patch_h,
            "lon": patch_w,
        }
        val_chunks = {
            "time": val_chunk_time,
            "bands": -1,
            "lat": patch_h,
            "lon": patch_w,
        }
        test_chunks = {
            "time": test_chunk_time,
            "bands": -1,
            "lat": patch_h,
            "lon": patch_w,
        }

        train_preds_da = train_preds_da.chunk(train_chunks)
        train_targs_da = train_targs_da.chunk(train_chunks)
        val_preds_da = val_preds_da.chunk(val_chunks)
        val_targs_da = val_targs_da.chunk(val_chunks)
        test_preds_da = test_preds_da.chunk(test_chunks)
        test_targs_da = test_targs_da.chunk(test_chunks)

        logger.info(
            "[DataModule] Chunking configuration:\n"
            f"  train_chunks = {train_chunks}\n"
            f"  val_chunks   = {val_chunks}\n"
            f"  test_chunks  = {test_chunks}"
        )

        # ------------------------------------------------------------------
        # 4) Optionally persist TRAIN only (for small temporal extents)
        # ------------------------------------------------------------------
        persist_train = bool(data_cfg.get("persist_train", False))
        if persist_train:
            T_train = int(train_preds_da.sizes.get("time", 0))
            persist_max_time = int(data_cfg.get("persist_max_time", 3650))

            if T_train <= persist_max_time:
                try:
                    client = get_client()
                    logger.info(
                        "[DataModule] Persisting TRAIN arrays to Dask cluster "
                        f"(T={T_train}, persist_max_time={persist_max_time})..."
                    )
                    train_preds_da, train_targs_da = client.persist(
                        [train_preds_da, train_targs_da]
                    )
                    logger.info("[DataModule] Persist of TRAIN arrays completed.")
                except Exception as e:
                    logger.warning(
                        f"[DataModule] Could not persist train arrays on Dask "
                        f"client (falling back to on-demand compute): {e}"
                    )
            else:
                logger.info(
                    "[DataModule] Skipping persist for TRAIN: "
                    f"T={T_train} > persist_max_time={persist_max_time}; "
                    "will stream directly from Zarr."
                )

        # ------------------------------------------------------------------
        # 5) Wrap into LazyPatchDataset for patch-wise streaming
        # ------------------------------------------------------------------
        stride = (int(patch_cfg["stride_y"]), int(patch_cfg["stride_x"]))
        patch_size = (patch_h, patch_w)

        self._train_ds = LazyPatchDataset(
            train_preds_da, train_targs_da, patch_size=patch_size, stride=stride
        )
        self._val_ds = LazyPatchDataset(
            val_preds_da, val_targs_da, patch_size=patch_size, stride=stride
        )
        self._test_ds = LazyPatchDataset(
            test_preds_da, test_targs_da, patch_size=patch_size, stride=stride
        )

        logger.info(
            "[DataModule] LazyPatchDataset sizes (number of patches): "
            f"train={len(self._train_ds)}, "
            f"val={len(self._val_ds)}, "
            f"test={len(self._test_ds)}"
        )

        self._is_setup = True

    # ------------------------------------------------------------------
    # Internal helper: common DataLoader kwargs
    # ------------------------------------------------------------------
    def _loader_common_kwargs(self) -> Dict[str, Any]:
        """
        Shared DataLoader kwargs.

        - num_workers=0: pure streaming via Dask in the main process (safest).
        - num_workers>0: a few workers; each will request Dask chunks, so keep
          this conservative to avoid oversubscribing CPU and RAM.
        """
        kwargs: Dict[str, Any] = dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

        if self.num_workers > 0:
            # Only meaningful when using worker processes
            training_cfg = self.cfg.get("training", {})
            prefetch_factor = int(training_cfg.get("prefetch_factor", 2))
            kwargs.update(
                dict(
                    persistent_workers=True,
                    prefetch_factor=prefetch_factor,
                )
            )

        return kwargs

    # ------------------------------------------------------------------
    # Lightning DataLoader hooks
    # ------------------------------------------------------------------
    def train_dataloader(self) -> DataLoader:
        if self._train_ds is None:
            raise RuntimeError("train_dataloader() called before setup().")
        return DataLoader(
            self._train_ds,
            shuffle=True,
            **self._loader_common_kwargs(),
        )

    def val_dataloader(self) -> DataLoader:
        if self._val_ds is None:
            raise RuntimeError("val_dataloader() called before setup().")
        return DataLoader(
            self._val_ds,
            shuffle=False,
            **self._loader_common_kwargs(),
        )

    def test_dataloader(self) -> DataLoader:
        if self._test_ds is None:
            raise RuntimeError("test_dataloader() called before setup().")
        return DataLoader(
            self._test_ds,
            shuffle=False,
            **self._loader_common_kwargs(),
        )

    def predict_dataloader(self) -> DataLoader:
        """
        Optional: use test set as predict set, or adapt as needed.
        """
        return self.test_dataloader()
