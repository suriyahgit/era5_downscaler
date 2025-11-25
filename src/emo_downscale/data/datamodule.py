# src/emo_downscale/data/datamodule.py

from typing import Any, Dict, Optional
import os

import lightning.pytorch as pl
from torch.utils.data import DataLoader
from dask.distributed import get_client  # still used for lazy mode
import zarr

from emo_downscale.data.openeo_loader import load_era5_emo1_cubes_from_cache_only
from emo_downscale.data.datasets import LazyPatchDataset, ArrayPatchDataset
from emo_downscale.logging_utils import get_logger

logger = get_logger("datamodule")


class DownscaleDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for ERA5→EMO1 downscaling.

    Supports two modes:

    1) PATCH-ZARR MODE (preferred if available)
       - Uses precomputed patch Zarrs written by scripts/prepare_patches.py
       - train_patches.zarr / val_patches.zarr / test_patches.zarr
       - Wrapped in ArrayPatchDataset (sample-wise training, no Dask/xarray).

    2) LAZY-ZARR MODE (fallback)
       - Uses cached year-wise Zarr (predictors_YYYY.zarr / targets_YYYY.zarr)
       - Streams patches lazily via LazyPatchDataset over xarray/dask.
       - No precomputed patch Zarrs needed.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg

        trainer_cfg = cfg.get("trainer", {})
        self.batch_size: int = int(trainer_cfg.get("batch_size", 256))
        self.num_workers: int = int(trainer_cfg.get("num_workers", 0) or 0)
        self.pin_memory: bool = bool(trainer_cfg.get("pin_memory", False))
        self.drop_last: bool = bool(trainer_cfg.get("drop_last", True))

        # Internal datasets
        self._train_ds = None
        self._val_ds = None
        self._test_ds = None

        self._is_setup: bool = False
        self._mode: str = "unknown"  # "patch_zarr" or "lazy_zarr"

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def prepare_data(self) -> None:
        """
        No-op: data is already prepared into cached Zarr (prepare_zarr.py)
        and patch Zarr (prepare_patches.py).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """
        1. Try PATCH-ZARR MODE:
           - If data.patch_dir contains train/val/test patch Zarrs,
             use those with ArrayPatchDataset.

        2. Otherwise, fallback to LAZY-ZARR MODE:
           - Use cached year-wise Zarr and LazyPatchDataset.
        """
        if self._is_setup:
            logger.debug("DownscaleDataModule.setup() called again; skipping.")
            return

        data_cfg: Dict[str, Any] = self.cfg["data"]

        # First preference: patch Zarrs, if present and allowed.
        if self._maybe_setup_from_patch_zarr(data_cfg):
            self._mode = "patch_zarr"
            self._is_setup = True
            logger.info("[DataModule] Using PATCH-ZARR MODE (ArrayPatchDataset).")
            return

        # Fallback: lazy patches over cached Zarr
        self._setup_from_lazy_zarr(data_cfg)
        self._mode = "lazy_zarr"
        self._is_setup = True
        logger.info("[DataModule] Using LAZY-ZARR MODE (LazyPatchDataset).")

    # ------------------------------------------------------------------
    # MODE 1: Precomputed patch Zarrs → ArrayPatchDataset
    # ------------------------------------------------------------------
    def _maybe_setup_from_patch_zarr(self, data_cfg: Dict[str, Any]) -> bool:
        """
        Return True if we successfully built datasets from patch Zarrs.
        Otherwise return False and let caller fall back to lazy mode.
        """
        patch_dir = data_cfg.get("patch_dir")
        use_patches_flag = bool(data_cfg.get("use_patches", True))

        if not patch_dir:
            logger.info("[DataModule] data.patch_dir not set; cannot use patch Zarrs.")
            return False

        if not use_patches_flag:
            logger.info(
                "[DataModule] data.use_patches=False → skipping patch Zarr mode."
            )
            return False

        def store_exists(name: str) -> bool:
            return os.path.isdir(os.path.join(patch_dir, name))

        expected_stores = [
            "train_patches.zarr",
            "val_patches.zarr",
            "test_patches.zarr",
        ]
        if not all(store_exists(s) for s in expected_stores):
            logger.info(
                "[DataModule] Not all patch Zarr stores found in patch_dir; "
                "falling back to lazy Zarr mode.\n"
                f"  patch_dir={patch_dir}\n"
                f"  expected={expected_stores}"
            )
            return False

        def load_split(name: str):
            path = os.path.join(patch_dir, f"{name}_patches.zarr")
            root = zarr.open_group(path, mode="r")
            X = root["X"]
            Y = root["Y"]
            logger.info(
                f"[PATCH-ZARR] {name}: store={path}, X.shape={X.shape}, Y.shape={Y.shape}"
            )
            return X, Y

        X_train, Y_train = load_split("train")
        X_val, Y_val = load_split("val")
        X_test, Y_test = load_split("test")

        self._train_ds = ArrayPatchDataset(X_train, Y_train)
        self._val_ds = ArrayPatchDataset(X_val, Y_val)
        self._test_ds = ArrayPatchDataset(X_test, Y_test)

        logger.info(
            "[DataModule] Patch Zarr datasets ready:\n"
            f"  train=N={len(self._train_ds)}\n"
            f"  val  =N={len(self._val_ds)}\n"
            f"  test =N={len(self._test_ds)}"
        )
        return True

    # ------------------------------------------------------------------
    # MODE 2: Lazy grid Zarr → LazyPatchDataset (current behavior)
    # ------------------------------------------------------------------
    def _setup_from_lazy_zarr(self, data_cfg: Dict[str, Any]) -> None:
        """
        Existing logic: use cached Zarr + LazyPatchDataset.
        """
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

        # --- Split by year ---
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

        # --- Rechunk per split (patch-aligned) ---
        patch_cfg = data_cfg["patch"]
        patch_h = int(patch_cfg["size_y"])
        patch_w = int(patch_cfg["size_x"])

        train_chunk_time = int(data_cfg.get("train_chunk_time", 1))
        val_chunk_time = int(data_cfg.get("val_chunk_time", max(4, train_chunk_time)))
        test_chunk_time = int(data_cfg.get("test_chunk_time", val_chunk_time))

        train_chunks = {
            "time": 4,
            "bands": -1,
            "lat": 128,
            "lon": 128,
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

        # --- Optional: persist TRAIN only ---
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

        # --- Wrap into LazyPatchDataset ---
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

    # ------------------------------------------------------------------
    # Internal helper: common DataLoader kwargs
    # ------------------------------------------------------------------
    def _loader_common_kwargs(self) -> Dict[str, Any]:
        """
        Shared DataLoader kwargs.

        For PATCH-ZARR MODE, feel free to set num_workers>0 in the config.
        For LAZY-ZARR MODE, num_workers>0 will spawn multiple processes that
        all talk to Dask; keep it modest.
        """
        kwargs: Dict[str, Any] = dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

        if self.num_workers > 0:
            trainer_cfg = self.cfg.get("trainer", {})
            prefetch_factor = int(trainer_cfg.get("prefetch_factor", 2))

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
        return self.test_dataloader()
