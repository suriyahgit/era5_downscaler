# src/emo_downscale/data/datamodule.py

from typing import Any, Dict, Optional

import numpy as np
import zarr
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from emo_downscale.data.datasets import ArrayPatchDataset
from emo_downscale.logging_utils import get_logger

logger = get_logger("datamodule")


def _load_zarr_patches(store_path: str):
    logger.info(f"[DataModule] Opening patch Zarr store: {store_path}")
    root = zarr.open_group(store_path, mode="r")

    X_z = root["X"]
    Y_z = root["Y"]

    logger.info(
        f"[DataModule] Found datasets: X.shape={X_z.shape}, Y.shape={Y_z.shape}, "
        f"X.chunks={X_z.chunks}, Y.chunks={Y_z.chunks}"
    )

    # Load fully into memory (you have ~100 GB; this is ~10–12 GB for train)
    X = np.asarray(X_z, dtype=np.float32)
    Y = np.asarray(Y_z, dtype=np.float32)

    logger.info(
        f"[DataModule] Loaded into memory: X.shape={X.shape}, Y.shape={Y.shape}, "
        f"dtype={X.dtype}"
    )

    return X, Y


class DownscaleDataModule(pl.LightningDataModule):
    """
    Lightning DataModule using *precomputed patch Zarrs*.

    Workflow:
      1. Run scripts/prepare_patches.py once to create:
         - <patch_dir>/train_patches.zarr
         - <patch_dir>/val_patches.zarr
         - <patch_dir>/test_patches.zarr

      2. This DataModule:
         - loads each split's X/Y into memory
         - wraps them with ArrayPatchDataset
         - returns standard PyTorch DataLoaders
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg

        trainer_cfg = cfg.get("trainer", {})
        self.batch_size: int = int(trainer_cfg.get("batch_size", 8))
        self.num_workers: int = int(trainer_cfg.get("num_workers", 4))
        self.pin_memory: bool = True
        self.drop_last: bool = True

        data_cfg = cfg["data"]
        self.patch_dir: str = data_cfg.get("patch_dir", "")
        if not self.patch_dir:
            raise ValueError(
                "Config must define data.patch_dir pointing to the patch Zarr directory."
            )

        self._train_ds: Optional[ArrayPatchDataset] = None
        self._val_ds: Optional[ArrayPatchDataset] = None
        self._test_ds: Optional[ArrayPatchDataset] = None

    # ------------------------------------------------------------------ #
    # Lightning hooks
    # ------------------------------------------------------------------ #
    def prepare_data(self) -> None:
        """
        No-op: patch Zarrs must already exist (prepared by scripts/prepare_patches.py).
        """
        logger.info("[DataModule] prepare_data(): expecting patch Zarrs to already exist.")

    def setup(self, stage: Optional[str] = None) -> None:
        if self._train_ds is not None and self._val_ds is not None and self._test_ds is not None:
            return  # already set up

        logger.info("[DataModule] setup(stage=%s)", stage)

        train_store = f"{self.patch_dir}/train_patches.zarr"
        val_store = f"{self.patch_dir}/val_patches.zarr"
        test_store = f"{self.patch_dir}/test_patches.zarr"

        X_train, Y_train = _load_zarr_patches(train_store)
        X_val, Y_val = _load_zarr_patches(val_store)
        X_test, Y_test = _load_zarr_patches(test_store)

        self._train_ds = ArrayPatchDataset(X_train, Y_train)
        self._val_ds = ArrayPatchDataset(X_val, Y_val)
        self._test_ds = ArrayPatchDataset(X_test, Y_test)

        logger.info(
            "[DataModule] Dataset sizes (patches): train=%d, val=%d, test=%d",
            len(self._train_ds),
            len(self._val_ds),
            len(self._test_ds),
        )

    # ------------------------------------------------------------------ #
    # DataLoaders
    # ------------------------------------------------------------------ #
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
