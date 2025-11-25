# src/emo_downscale/data/datasets.py

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr

from emo_downscale.logging_utils import get_logger

logger = get_logger("data.datasets")


# ---------------------------------------------------------------------
# 1) ArrayPatchDataset  (for fully in-memory numpy arrays: X, Y)
# ---------------------------------------------------------------------
class ArrayPatchDataset(Dataset):
    """
    Simple dataset for pre-extracted patch tensors stored as numpy arrays.

    X: (N, Cx, H, W)
    Y: (N, Cy, H, W)
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"ArrayPatchDataset: X and Y must have same first dim. "
                f"Got X.shape={X.shape}, Y.shape={Y.shape}"
            )
        self.X = X
        self.Y = Y

        logger.info(
            "ArrayPatchDataset initialized: "
            f"N={self.X.shape[0]}, "
            f"X.shape={self.X.shape}, Y.shape={self.Y.shape}"
        )

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx]).float()
        y = torch.from_numpy(self.Y[idx]).float()
        return x, y


# ---------------------------------------------------------------------
# 2) LazyPatchDataset  (for Dask/xarray-backed cubes: time, bands, lat, lon)
# ---------------------------------------------------------------------
class LazyPatchDataset(Dataset):
    """
    Patch-wise dataset that lazily extracts patches from xarray.DataArray
    objects backed by Dask / Zarr.

    predictors_da: (time, bands, lat, lon)
    targets_da:    (time, bands, lat, lon)

    We never materialize all patches at once; each __getitem__ computes
    exactly one patch via xarray indexing → dask → numpy → torch.
    """

    def __init__(
        self,
        predictors_da: xr.DataArray,
        targets_da: xr.DataArray,
        patch_size: Tuple[int, int],
        stride: Tuple[int, int],
    ):
        # Basic checks
        if predictors_da.dims != ("time", "bands", "lat", "lon"):
            raise ValueError(
                f"LazyPatchDataset expects predictors dims ('time','bands','lat','lon'), "
                f"got {predictors_da.dims}"
            )
        if targets_da.dims != ("time", "bands", "lat", "lon"):
            raise ValueError(
                f"LazyPatchDataset expects targets dims ('time','bands','lat','lon'), "
                f"got {targets_da.dims}"
            )

        if predictors_da.sizes["time"] != targets_da.sizes["time"]:
            raise ValueError("Predictors and targets must have same time length.")
        if predictors_da.sizes["lat"] != targets_da.sizes["lat"]:
            raise ValueError("Predictors and targets must have same lat size.")
        if predictors_da.sizes["lon"] != targets_da.sizes["lon"]:
            raise ValueError("Predictors and targets must have same lon size.")

        self.predictors = predictors_da
        self.targets = targets_da

        self.patch_h, self.patch_w = map(int, patch_size)
        self.stride_y, self.stride_x = map(int, stride)

        self.T = int(predictors_da.sizes["time"])
        self.Cx = int(predictors_da.sizes["bands"])
        self.Cy = int(targets_da.sizes["bands"])
        self.H = int(predictors_da.sizes["lat"])
        self.W = int(predictors_da.sizes["lon"])

        if self.patch_h > self.H or self.patch_w > self.W:
            raise ValueError(
                f"Patch size ({self.patch_h}, {self.patch_w}) larger than domain "
                f"(H={self.H}, W={self.W})."
            )

        # Compute patch grid per time slice
        self.ny = 1 + (self.H - self.patch_h) // self.stride_y
        self.nx = 1 + (self.W - self.patch_w) // self.stride_x

        if self.ny <= 0 or self.nx <= 0:
            raise ValueError(
                "No patches can be formed with given patch_size/stride "
                f"on domain (H={self.H}, W={self.W}). Got ny={self.ny}, nx={self.nx}."
            )

        self.patches_per_t = self.ny * self.nx
        self.N = self.T * self.patches_per_t

        logger.info(
            "LazyPatchDataset initialized:\n"
            f"  predictors: time={self.T}, bands={self.Cx}, H={self.H}, W={self.W}\n"
            f"  targets:    time={self.T}, bands={self.Cy}, H={self.H}, W={self.W}\n"
            f"  patch_size=({self.patch_h},{self.patch_w}), "
            f"stride=({self.stride_y},{self.stride_x})\n"
            f"  grid: ny={self.ny}, nx={self.nx}, patches_per_t={self.patches_per_t}\n"
            f"  total patches N={self.N}"
        )

    def __len__(self) -> int:
        return self.N

    def _index_to_coords(self, idx: int):
        """
        Map a flat index [0, N) to (t, y0, x0) in the original grid.
        """
        if idx < 0 or idx >= self.N:
            raise IndexError(f"Index {idx} out of range [0, {self.N}).")

        t = idx // self.patches_per_t
        rem = idx % self.patches_per_t
        iy = rem // self.nx
        ix = rem % self.nx

        y0 = iy * self.stride_y
        x0 = ix * self.stride_x

        return t, y0, x0

    def __getitem__(self, idx: int):
        t, y0, x0 = self._index_to_coords(idx)

        y1 = y0 + self.patch_h
        x1 = x0 + self.patch_w

        # xarray indexing: still Dask-backed, compute only this slice
        x_da = self.predictors.isel(
            time=t,
            bands=slice(None),
            lat=slice(y0, y1),
            lon=slice(x0, x1),
        )
        y_da = self.targets.isel(
            time=t,
            bands=slice(None),
            lat=slice(y0, y1),
            lon=slice(x0, x1),
        )

        # Compute to numpy (patch-sized only), then to torch tensors
        x_np = np.asarray(x_da.data, dtype=np.float32)
        y_np = np.asarray(y_da.data, dtype=np.float32)

        # Ensure shape (C, H, W)
        # x_da shape is (bands, lat, lon) after isel on time
        if x_np.ndim != 3:
            raise RuntimeError(f"Expected x_np.ndim==3, got {x_np.ndim}")
        if y_np.ndim != 3:
            raise RuntimeError(f"Expected y_np.ndim==3, got {y_np.ndim}")

        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_np)

        return x, y
