# src/emo_downscale/data/datasets.py

from typing import Tuple
import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
from emo_downscale.logging_utils import get_logger

logger = get_logger("datasets")


class LazyPatchDataset(Dataset):
    """
    Highly optimised Dask-backed patch dataset.

    - Keeps predictors/targets as xarray.DataArray (dask-backed).
    - Does NOT store a huge list of indices; uses pure index arithmetic.
    - Extracts exactly ONE patch per __getitem__, triggering Dask compute
      only for that patch's chunks.
    - Designed to handle long temporal extents efficiently.

    Assumes:
      preds_da, targs_da dims: ("time", "bands", "lat", "lon")
      patch_size: (patch_height, patch_width)
      stride:     (stride_y, stride_x)
    """

    def __init__(
        self,
        preds_da: xr.DataArray,   # (time, bands, lat, lon)
        targs_da: xr.DataArray,   # (time, bands, lat, lon)
        patch_size: Tuple[int, int],
        stride: Tuple[int, int],
        dtype: str = "float32",
    ):
        assert preds_da.dims == ("time", "bands", "lat", "lon"), (
            f"preds_da dims must be ('time','bands','lat','lon'), got {preds_da.dims}"
        )
        assert targs_da.dims == ("time", "bands", "lat", "lon"), (
            f"targs_da dims must be ('time','bands','lat','lon'), got {targs_da.dims}"
        )

        self.preds_da = preds_da
        self.targs_da = targs_da
        self.patch_size = patch_size
        self.stride = stride
        self.dtype = dtype

        # Use xarray sizes (cheap, metadata only; fine for Dask)
        T = int(preds_da.sizes["time"])
        C = int(preds_da.sizes["bands"])
        H = int(preds_da.sizes["lat"])
        W = int(preds_da.sizes["lon"])

        ph, pw = patch_size
        sy, sx = stride

        if ph > H or pw > W:
            raise ValueError(
                f"Patch size {patch_size} is larger than domain "
                f"(H={H}, W={W})."
            )

        # Number of patch positions along each spatial axis
        ny = 1 + (H - ph) // sy
        nx = 1 + (W - pw) // sx
        if ny <= 0 or nx <= 0:
            raise ValueError(
                "No patches can be formed with given patch_size/stride "
                f"on domain (H={H}, W={W}). Got ny={ny}, nx={nx}."
            )

        self.T = T
        self.C = C
        self.H = H
        self.W = W
        self.ny = ny
        self.nx = nx
        self.n_patches = T * ny * nx

        logger.info(
            "[LazyPatchDataset] Init:\n"
            f"  time={T}, bands={C}, H={H}, W={W}\n"
            f"  patch_size={patch_size}, stride={stride}\n"
            f"  grid: ny={ny}, nx={nx}, total_patches={self.n_patches}"
        )

    # ------------------------------------------------------------------ #
    # Index arithmetic: map [0, n_patches) -> (t, y0, x0)
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return self.n_patches

    def _decode_index(self, idx: int) -> Tuple[int, int, int]:
        """
        Decode a global patch index into (t, y0, x0).

        Layout:
          - fastest axis: x (lon)
          - then y (lat)
          - then time
        """
        if idx < 0 or idx >= self.n_patches:
            raise IndexError(f"Index {idx} out of range [0, {self.n_patches}).")

        # patches per time slice
        patches_per_t = self.ny * self.nx

        t = idx // patches_per_t
        rem = idx % patches_per_t
        iy = rem // self.nx
        ix = rem % self.nx

        ph, pw = self.patch_size
        sy, sx = self.stride

        y0 = iy * sy
        x0 = ix * sx

        return int(t), int(y0), int(x0)

    # ------------------------------------------------------------------ #
    # Patch extraction
    # ------------------------------------------------------------------ #
    def __getitem__(self, idx: int):
        t, y0, x0 = self._decode_index(idx)
        ph, pw = self.patch_size

        # Lazy slice — still a Dask-backed DataArray, no compute yet
        pred_da = self.preds_da.isel(
            time=t,
            lat=slice(y0, y0 + ph),
            lon=slice(x0, x0 + pw),
        )
        targ_da = self.targs_da.isel(
            time=t,
            lat=slice(y0, y0 + ph),
            lon=slice(x0, x0 + pw),
        )

        # Convert to NumPy; .values triggers Dask compute just for this slice
        # and returns a numpy.ndarray with shape (C, ph, pw).
        pred_np = np.asarray(pred_da.values, dtype=self.dtype)
        targ_np = np.asarray(targ_da.values, dtype=self.dtype)

        # Extra safety: catch unexpected shapes
        if pred_np.ndim != 3 or targ_np.ndim != 3:
            logger.error(
                "[LazyPatchDataset] Unexpected patch shape at idx=%d "
                "(t=%d, y0=%d, x0=%d): pred_np.shape=%s, targ_np.shape=%s",
                idx,
                t,
                y0,
                x0,
                pred_np.shape,
                targ_np.shape,
            )
            raise RuntimeError("LazyPatchDataset: patch is not 3D (C, H, W).")

        # Zero-copy into torch if dtype already matches
        x = torch.from_numpy(pred_np)  # (C, ph, pw)
        y = torch.from_numpy(targ_np)

        return x, y
