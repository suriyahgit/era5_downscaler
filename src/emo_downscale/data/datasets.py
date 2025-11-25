from typing import Tuple, List
import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
from emo_downscale.logging_utils import get_logger

logger = get_logger("datasets")


class LazyPatchDataset(Dataset):
    """
    Efficient Dask-backed patch extractor:

    - No global preload
    - One patch computed per __getitem__
    - Zero-cost indexing (math-based, no huge index lists)
    - No .values() (safer for Dask)
    """

    def __init__(
        self,
        preds_da: xr.DataArray,   # (time, bands, lat, lon)
        targs_da: xr.DataArray,   # (time, bands, lat, lon)
        patch_size: Tuple[int, int],
        stride: Tuple[int, int],
        dtype: str = "float32",
    ):
        assert preds_da.dims == ("time", "bands", "lat", "lon")
        assert targs_da.dims == ("time", "bands", "lat", "lon")

        self.preds_da = preds_da
        self.targs_da = targs_da
        self.patch_size = patch_size
        self.stride = stride
        self.dtype = dtype

        T, C, Y, X = preds_da.shape
        ph, pw = patch_size
        sy, sx = stride

        self.T = T
        self.Y = Y
        self.X = X
        self.ph = ph
        self.pw = pw
        self.sy = sy
        self.sx = sx

        # compute grid sizes mathematically
        self.ny = 1 + (Y - ph) // sy
        self.nx = 1 + (X - pw) // sx
        self.samples_per_t = self.ny * self.nx
        self.total_samples = T * self.samples_per_t

    def __len__(self):
        return self.total_samples

    def _decode_index(self, idx):
        """Convert linear idx → (t, y0, x0). No list stored."""
        t = idx // self.samples_per_t
        rem = idx % self.samples_per_t

        y_idx = rem // self.nx
        x_idx = rem % self.nx

        y0 = y_idx * self.sy
        x0 = x_idx * self.sx
        return t, y0, x0

    def __getitem__(self, idx):
        t, y0, x0 = self._decode_index(idx)
        ph, pw = self.patch_size

        # lazy slice
        pred_arr = self.preds_da.isel(
            time=t,
            lat=slice(y0, y0 + ph),
            lon=slice(x0, x0 + pw),
        )

        targ_arr = self.targs_da.isel(
            time=t,
            lat=slice(y0, y0 + ph),
            lon=slice(x0, x0 + pw),
        )

        # compute only this patch
        x = torch.as_tensor(pred_arr.compute(), dtype=torch.float32)
        y = torch.as_tensor(targ_arr.compute(), dtype=torch.float32)

        return x, y
