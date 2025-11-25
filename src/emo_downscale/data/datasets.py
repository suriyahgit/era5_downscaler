from typing import Tuple, List
import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
from emo_downscale.logging_utils import get_logger

logger = get_logger("datasets")


class LazyPatchDataset(Dataset):
    """
    Dataset that:
      - keeps predictors/targets as xarray DataArray (dask-backed)
      - stores only (time, y0, x0) indices
      - extracts and computes ONE patch per __getitem__
    """

    def __init__(
        self,
        preds_da: xr.DataArray,  # (time, C, lat, lon)
        targs_da: xr.DataArray,  # (time, C_out, lat, lon)
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

        indices: List[Tuple[int, int, int]] = []
        for t in range(T):
            for y0 in range(0, Y - ph + 1, sy):
                for x0 in range(0, X - pw + 1, sx):
                    indices.append((t, y0, x0))

        if not indices:
            raise ValueError(
                "No patches extracted — check patch size/stride vs domain size."
            )

        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        t, y0, x0 = self.indices[idx]
        ph, pw = self.patch_size

        # Slice ONE patch lazily; dask computes only this piece
        pred_patch = self.preds_da.isel(
            time=t,
            lat=slice(y0, y0 + ph),
            lon=slice(x0, x0 + pw),
        ).values.astype(
            self.dtype
        )  # triggers compute for this patch only

        targ_patch = self.targs_da.isel(
            time=t,
            lat=slice(y0, y0 + ph),
            lon=slice(x0, x0 + pw),
        ).values.astype(self.dtype)

        # shape: (C, ph, pw)
        x = torch.from_numpy(pred_patch)
        y = torch.from_numpy(targ_patch)
        return x, y
