# scripts/prepare_patches.py

import argparse
import os
from typing import Any, Dict, Tuple

import numpy as np
import xarray as xr
import zarr

from emo_downscale.config import load_config
from emo_downscale.data.openeo_loader import load_era5_emo1_cubes_from_cache_only
from emo_downscale.logging_utils import setup_global_logger, get_logger

logger = get_logger("prepare_patches")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute patch-level Zarr datasets")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g. configs/emo1_unet.yaml)",
    )
    return parser.parse_args()


def get_year_bounds(split_cfg: Dict[str, Any], key: str) -> Tuple[int, int]:
    years = split_cfg.get(key)
    if not years or len(years) != 2:
        raise ValueError(f"data.split.{key} must be a 2-element [start_year, end_year] list.")
    return int(years[0]), int(years[1])


def subset_by_year(ds: xr.DataArray, start_year: int, end_year: int) -> xr.DataArray:
    years = ds["time"].dt.year
    mask = (years >= start_year) & (years <= end_year)
    return ds.sel(time=mask)


def compute_patch_grid(
    H: int,
    W: int,
    patch_y: int,
    patch_x: int,
    stride_y: int,
    stride_x: int,
) -> Tuple[int, int]:
    if patch_y > H or patch_x > W:
        raise ValueError(
            f"Patch size ({patch_y}, {patch_x}) larger than domain (H={H}, W={W})"
        )
    ny = 1 + (H - patch_y) // stride_y
    nx = 1 + (W - patch_x) // stride_x
    if ny <= 0 or nx <= 0:
        raise ValueError(
            "No patches can be formed with given patch_size/stride "
            f"on domain (H={H}, W={W}). Got ny={ny}, nx={nx}."
        )
    return ny, nx


def extract_patches_numpy(
    arr: np.ndarray,  # (T, C, H, W)
    patch_y: int,
    patch_x: int,
    stride_y: int,
    stride_x: int,
) -> np.ndarray:
    """
    Efficiently extract patches into a single pre-allocated array.

    arr: (T, C, H, W)
    returns: (N_patches, C, patch_y, patch_x)
    """
    T, C, H, W = arr.shape
    ny, nx = compute_patch_grid(H, W, patch_y, patch_x, stride_y, stride_x)
    patches_per_t = ny * nx
    total_patches = T * patches_per_t

    logger.info(
        f"[extract_patches] arr.shape={arr.shape}, patch=({patch_y},{patch_x}), "
        f"stride=({stride_y},{stride_x}), ny={ny}, nx={nx}, total_patches={total_patches}"
    )

    out = np.empty((total_patches, C, patch_y, patch_x), dtype=arr.dtype)

    idx = 0
    for t in range(T):
        for iy in range(ny):
            y0 = iy * stride_y
            for ix in range(nx):
                x0 = ix * stride_x
                out[idx] = arr[t, :, y0 : y0 + patch_y, x0 : x0 + patch_x]
                idx += 1

    if idx != total_patches:
        raise RuntimeError(
            f"extract_patches_numpy internal error: filled {idx} patches, expected {total_patches}"
        )

    return out


def write_patches_to_zarr(
    X: np.ndarray,
    Y: np.ndarray,
    store_path: str,
    patch_cfg: Dict[str, Any],
    split_name: str,
) -> None:
    os.makedirs(os.path.dirname(store_path), exist_ok=True)
    if os.path.exists(store_path):
        logger.info(f"[zarr] Removing existing store: {store_path}")
        import shutil

        shutil.rmtree(store_path)

    logger.info(
        f"[zarr] Writing {split_name} patches to '{store_path}'\n"
        f"  X.shape={X.shape}, Y.shape={Y.shape}"
    )

    root = zarr.open_group(store_path, mode="w")

    # chunk along samples; channels & spatial fully contiguous
    n_samples, Cx, Hy, Wx = X.shape
    _, Cy, Hy2, Wx2 = Y.shape
    assert Hy == Hy2 and Wx == Wx2

    # choose a sample-chunk ~64 or 128
    sample_chunk = min(128, n_samples)

    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.BITSHUFFLE)

    root.create_dataset(
        "X",
        data=X,
        chunks=(sample_chunk, Cx, Hy, Wx),
        compressor=compressor,
        overwrite=True,
    )
    root.create_dataset(
        "Y",
        data=Y,
        chunks=(sample_chunk, Cy, Hy, Wx),
        compressor=compressor,
        overwrite=True,
    )

    root.attrs["patch_size_y"] = int(patch_cfg["size_y"])
    root.attrs["patch_size_x"] = int(patch_cfg["size_x"])
    root.attrs["stride_y"] = int(patch_cfg["stride_y"])
    root.attrs["stride_x"] = int(patch_cfg["stride_x"])
    root.attrs["split"] = split_name

    logger.info(f"[zarr] Finished writing {split_name} patches to '{store_path}'.")


def main() -> None:
    args = parse_args()
    cfg: Dict[str, Any] = load_config(args.config)
    run_name = cfg.get("run_name", "prepare_patches")
    setup_global_logger(run_name)
    global logger
    logger = get_logger("prepare_patches")

    data_cfg = cfg["data"]
    patch_cfg = data_cfg["patch"]
    split_cfg = data_cfg["split"]

    patch_y = int(patch_cfg["size_y"])
    patch_x = int(patch_cfg["size_x"])
    stride_y = int(patch_cfg["stride_y"])
    stride_x = int(patch_cfg["stride_x"])

    patch_dir = data_cfg.get("patch_dir")
    if not patch_dir:
        raise ValueError(
            "Config must define data.patch_dir pointing to a directory for patch Zarrs."
        )

    logger.info("=== Loading cached ERA5/EMO1 from year-wise Zarr ===")
    preds_da, targs_da = load_era5_emo1_cubes_from_cache_only(data_cfg)

    # Ensure dims & order
    preds_da = preds_da.transpose("time", "bands", "lat", "lon")
    targs_da = targs_da.transpose("time", "bands", "lat", "lon")

    logger.info(
        f"[loaded] preds_da: {preds_da.sizes}, targs_da: {targs_da.sizes}, "
        f"dtype preds={preds_da.dtype}, targs={targs_da.dtype}"
    )

    # Convert to float32 to save RAM a bit
    preds_da = preds_da.astype("float32")
    targs_da = targs_da.astype("float32")

    # ---- Build splits by year ----
    train_start, train_end = get_year_bounds(split_cfg, "train_years")
    val_start, val_end = get_year_bounds(split_cfg, "val_years")
    test_start, test_end = get_year_bounds(split_cfg, "test_years")

    splits = {
        "train": (train_start, train_end),
        "val": (val_start, val_end),
        "test": (test_start, test_end),
    }

    for split_name, (ys, ye) in splits.items():
        logger.info(f"=== Building {split_name} split for years [{ys}, {ye}] ===")

        preds_split = subset_by_year(preds_da, ys, ye)
        targs_split = subset_by_year(targs_da, ys, ye)

        logger.info(
            f"[{split_name}] after year subset: preds={preds_split.sizes}, "
            f"targs={targs_split.sizes}"
        )

        # Load into memory
        preds_np = preds_split.values  # (T, Cx, H, W)
        targs_np = targs_split.values  # (T, Cy, H, W)

        T, Cx, H, W = preds_np.shape
        _, Cy, H2, W2 = targs_np.shape
        assert H == H2 and W == W2

        logger.info(f"[{split_name}] loaded into memory: preds_np.shape={preds_np.shape}, "
                    f"targs_np.shape={targs_np.shape}")

        # Extract patches
        X_patches = extract_patches_numpy(
            preds_np, patch_y=patch_y, patch_x=patch_x, stride_y=stride_y, stride_x=stride_x
        )
        Y_patches = extract_patches_numpy(
            targs_np, patch_y=patch_y, patch_x=patch_x, stride_y=stride_y, stride_x=stride_x
        )

        logger.info(
            f"[{split_name}] patches: X.shape={X_patches.shape}, Y.shape={Y_patches.shape}"
        )

        # Write to Zarr
        store_path = os.path.join(patch_dir, f"{split_name}_patches.zarr")
        write_patches_to_zarr(
            X=X_patches,
            Y=Y_patches,
            store_path=store_path,
            patch_cfg=patch_cfg,
            split_name=split_name,
        )

    logger.info("All splits done. Patch Zarrs ready for training.")


if __name__ == "__main__":
    main()
