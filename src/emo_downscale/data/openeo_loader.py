# src/emo_downscale/data/openeo_loader.py

from typing import Dict, Tuple
import os
import xarray as xr
from openeo.local import LocalConnection
from dask.distributed import get_client

from emo_downscale.logging_utils import get_logger
logger = get_logger("openeo_loader")

TIME_CHUNK = 1  # keep patch-aligned

def _zarr_paths(data_cfg: Dict) -> Tuple[str, str]:
    pred_dir = data_cfg.get("predictors_feature_dir")
    targ_dir = data_cfg.get("targets_feature_dir")
    pred_name = data_cfg.get("predictors_zarr_name", "predictors.zarr")
    targ_name = data_cfg.get("targets_zarr_name", "targets.zarr")

    pred_store = os.path.join(pred_dir, pred_name) if pred_dir else None
    targ_store = os.path.join(targ_dir, targ_name) if targ_dir else None
    return pred_store, targ_store


def _open_cached_if_available(data_cfg: Dict):
    use_cached = data_cfg.get("use_cached_zarr", False)
    if not use_cached:
        return None, None

    pred_store, targ_store = _zarr_paths(data_cfg)
    if not (pred_store and targ_store):
        return None, None

    if os.path.exists(pred_store) and os.path.exists(targ_store):
        logger.info(f"Opening cached predictors from {pred_store}")
        logger.info(f"Opening cached targets    from {targ_store}")

        pred_ds = xr.open_zarr(pred_store, consolidated=True)
        targ_ds = xr.open_zarr(targ_store, consolidated=True)

        # expect single-variable datasets named "predictors"/"targets"
        preds_da = pred_ds[list(pred_ds.data_vars)[0]]
        targs_da = targ_ds[list(targ_ds.data_vars)[0]]

        logger.info("Loaded cached predictors/targets Zarr successfully.")
        return preds_da, targs_da

    return None, None

def load_era5_emo1_cubes(data_cfg: Dict) -> Tuple[xr.DataArray, xr.DataArray]:
    # 0. Try cached Zarr first
    preds_da, emo1_da = _open_cached_if_available(data_cfg)
    if preds_da is not None and emo1_da is not None:
        # Ensure correct order (time, bands, lat, lon) just in case
        preds_da = preds_da.transpose("time", "bands", "lat", "lon")
        emo1_da  = emo1_da.transpose("time", "bands", "lat", "lon")
        return preds_da, emo1_da

    # 1. Attach to global Dask cluster (like you already do)
    try:
        client = get_client()
        logger.info(f"Using existing Dask client: {client}")
    except ValueError:
        logger.warning("No active Dask client found – falling back to default scheduler.")

    spatial = data_cfg["spatial"]
    temporal = [data_cfg["temporal"]["start"], data_cfg["temporal"]["end"]]
    bands_cfg = data_cfg["bands"]
    urls = data_cfg["stac_urls"]

    patch_cfg = data_cfg["patch"]
    patch_y = patch_cfg["size_y"]
    patch_x = patch_cfg["size_x"]

    logger.info("Creating LocalConnection to openEO backend (./)...")
    conn = LocalConnection("./")

    # 2. Build openEO graph (lazy)
    logger.info("Building ERA5 / pressure / EMO1 / DEM process graph...")
    era5_single = conn.load_stac(
        url=urls["ERA5_T2M_SSRD_TP"],
        spatial_extent=spatial,
        temporal_extent=temporal,
        bands=bands_cfg["era5"],
    )
    era5_pressure = conn.load_stac(
        url=urls["ERA5_PRESSURE"],
        spatial_extent=spatial,
        temporal_extent=temporal,
        bands=bands_cfg["pressure"],
    )
    emo1 = conn.load_stac(
        url=urls["EMO1_TA24_PR_RG_PET_DAILY"],
        spatial_extent=spatial,
        temporal_extent=temporal,
        bands=bands_cfg["emo1"],
    )
    dem = conn.load_stac(
        url=urls["EMO1_DEM"],
        spatial_extent=spatial,
        bands=bands_cfg["dem"],
    )

    era5_cube = era5_single.merge_cubes(era5_pressure)
    remap = era5_cube.resample_cube_spatial(dem, method="bilinear")
    dem_expanded = dem.resample_cube_temporal(remap)
    predictors_cube = remap.merge_cubes(dem_expanded)

    logger.info("Executing process graphs to xarray (building dask graph)...")
    predictors_x = predictors_cube.execute()
    emo1_x = emo1.execute()

    # 3. Normalize to (time, bands, lat, lon)  [same as you have now]
    if isinstance(predictors_x, xr.Dataset):
        pred_var = list(predictors_x.data_vars)[0]
        predictors_da = predictors_x[pred_var]
    else:
        predictors_da = predictors_x

    if isinstance(emo1_x, xr.Dataset):
        targ_var = list(emo1_x.data_vars)[0]
        emo1_da = emo1_x[targ_var]
    else:
        emo1_da = emo1_x

    dims_pred = predictors_da.dims
    time_dim = next(d for d in dims_pred if "time" in d)
    lat_dim = next(d for d in dims_pred if d in ("lat", "y", "latitude", "ycoord"))
    lon_dim = next(d for d in dims_pred if d in ("lon", "x", "longitude", "xcoord"))
    bands_dim = next(d for d in dims_pred if d not in (time_dim, lat_dim, lon_dim))

    predictors_da = predictors_da.transpose(time_dim, bands_dim, lat_dim, lon_dim)
    emo1_da      = emo1_da.transpose(time_dim, bands_dim, lat_dim, lon_dim)

    # Dimension names after transpose
    time_dim = "time"
    bands_dim = "bands"
    lat_dim = "lat"
    lon_dim = "lon"

    # Coarse chunks for writing to Zarr (configurable, but with safe defaults)
    write_chunks = {
        time_dim: data_cfg.get("write_chunk_time", 8),
        bands_dim: -1,  # all bands together
        lat_dim: data_cfg.get("write_chunk_lat", 256),
        lon_dim: data_cfg.get("write_chunk_lon", 256),
    }

    predictors_write = predictors_da.chunk(write_chunks)
    emo1_write = emo1_da.chunk(write_chunks)

    logger.info(
        f"Using coarse chunks for Zarr write: {write_chunks}"
    )

        # 5. Optionally write to Zarr cache (using coarse chunks)
    if data_cfg.get("write_cached_zarr", False):
        pred_store, targ_store = _zarr_paths(data_cfg)
        if pred_store and targ_store:
            logger.info(f"Writing predictors Zarr to {pred_store}")
            predictors_write.to_dataset(name="predictors").to_zarr(
                pred_store,
                mode="w",
                consolidated=True,
            )

            logger.info(f"Writing targets Zarr to {targ_store}")
            emo1_write.to_dataset(name="targets").to_zarr(
                targ_store,
                mode="w",
                consolidated=True,
            )

            logger.info("Finished writing cached Zarr stores.")


    return predictors_da, emo1_da
