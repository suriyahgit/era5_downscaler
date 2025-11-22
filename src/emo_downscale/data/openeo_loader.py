from typing import Dict, Tuple

import os

import numpy as np
import xarray as xr
from dask.distributed import Client, LocalCluster
from openeo.local import LocalConnection

from emo_downscale.logging_utils import get_logger
logger = get_logger("openeo_loader")

# ---------------------------------------------------------------------
# Chunking constants
# ---------------------------------------------------------------------
TIME_CHUNK = 64  # fixed time-steps per chunk
# lat / lon chunks are taken from patch size (e.g. 128 x 128)


# ---------------------------------------------------------------------
# Optional Dask helper
# ---------------------------------------------------------------------
def build_local_cluster(n_workers: int = 8, threads_per_worker: int = 1) -> Client:
    """
    Optional helper to spin up a local Dask cluster before running
    the openEO process graph.
    """
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        worker_dashboard_address=False,
        diagnostics_port=None,
    )
    client = Client(cluster)
    logger.info(
        f"Started LocalCluster with {n_workers} workers, "
        f"{threads_per_worker} threads/worker."
    )
    return client


# ---------------------------------------------------------------------
# Generic feature-wise writer (time, lat, lon per feature)
# ---------------------------------------------------------------------
def _write_featurewise_simple(
    da: xr.DataArray,
    base_dir: str,
    patch_y: int,
    patch_x: int,
    role: str,
) -> None:
    """
    Save a multi-band DataArray feature-wise.

    Parameters
    ----------
    da : xr.DataArray
        DataArray with dims (time, bands, lat, lon).
    base_dir : str
        Directory where feature_* Zarrs will be written.
    patch_y : int
        Chunk size for latitude (usually patch size, e.g. 128).
    patch_x : int
        Chunk size for longitude (usually patch size, e.g. 128).
    role : str
        Label for logging (e.g. "predictors" or "targets").

    Result
    ------
    Creates:
        base_dir/feature_0.zarr
        base_dir/feature_1.zarr
        ...
    Each Zarr has dims (time, lat, lon) and chunks:
        time = TIME_CHUNK (64)
        lat  = patch_y
        lon  = patch_x
    """

    os.makedirs(base_dir, exist_ok=True)

    time_dim, bands_dim, lat_dim, lon_dim = da.dims
    n_bands = da.sizes[bands_dim]

    if "bands" in da.coords:
        band_labels = da.coords["bands"].values
    else:
        band_labels = np.arange(n_bands)

    logger.info(
        f"Writing {role} feature-wise to '{base_dir}' "
        f"with chunks (time={TIME_CHUNK}, lat={patch_y}, lon={patch_x})."
    )

    for b in range(n_bands):
        band_label = band_labels[b]

        # Select one band and drop bands dimension -> (time, lat, lon)
        band_da = (
            da.isel({bands_dim: b})
            .squeeze(bands_dim, drop=True)
            .chunk(
                {
                    time_dim: TIME_CHUNK,
                    lat_dim: patch_y,
                    lon_dim: patch_x,
                }
            )
        )

        out_path = os.path.join(base_dir, f"feature_{b}.zarr")
        logger.info(f"Saving {role} feature {b} ({band_label}) → {out_path}")

        band_da.to_zarr(out_path, mode="w", consolidated=True)


# ---------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------
def load_era5_emo1_cubes(data_cfg: Dict) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Build and execute the openEO process graph for ERA5/pressure/EMO1/DEM,
    then save predictors and targets as feature-wise Zarr stores.

    Workflow
    --------
      1. Load ERA5 single-level, ERA5 pressure, EMO1, DEM via STAC.
      2. Resample ERA5 to DEM grid, merge DEM into predictors.
      3. Execute to xarray objects (dask-backed).
      4. Normalize both predictors and targets to dims (time, bands, lat, lon).
      5. Save predictors feature-wise:
           predictors_feature_dir/feature_0.zarr, feature_1.zarr, ...
         each with dims (time, lat, lon), chunks (time=64, lat=patch_y, lon=patch_x).
      6. Save targets feature-wise:
           targets_feature_dir/feature_0.zarr, feature_1.zarr, ...
         same dims & chunking as predictors.

    Returns
    -------
    predictors_da : xr.DataArray
        Predictors DataArray with dims (time, bands, lat, lon), dask-backed.
    emo1_da : xr.DataArray
        Targets DataArray with dims (time, bands, lat, lon), dask-backed.
    """

    # ------------------------------------------------------------------
    # Read config
    # ------------------------------------------------------------------
    spatial = data_cfg["spatial"]
    temporal = [data_cfg["temporal"]["start"], data_cfg["temporal"]["end"]]
    bands_cfg = data_cfg["bands"]
    urls = data_cfg["stac_urls"]

    patch_cfg = data_cfg["patch"]
    patch_y = patch_cfg["size_y"]  # e.g., 128
    patch_x = patch_cfg["size_x"]  # e.g., 128

    predictors_feature_dir = data_cfg.get(
        "predictors_feature_dir",
        "/mnt/CEPH_PROJECTS/InterTwin/Climate_Downscaling/PAPER/v2/train_predictors_features",
    )
    targets_feature_dir = data_cfg.get(
        "targets_feature_dir",
        "/mnt/CEPH_PROJECTS/InterTwin/Climate_Downscaling/PAPER/v2/train_targets_features",
    )

    logger.info("Creating LocalConnection to openEO backend (./)...")
    conn = LocalConnection("./")  # local openEO backend

    # ------------------------------------------------------------------
    # Build process graph (lazy)
    # ------------------------------------------------------------------
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

    # Merge ERA5 fields, resample to DEM grid, bring DEM into predictors
    era5_cube = era5_single.merge_cubes(era5_pressure)
    remap = era5_cube.resample_cube_spatial(dem, method="bilinear")
    dem_expanded = dem.resample_cube_temporal(remap)
    predictors_cube = remap.merge_cubes(dem_expanded)

    # ------------------------------------------------------------------
    # Execute to xarray (dask-backed)
    # ------------------------------------------------------------------
    logger.info("Executing predictors and targets process graphs to xarray...")
    predictors_x = predictors_cube.execute()
    emo1_x = emo1.execute()

    logger.info(
        f"Predictors raw type: {type(predictors_x)}, "
        f"dims: {getattr(predictors_x, 'dims', None)}, "
        f"sizes: {getattr(predictors_x, 'sizes', None)}"
    )
    logger.info(
        f"Targets raw type: {type(emo1_x)}, "
        f"dims: {getattr(emo1_x, 'dims', None)}, "
        f"sizes: {getattr(emo1_x, 'sizes', None)}"
    )

    # ------------------------------------------------------------------
    # Normalize to DataArray with dims (time, bands, lat, lon)
    # ------------------------------------------------------------------
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

    # Infer dimension names and reorder to (time, bands, lat, lon)
    dims_pred = predictors_da.dims
    logger.info(f"Predictors dims before transpose: {dims_pred}")

    time_dim = next(d for d in dims_pred if "time" in d)
    lat_dim = next(d for d in dims_pred if d in ("lat", "y", "latitude", "ycoord"))
    lon_dim = next(d for d in dims_pred if d in ("lon", "x", "longitude", "xcoord"))
    bands_dim = next(d for d in dims_pred if d not in (time_dim, lat_dim, lon_dim))

    predictors_da = predictors_da.transpose(time_dim, bands_dim, lat_dim, lon_dim)
    emo1_da = emo1_da.transpose(time_dim, bands_dim, lat_dim, lon_dim)

    logger.info(f"Predictors dims after transpose: {predictors_da.dims}")
    logger.info(f"Targets dims after transpose:    {emo1_da.dims}")

    # Optionally chunk them in-memory too (helps future ops, but not required)
    predictors_da = predictors_da.chunk(
        {
            time_dim: TIME_CHUNK,
            bands_dim: 1,
            lat_dim: patch_y,
            lon_dim: patch_x,
        }
    )
    emo1_da = emo1_da.chunk(
        {
            time_dim: TIME_CHUNK,
            bands_dim: 1,
            lat_dim: patch_y,
            lon_dim: patch_x,
        }
    )

    # ------------------------------------------------------------------
    # Save predictors feature-wise: (time, lat, lon) per feature
    # ------------------------------------------------------------------
    _write_featurewise_simple(
        da=predictors_da,
        base_dir=predictors_feature_dir,
        patch_y=patch_y,
        patch_x=patch_x,
        role="predictors",
    )

    # ------------------------------------------------------------------
    # Save targets feature-wise: (time, lat, lon) per feature
    # ------------------------------------------------------------------
    _write_featurewise_simple(
        da=emo1_da,
        base_dir=targets_feature_dir,
        patch_y=patch_y,
        patch_x=patch_x,
        role="targets",
    )

    logger.info(
        "Finished writing predictors and targets as feature-wise Zarr stores."
    )

    # Return dask-backed DataArrays (still (time, bands, lat, lon))
    return predictors_da, emo1_da
