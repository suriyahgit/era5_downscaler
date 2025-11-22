# src/emo_downscale/data/openeo_loader.py

from typing import Dict, Tuple
import xarray as xr
from openeo.local import LocalConnection
import dask
from dask.distributed import get_client

from emo_downscale.logging_utils import get_logger
logger = get_logger("openeo_loader")

TIME_CHUNK = 64  # time chunk (you can tune if needed)

def load_era5_emo1_cubes(data_cfg: Dict) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Load ERA5/pressure/EMO1/DEM via STAC (data on S3), build the openEO process graph,
    and return *dask-backed* DataArrays with dims (time, bands, lat, lon).

    Nothing is fully loaded into memory here: we only build the graph and define chunks.
    Actual S3 reads happen later, when the PyTorch DataLoader asks for patches.
    """

    # Ensure we attach to the global Dask cluster created in train.py
    try:
        client = get_client()
        logger.info(f"Using existing Dask client: {client}")
    except ValueError:
        logger.warning("No active Dask client found – falling back to default scheduler.")




    # ------------------------------------------------------------------
    # Read config
    # ------------------------------------------------------------------
    spatial = data_cfg["spatial"]
    temporal = [data_cfg["temporal"]["start"], data_cfg["temporal"]["end"]]
    bands_cfg = data_cfg["bands"]
    urls = data_cfg["stac_urls"]

    logger.debug("=== ENTER load_era5_emo1_cubes ===")
    logger.debug(f"Spatial: {spatial}")
    logger.debug(f"Temporal: {temporal}")
    logger.debug(f"Bands: {bands_cfg}")

    patch_cfg = data_cfg["patch"]
    patch_y = patch_cfg["size_y"]  # e.g., 128
    patch_x = patch_cfg["size_x"]  # e.g., 128

    logger.info("Creating LocalConnection to openEO backend (./)...")
    conn = LocalConnection("./")

    # ------------------------------------------------------------------
    # Build openEO process graph (still lazy)
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
    # Execute to xarray – this still returns *dask-backed* objects
    # ------------------------------------------------------------------
    logger.info("Executing process graphs to xarray (building dask graph)...")
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

    # ------------------------------------------------------------------
    # Set chunking aligned to patches (and moderate time)
    # ------------------------------------------------------------------
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

    logger.info("Finished building dask-backed predictors and targets (lazy).")
    return predictors_da, emo1_da
