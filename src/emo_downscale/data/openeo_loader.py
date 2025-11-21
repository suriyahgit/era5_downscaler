from typing import Dict, Tuple

from openeo.local import LocalConnection
from dask.distributed import LocalCluster, Client
import xarray as xr
import logging
logger = logging.getLogger(__name__)



def build_local_cluster(n_workers: int = 8, threads_per_worker: int = 1) -> Client:
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        worker_dashboard_address=False,
        diagnostics_port=None,
    )
    client = Client(cluster)
    return client


def load_era5_emo1_cubes(data_cfg: Dict) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Use openeo-processes-dask to:
      - load ERA5, pressure, EMO1, DEM via STAC
      - resample ERA5 to EMO1 grid
      - merge DEM to predictors
    Returns:
      predictors_cube (xarray.Dataset)
      target_cube (xarray.Dataset)  # EMO1
    """

    spatial = data_cfg["spatial"]
    temporal = [data_cfg["temporal"]["start"], data_cfg["temporal"]["end"]]
    bands = data_cfg["bands"]
    urls = data_cfg["stac_urls"]

    conn = LocalConnection("./")  # your existing local backend

    era5_single = conn.load_stac(
        url=urls["ERA5_T2M_SSRD_TP"],
        spatial_extent=spatial,
        temporal_extent=temporal,
        bands=bands["era5"],
    )

    era5_pressure = conn.load_stac(
        url=urls["ERA5_PRESSURE"],
        spatial_extent=spatial,
        temporal_extent=temporal,
        bands=bands["pressure"],
    )

    emo1 = conn.load_stac(
        url=urls["EMO1_TA24_PR_RG_PET_DAILY"],
        spatial_extent=spatial,
        temporal_extent=temporal,
        bands=bands["emo1"],
    )

    dem = conn.load_stac(
        url=urls["EMO1_DEM"],
        spatial_extent=spatial,
        bands=bands["dem"],
    )

    era5_cube = era5_single.merge_cubes(era5_pressure)
    remap = era5_cube.resample_cube_spatial(dem, method="bilinear")
    dem_expanded = dem.resample_cube_temporal(remap)
    predictors_cube = remap.merge_cubes(dem_expanded)
    predictors_cube = predictors_cube.execute()
    emo1 = emo1.execute()

    # At this point predictors_cube & emo1 should share time/y/x grid
    #predictors_cube, emo1_aligned = xr.align(predictors_cube, emo1, join="inner")
    logger.info(f"Predictor cube shape: {predictors_cube.shape}")
    logger.info(f"Target cube shape: {emo1.shape}")
    return predictors_cube, emo1
