# src/emo_downscale/data/openeo_loader.py

from typing import Dict, Tuple
import os
import xarray as xr
from openeo.local import LocalConnection
from dask.distributed import get_client
import copy
import dask.array as da  # optional, but often handy

from emo_downscale.logging_utils import get_logger

# src/emo_downscale/data/openeo_loader.py
import glob
import re

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

def _select_and_validate_bands(preds_da: xr.DataArray,
                               targs_da: xr.DataArray,
                               data_cfg: Dict) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Select only the relevant bands from predictors/targets according to data_cfg["bands"],
    then assert that all required bands are present and ordered as in the config.
    """
    bands_cfg = data_cfg.get("bands", {})

    # expected predictors bands: era5 + pressure + dem
    expected_pred_bands = (
        bands_cfg.get("era5", [])
        + bands_cfg.get("pressure", [])
        + bands_cfg.get("dem", [])
    )
    # expected targets bands: emo1
    expected_targ_bands = bands_cfg.get("emo1", [])

    # --- Sanity: predictors must have "bands" coord ---
    if "bands" not in preds_da.coords:
        raise AssertionError(
            "[ZARR ERROR] Predictors DataArray has no 'bands' coordinate; "
            "cannot map ERA5/pressure/DEM features."
        )

    if "bands" not in targs_da.coords:
        raise AssertionError(
            "[ZARR ERROR] Targets DataArray has no 'bands' coordinate; "
            "cannot map EMO1 features."
        )

    # --- First: select only relevant bands (this drops any extras) ---
    # If some bands are missing, .sel will raise KeyError; we catch and rephrase.
    try:
        preds_da_sel = preds_da.sel(bands=expected_pred_bands)
    except KeyError:
        available = list(preds_da.coords["bands"].values)
        missing = sorted(set(expected_pred_bands) - set(available))
        raise AssertionError(
            "[ZARR ERROR] Missing predictor bands in cached Zarr.\n"
            f"  Expected (from YAML): {expected_pred_bands}\n"
            f"  Available in Zarr:    {available}\n"
            f"  Missing:              {missing}"
        )

    try:
        targs_da_sel = targs_da.sel(bands=expected_targ_bands)
    except KeyError:
        available = list(targs_da.coords["bands"].values)
        missing = sorted(set(expected_targ_bands) - set(available))
        raise AssertionError(
            "[ZARR ERROR] Missing target bands in cached Zarr.\n"
            f"  Expected (from YAML): {expected_targ_bands}\n"
            f"  Available in Zarr:    {available}\n"
            f"  Missing:              {missing}"
        )

    # --- Then: assert that the selected bands match exactly the config order ---
    pred_selected = list(preds_da_sel.coords["bands"].values)
    targ_selected = list(targs_da_sel.coords["bands"].values)

    assert pred_selected == expected_pred_bands, (
        "[ZARR ERROR] Predictor bands order/content mismatch after selection.\n"
        f"  Expected (from YAML): {expected_pred_bands}\n"
        f"  Selected from Zarr:   {pred_selected}"
    )

    assert targ_selected == expected_targ_bands, (
        "[ZARR ERROR] Target bands order/content mismatch after selection.\n"
        f"  Expected (from YAML): {expected_targ_bands}\n"
        f"  Selected from Zarr:   {targ_selected}"
    )

    logger.info(
        "[load] Zarr band verification passed.\n"
        f"  Predictors bands: {pred_selected}\n"
        f"  Targets bands:    {targ_selected}"
    )

    return preds_da_sel, targs_da_sel



def _open_cached_if_available(data_cfg: Dict):
    use_cached = data_cfg.get("use_cached_zarr", False)
    if not use_cached:
        return None, None

    # 1) Try year-wise Zarrs first (if you already implemented that)
    preds_da, targs_da = _open_yearwise_zarr_if_available(data_cfg)
    if preds_da is not None and targs_da is not None:
        # IMPORTANT: also restrict + validate bands here
        preds_da, targs_da = _select_and_validate_bands(preds_da, targs_da, data_cfg)
        logger.info("[load] Using year-wise cached Zarr (open_mfdataset).")
        return preds_da, targs_da

    # 2) Fallback to single monolithic Zarr (old behavior)
    pred_store, targ_store = _zarr_paths(data_cfg)
    if not (pred_store and targ_store):
        return None, None

    if os.path.exists(pred_store) and os.path.exists(targ_store):
        logger.info(f"Opening cached predictors from {pred_store}")
        logger.info(f"Opening cached targets    from {targ_store}")

        pred_ds = xr.open_zarr(pred_store, consolidated=True)
        targ_ds = xr.open_zarr(targ_store, consolidated=True)

        # Get the underlying DataArrays (likely 'predictors' and 'targets')
        preds_da = pred_ds[list(pred_ds.data_vars)[0]]
        targs_da = targ_ds[list(targ_ds.data_vars)[0]]

        # ✅ Select only relevant bands THEN assert
        preds_da, targs_da = _select_and_validate_bands(preds_da, targs_da, data_cfg)

        logger.info("Loaded cached predictors/targets Zarr successfully.")
        return preds_da, targs_da

    return None, None




def _load_era5_emo1_core(data_cfg: Dict) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Core loader:
      - builds ERA5/EMO1/DEM graph via openEO
      - executes to xarray (dask-backed)
      - normalizes to (time, bands, lat, lon)
    NO caching, NO zarr writing here.
    """

    # Attach to cluster if present
    try:
        client = get_client()
        logger.info(f"[core] Using existing Dask client: {client}")
    except ValueError:
        logger.warning(
            "[core] No active Dask client found – falling back to default scheduler."
        )

    spatial = data_cfg["spatial"]
    temporal = [data_cfg["temporal"]["start"], data_cfg["temporal"]["end"]]
    bands_cfg = data_cfg["bands"]
    urls = data_cfg["stac_urls"]

    logger.info(f"[core] temporal range: {temporal[0]} → {temporal[1]}")

    logger.info("[core] Creating LocalConnection to openEO backend (./)...")
    conn = LocalConnection("./")

    # 1. Build openEO graph
    logger.info("[core] Building ERA5 / pressure / EMO1 / DEM process graph...")
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

    logger.info("[core] Executing process graphs to xarray (building dask graph)...")
    predictors_x = predictors_cube.execute()
    emo1_x = emo1.execute()

    # 2. Normalize to DataArray
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

    # 3. Normalize dims to (time, bands, lat, lon)
    dims_pred = predictors_da.dims
    time_dim = next(d for d in dims_pred if "time" in d)
    lat_dim = next(d for d in dims_pred if d in ("lat", "y", "latitude", "ycoord"))
    lon_dim = next(d for d in dims_pred if d in ("lon", "x", "longitude", "xcoord"))
    bands_dim = next(d for d in dims_pred if d not in (time_dim, lat_dim, lon_dim))

    predictors_da = predictors_da.transpose(time_dim, bands_dim, lat_dim, lon_dim)
    emo1_da = emo1_da.transpose(time_dim, bands_dim, lat_dim, lon_dim)

    logger.info(f"[core] Predictors dims after transpose: {predictors_da.dims}")
    logger.info(f"[core] Targets dims after transpose:    {emo1_da.dims}")

    return predictors_da, emo1_da


def load_era5_emo1_cubes(data_cfg: Dict) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Public loader used by the training DataModule.

    It:
      - optionally opens cached Zarr
      - otherwise builds predictors/targets via _load_era5_emo1_core
      - optionally writes Zarr cache
      - returns dask-backed DataArrays (time, bands, lat, lon)
    """
    # 0. Try cached Zarr first
    preds_da, emo1_da = _open_cached_if_available(data_cfg)
    if preds_da is not None and emo1_da is not None:
        # --- NEW: apply spatial cropping from YAML ---
        spatial = data_cfg["spatial"]
        west, east = spatial["west"], spatial["east"]
        south, north = spatial["south"], spatial["north"]
        temporal = data_cfg["temporal"]
        start, end = temporal["start"], temporal["end"]

        
        # assume dims are ("time", "bands", "lat", "lon")
        preds_da = preds_da.sel(lat=slice(north, south), lon=slice(west, east))
        emo1_da  = emo1_da.sel(lat=slice(north, south), lon=slice(west, east))

        preds_da = preds_da.sel(time=slice(start, end))
        emo1_da = emo1_da.sel(time=slice(start, end))
        
        preds_da = preds_da.transpose("time", "bands", "lat", "lon")
        emo1_da  = emo1_da.transpose("time", "bands", "lat", "lon")

        

        logger.info(
            "[load] Loaded from cached Zarr + applied spatial crop "
            f"lat=[{south}, {north}], lon=[{west}, {east}]"
        )
        return preds_da, emo1_da

    # 1. Build from openEO for full temporal range
    predictors_da, emo1_da = _load_era5_emo1_core(data_cfg)

    # 2. Coarse chunks for writing to Zarr
    time_dim, bands_dim, lat_dim, lon_dim = "time", "bands", "lat", "lon"
    write_chunks = {
        time_dim: data_cfg.get("write_chunk_time", 8),
        bands_dim: -1,
        lat_dim: data_cfg.get("write_chunk_lat", 256),
        lon_dim: data_cfg.get("write_chunk_lon", 256),
    }

    predictors_write = predictors_da.chunk(write_chunks)
    emo1_write = emo1_da.chunk(write_chunks)
    logger.info(f"[load] Using coarse chunks for Zarr write: {write_chunks}")

    # 3. Optionally write Zarr cache
    if data_cfg.get("write_cached_zarr", False):
        pred_store, targ_store = _zarr_paths(data_cfg)
        if pred_store and targ_store:
            logger.info(f"[load] Writing predictors Zarr to {pred_store}")
            predictors_write.to_dataset(name="predictors").to_zarr(
                pred_store,
                mode="w",
                consolidated=True,
            )
            logger.info(f"[load] Writing targets Zarr to {targ_store}")
            emo1_write.to_dataset(name="targets").to_zarr(
                targ_store,
                mode="w",
                consolidated=True,
            )
            logger.info("[load] Finished writing cached Zarr stores.")

    return predictors_da, emo1_da

def _open_yearwise_zarr_if_available(data_cfg: Dict):
    """
    If per-year Zarr stores exist (predictors_<year>.zarr / targets_<year>.zarr),
    open them with xarray.open_mfdataset and return concatenated DataArrays.
    """
    base_pred_store, base_targ_store = _zarr_paths(data_cfg)
    if not base_pred_store or not base_targ_store:
        return None, None

    pred_dir = os.path.dirname(base_pred_store)
    targ_dir = os.path.dirname(base_targ_store)

    pred_base = os.path.basename(base_pred_store)  # e.g. "predictors.zarr"
    targ_base = os.path.basename(base_targ_store)  # e.g. "targets.zarr"

    pred_root = pred_base[:-5] if pred_base.endswith(".zarr") else pred_base
    targ_root = targ_base[:-5] if targ_base.endswith(".zarr") else targ_base

    pred_pattern = os.path.join(pred_dir, f"{pred_root}_*.zarr")
    targ_pattern = os.path.join(targ_dir, f"{targ_root}_*.zarr")

    pred_paths = sorted(glob.glob(pred_pattern))
    targ_paths = sorted(glob.glob(targ_pattern))

    if not pred_paths or not targ_paths:
        logger.info("[cache-yearwise] No per-year Zarr stores found.")
        return None, None

    # Optionally: enforce same years on both sides
    year_re = re.compile(r".*_(\d{4})\.zarr$")
    def years_from_paths(paths):
        out = {}
        for p in paths:
            m = year_re.match(p)
            if m:
                out[int(m.group(1))] = p
        return out

    pred_years = years_from_paths(pred_paths)
    targ_years = years_from_paths(targ_paths)
    common_years = sorted(set(pred_years) & set(targ_years))
    if not common_years:
        logger.warning("[cache-yearwise] No overlapping years between predictors/targets.")
        return None, None

    pred_paths_sorted = [pred_years[y] for y in common_years]
    targ_paths_sorted = [targ_years[y] for y in common_years]

    logger.info(f"[cache-yearwise] Opening predictors Zarr for years: {common_years}")
    pred_ds = xr.open_mfdataset(
        pred_paths_sorted,
        engine="zarr",
        concat_dim="time",
        combine="nested",
        parallel=True,
        chunks="auto",  # keep on-disk chunking (already patch-aligned)
    )

    logger.info(f"[cache-yearwise] Opening targets Zarr for years: {common_years}")
    targ_ds = xr.open_mfdataset(
        targ_paths_sorted,
        engine="zarr",
        concat_dim="time",
        combine="nested",
        parallel=True,
        chunks="auto",
    )

    # Each store has the same variable name as in prepare_year()
    preds_da = pred_ds["predictors"]
    targs_da = targ_ds["targets"]

    logger.info(
        "[cache-yearwise] Loaded concatenated predictors/targets: "
        f"shape preds={preds_da.shape}, targs={targs_da.shape}"
    )
    return preds_da, targs_da

