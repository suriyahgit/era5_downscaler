# scripts/prepare_zarr.py

import argparse
import copy
from typing import Any, Dict

import dask
from dask.distributed import Client, LocalCluster
import xarray as xr

from datetime import datetime, timedelta


from emo_downscale.config import load_config
from emo_downscale.data.openeo_loader import _load_era5_emo1_core, _zarr_paths
from emo_downscale.logging_utils import setup_global_logger, get_logger

logger = get_logger("prepare_zarr")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare cached Zarr predictors/targets from openEO (year-wise)."
    )
    p.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config (e.g. configs/emo1_unet.yaml)",
    )
    return p.parse_args()


def build_cluster() -> Client:
    """
    Use a similar cluster to train.py, but dedicated to data prep.
    """
    cluster = LocalCluster(
        n_workers=14,
        threads_per_worker=1,
        memory_limit="7GB",
        worker_dashboard_address=False,
        diagnostics_port=None,
        silence_logs="WARNING",
    )
    client = Client(cluster)
    # ensure dask uses this
    dask.config.set(scheduler="distributed")
    logger.info(f"Started LocalCluster for data prep: {client}")
    return client


def year_range_from_cfg(data_cfg: Dict[str, Any]) -> range:
    """
    Derive full [start_year, end_year] from the temporal range in config.
    """
    start = data_cfg["temporal"]["start"]
    end   = data_cfg["temporal"]["end"]
    start_year = int(start[:4])
    end_year   = int(end[:4])
    return range(start_year, end_year + 1)


def prepare_year(
    base_data_cfg: Dict[str, Any],
    year: int,
    pred_store: str,
    targ_store: str,
    first: bool,
) -> None:
    """
    Process a single year:
      - override temporal range
      - call core openEO loader
      - coarsely chunk
      - write/append to Zarr along time
    """
    data_cfg = copy.deepcopy(base_data_cfg)
    data_cfg["temporal"]["start"] = f"{year}-01-01"
    data_cfg["temporal"]["end"] = (datetime(year, 12, 31) + timedelta(days=1)).strftime("%Y-%m-%d")

    # ensure we don't enter any cached path inside core
    data_cfg["use_cached_zarr"] = False
    data_cfg["write_cached_zarr"] = False

    logger.info(f"=== Year {year}: loading ERA5/EMO1 cubes ===")
    preds_da, emo1_da = _load_era5_emo1_core(data_cfg)

    # Coarse chunking for writing
    time_dim, bands_dim, lat_dim, lon_dim = "time", "bands", "lat", "lon"
    write_chunks = {
        time_dim: data_cfg.get("write_chunk_time", 1),
        bands_dim: 1,
        lat_dim: data_cfg.get("write_chunk_lat", 180),
        lon_dim: data_cfg.get("write_chunk_lon", 180),
    }
    preds_write = preds_da.chunk(write_chunks)
    emo1_write  = emo1_da.chunk(write_chunks)
    logger.info(f"Year {year}: write_chunks = {write_chunks}")

    # Convert to Dataset
    preds_ds = preds_write.to_dataset(name="predictors")
    targs_ds = emo1_write.to_dataset(name="targets")

    mode = "w" if first else "a"

    logger.info(
        f"Year {year}: writing to Zarr (mode={mode}) → {pred_store}, {targ_store}"
    )

    if first:
        # first year: create store
        preds_ds.to_zarr(pred_store, mode="w", consolidated=True)
        targs_ds.to_zarr(targ_store, mode="w", consolidated=True)
    else:
        # subsequent years: append along time
        preds_ds.to_zarr(pred_store, mode="a", append_dim="time")
        targs_ds.to_zarr(targ_store, mode="a", append_dim="time")

    logger.info(f"Year {year}: finished writing to Zarr.")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    run_name = cfg.get("run_name", "prepare_zarr")
    setup_global_logger(run_name + "_prepare")

    data_cfg = cfg["data"]
    pred_store, targ_store = _zarr_paths(data_cfg)

    if not pred_store or not targ_store:
        raise RuntimeError("predictors_feature_dir or targets_feature_dir not set in config.")

    client = build_cluster()

    years = list(year_range_from_cfg(data_cfg))
    logger.info(f"Preparing Zarr for years: {years}")
    logger.info(f"Predictors store: {pred_store}")
    logger.info(f"Targets store:    {targ_store}")

    # ------------------------------------------------------------------
    # Simple, robust: sequential year processing
    # This avoids any giant graph and stays within memory comfortably.
    # ------------------------------------------------------------------
    first = True
    for y in years:
        prepare_year(data_cfg, y, pred_store, targ_store, first=first)
        first = False

    logger.info("All years processed. Zarr caches ready.")
    client.close()


if __name__ == "__main__":
    main()