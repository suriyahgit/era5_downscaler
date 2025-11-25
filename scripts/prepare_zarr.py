# scripts/prepare_zarr.py

import argparse
import copy
from typing import Any, Dict

import dask
from dask.distributed import Client, LocalCluster
import xarray as xr  # still useful for types / future use
import os

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


# ---------------------------------------------------------------------------
# NEW: helper to dump Dask scheduler + worker info when something goes wrong
# ---------------------------------------------------------------------------
def dump_dask_state(client: Client, log=logger, prefix: str = "[DASK DEBUG]") -> None:
    """
    Try to log scheduler + worker info and worker logs.
    Safe to call in an except block.
    """
    try:
        info = client.scheduler_info()
        workers = info.get("workers", {})
        log.error(f"{prefix} Scheduler has {len(workers)} workers.")

        for wid, w in workers.items():
            mem_limit = w.get("memory_limit", 0)
            nthreads = w.get("nthreads", "?")
            last_seen = w.get("last_seen", "?")
            log.error(
                f"{prefix} Worker {wid}: nthreads={nthreads}, "
                f"memory_limit={mem_limit/1e9:.2f} GB, last_seen={last_seen}"
            )

        # Worker logs – this can be a lot, so we tail it
        logs = client.get_worker_logs()  # {worker-id: list[(level, msg)]} or {id: str}
        for wid, content in logs.items():
            if isinstance(content, list):
                lines = [msg for level, msg in content]
                tail = "\n".join(lines[-200:])  # last ~200 lines
            else:
                tail = content[-8000:]  # last ~8k chars if it's a string
            log.error(f"{prefix} === Tail of logs for {wid} ===\n{tail}")
    except Exception as e:
        log.error(f"{prefix} Failed to dump Dask state: {e}", exc_info=True)


def build_cluster() -> Client:
    """
    Use a similar cluster to train.py, but dedicated to data prep.
    """
    cluster = LocalCluster(
        n_workers=8,
        threads_per_worker=1,
        memory_limit="12GB",
        # for debugging, turn ON dashboard instead of hiding it
        dashboard_address=":5054",
        diagnostics_port=5055,
        silence_logs="WARNING",
    )
    client = Client(cluster)
    # ensure dask uses this
    dask.config.set(scheduler="distributed")
    logger.info(f"Started LocalCluster for data prep: {client}")
    logger.info(f"Dask dashboard at {cluster.dashboard_link}")
    return client


def year_range_from_cfg(data_cfg: Dict[str, Any]) -> range:
    """
    Derive full [start_year, end_year] from the temporal range in config.
    """
    start = data_cfg["temporal"]["start"]
    end = data_cfg["temporal"]["end"]
    start_year = int(start[:4])
    end_year = int(end[:4])
    return range(start_year, end_year + 1)


def prepare_year(
    base_data_cfg: Dict[str, Any],
    year: int,
    pred_store: str,
    targ_store: str,
) -> None:
    """
    Process a single year:
      - override temporal range
      - call core openEO loader
      - coarsely chunk
      - write Zarr for *this* year only (no appending).
    """
    data_cfg = copy.deepcopy(base_data_cfg)
    data_cfg["temporal"]["start"] = f"{year}-01-01"
    # include last day by going to Jan 1 of next year
    data_cfg["temporal"]["end"] = (datetime(year, 12, 31) + timedelta(days=1)).strftime(
        "%Y-%m-%d"
    )

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
    emo1_write = emo1_da.chunk(write_chunks)
    logger.info(f"Year {year}: write_chunks = {write_chunks}")

    # Convert to Dataset
    preds_ds = preds_write.to_dataset(name="predictors")
    targs_ds = emo1_write.to_dataset(name="targets")

    logger.info(f"Year {year}: writing per-year Zarr → {pred_store}, {targ_store}")

    # always write a fresh store for that year
    preds_ds.to_zarr(pred_store, mode="w", consolidated=True)
    targs_ds.to_zarr(targ_store, mode="w", consolidated=True)

    del preds_da, emo1_da, preds_write, emo1_write, preds_ds, targs_ds
    import gc

    gc.collect()

    logger.info(f"Year {year}: finished writing year-specific Zarr.")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    run_name = cfg.get("run_name", "prepare_zarr")
    setup_global_logger(run_name + "_prepare")

    data_cfg = cfg["data"]
    base_pred_store, base_targ_store = _zarr_paths(data_cfg)

    if not base_pred_store or not base_targ_store:
        raise RuntimeError(
            "predictors_feature_dir or targets_feature_dir not set in config."
        )

    # Derive directory + base names
    pred_dir = os.path.dirname(base_pred_store)
    targ_dir = os.path.dirname(base_targ_store)

    pred_base = os.path.basename(base_pred_store)  # e.g. "predictors.zarr"
    targ_base = os.path.basename(base_targ_store)  # e.g. "targets.zarr"

    # Strip optional ".zarr" suffix
    pred_root = pred_base[:-5] if pred_base.endswith(".zarr") else pred_base
    targ_root = targ_base[:-5] if targ_base.endswith(".zarr") else targ_base

    client = build_cluster()

    try:
        years = list(year_range_from_cfg(data_cfg))
        logger.info(f"Preparing per-year Zarr for years: {years}")
        logger.info(f"Base predictors dir: {pred_dir}, root: {pred_root}")
        logger.info(f"Base targets    dir: {targ_dir}, root: {targ_root}")

        for y in years:
            year_pred_store = os.path.join(pred_dir, f"{pred_root}_{y}.zarr")
            year_targ_store = os.path.join(targ_dir, f"{targ_root}_{y}.zarr")
            logger.info(f"Year {y}: predictors store → {year_pred_store}")
            logger.info(f"Year {y}: targets    store → {year_targ_store}")

            # soft reset: clear scheduler state + restart workers
            logger.info(f"Restarting Dask cluster before processing year {y}...")
            client.restart()

            try:
                prepare_year(data_cfg, y, year_pred_store, year_targ_store)
            except Exception:
                logger.exception(f"Year {y}: failure during prepare_year.")
                try:
                    dump_dask_state(client, logger)
                except Exception:
                    logger.exception("Also failed to inspect Dask workers.")
                raise

        logger.info("All years processed. Year-wise Zarr caches ready.")

    finally:
        logger.info("Closing Dask client and cluster.")
        client.close()


if __name__ == "__main__":
    main()
