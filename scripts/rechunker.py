#!/usr/bin/env python3

import os
import xarray as xr
import dask
from dask.distributed import Client, LocalCluster


BASE_DIR = "/mnt/CEPH_PROJECTS/InterTwin/Climate_Downscaling/PAPER/v2"

FEAT_DIR = os.path.join(BASE_DIR, "features_cnn")
TARG_DIR = os.path.join(BASE_DIR, "targets_cnn")

FEAT_OUT = os.path.join(BASE_DIR, "predictors_monolithic.zarr")
TARG_OUT = os.path.join(BASE_DIR, "targets_monolithic.zarr")

YEARS = range(2000, 2021)

TARGET_CHUNKS = {
    "time": 256,
    "bands": 1,
    "lat": 32,
    "lon": 32,
}


def start_cluster():
    cluster = LocalCluster(
        n_workers=4,
        threads_per_worker=2,
        memory_limit="24GB",  # total ~96GB
        dashboard_address=":8787",
    )
    client = Client(cluster)

    dask.config.set({
        "distributed.worker.memory.spill": True,
        "distributed.worker.memory.target": 0.95,
        "distributed.scheduler.work-stealing": True,
    })
    return client


def open_years(base_dir, prefix):
    stores = [
        os.path.join(base_dir, f"{prefix}_{y}.zarr")
        for y in YEARS
    ]

    ds = xr.open_mfdataset(
        stores,
        engine="zarr",
        combine="nested",
        concat_dim="time",
        parallel=True,
        chunks="auto",
    )

    return ds.sortby("time")


def rechunk_and_write(ds, out_path):
    print(f"\nRechunking and writing → {out_path}")

    # ensure uniform existing chunks
    ds = ds.unify_chunks()

    # drop any encoding["chunks"] metadata
    for v in ds.variables:
        ds[v].encoding.pop("chunks", None)

    # apply new chunking
    ds = ds.chunk(TARGET_CHUNKS)

    # write monolithic zarr
    ds.to_zarr(
        out_path,
        mode="w",
        consolidated=True,
        compute=True,
    )

    print("Done:", out_path)


def main():
    start_cluster()

    print("\n=== Processing Predictors ===")
    pred = open_years(FEAT_DIR, "predictors")
    rechunk_and_write(pred, FEAT_OUT)

    print("\n=== Processing Targets ===")
    targ = open_years(TARG_DIR, "targets")
    rechunk_and_write(targ, TARG_OUT)


if __name__ == "__main__":
    main()
