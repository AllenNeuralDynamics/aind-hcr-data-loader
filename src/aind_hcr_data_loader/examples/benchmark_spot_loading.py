# -*- coding: utf-8 -*-
"""Quick benchmark for pairwise spot parquet loading modes.

This script compares:
1) Full parquet load (no filtering)
2) Current loader filter (loads parquet, then filters by cell_id in pandas)
3) Optional pyarrow predicate filter (potential parquet IO speedup)

Example:
    python benchmark_spot_loading.py --mouse-id 790322 --repeats 3
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from statistics import mean

import pandas as pd

from aind_hcr_data_loader.loaders import get_hcr_dataset_pairwise


def _bench(fn, repeats: int = 3):
    times = []
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return out, times


def _fmt_stats(label: str, times: list[float], n_rows: int) -> str:
    return (
        f"{label:<30} | rows={n_rows:<10d} "
        f"| mean={mean(times):.3f}s | min={min(times):.3f}s | max={max(times):.3f}s"
    )


def _load_spots_pyarrow_filtered(parquet_path: Path, cell_ids: list[int]) -> pd.DataFrame:
    """Load parquet using a pyarrow predicate filter on cell_id."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    dataset = ds.dataset(parquet_path, format="parquet")
    table = dataset.to_table(filter=pc.field("cell_id").isin(pa.array(cell_ids)))
    return table.to_pandas()


def main():
    parser = argparse.ArgumentParser(description="Benchmark pairwise spot parquet loading")
    parser.add_argument("--mouse-id", type=str, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("/root/capsule/data"))
    parser.add_argument("--catalog-path", type=Path, default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--include-removed",
        action="store_true",
        help="Include removed spots where applicable",
    )
    args = parser.parse_args()

    print("Preparing dataset and pairwise handles (no spot load yet)...")
    dataset, pw_ds, _ = get_hcr_dataset_pairwise(
        mouse_id=args.mouse_id,
        data_dir=args.data_dir,
        load_spots=False,
        catalog_path=args.catalog_path,
    )

    if pw_ds is None:
        raise RuntimeError("No pairwise_unmixing asset found for this mouse.")

    print("Loading coreg IDs...")
    coreg_df = dataset.load_coreg_table()
    if "hcr_id" not in coreg_df.columns:
        raise RuntimeError("Coreg table missing expected 'hcr_id' column.")
    coreg_ids = coreg_df["hcr_id"].dropna().unique().tolist()
    print(f"Coreg hcr_id count: {len(coreg_ids)}")

    print("\nRunning benchmarks...")

    full_df, full_times = _bench(
        lambda: pw_ds.load_spots_parquet(return_removed=args.include_removed),
        repeats=args.repeats,
    )

    filtered_df, filtered_times = _bench(
        lambda: pw_ds.load_spots_parquet(
            return_removed=args.include_removed,
            cell_ids=coreg_ids,
        ),
        repeats=args.repeats,
    )

    print(_fmt_stats("full load", full_times, len(full_df)))
    print(_fmt_stats("loader filter by coreg_ids", filtered_times, len(filtered_df)))

    # Optional pyarrow predicate filter benchmark (unmixed parquet only)
    try:
        if pw_ds.unmixed_spots_parquet is None:
            raise RuntimeError("No unmixed_spots_parquet path available.")

        arrow_df, arrow_times = _bench(
            lambda: _load_spots_pyarrow_filtered(pw_ds.unmixed_spots_parquet, coreg_ids),
            repeats=args.repeats,
        )
        print(_fmt_stats("pyarrow predicate filter", arrow_times, len(arrow_df)))
    except Exception as exc:
        print(f"pyarrow predicate filter benchmark skipped: {exc}")

    print("\nInterpretation:")
    print("- If 'loader filter by coreg_ids' is close to 'full load', parquet IO dominates.")
    print("- If 'pyarrow predicate filter' is faster, pushdown filtering is helping.")


if __name__ == "__main__":
    main()
