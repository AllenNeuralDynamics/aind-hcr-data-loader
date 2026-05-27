# -*- coding: utf-8 -*-
"""
Convenience loader functions that orchestrate the full
attach → load HCR dataset → load pairwise dataset workflow.
"""

from pathlib import Path

from aind_hcr_data_loader.codeocean_utils import (
    MouseRecord,
    attach_mouse_record_to_workstation,
    print_attach_results,
)
from aind_hcr_data_loader.hcr_dataset import create_hcr_dataset_from_schema
from aind_hcr_data_loader.pairwise_dataset import create_pairwise_unmixing_dataset

DATA_DIR = Path("/root/capsule/data")
BUCKET_NAME = "aind-open-data"


def get_hcr_dataset_pairwise(
    mouse_id: str,
    data_dir: Path = DATA_DIR,
    bucket_name: str = BUCKET_NAME,
    load_spots: bool = True,
    return_removed: bool = False,
    catalog_path: Path | None = None,
) -> tuple:
    """
    Attach assets, load the HCR dataset and (optionally) the pairwise-unmixing
    dataset for a single mouse.

    Returns
    -------
    (dataset, pw_ds, spots)
        ``pw_ds`` and ``spots`` are ``None`` when no pairwise asset exists in
        the catalog record, or when ``load_spots=False``.
        Pass ``return_removed=True`` to include removed spots in the combined
        DataFrame (distinguished by the ``removed`` bool column).
    """
    mouse_id = str(mouse_id)

    if catalog_path is None:
        catalog_path = Path(
            f"/src/ophys-mfish-dataset-catalog/mice/{mouse_id}.json"
            #f"/opt/venv/src/ophys-mfish-dataset-catalog/mice/{mouse_id}.json"
        )

    # ── attach & load ────────────────────────────────────────────────────────
    record = MouseRecord.from_json_file(catalog_path)
    results = attach_mouse_record_to_workstation(record)
    print_attach_results(results)

    dataset = create_hcr_dataset_from_schema(catalog_path, data_dir)
    dataset.summary()

    # ── pairwise unmixing (optional) ─────────────────────────────────────────
    pairwise_asset_name = record.derived_assets.get("pairwise_unmixing")
    spots = None

    if pairwise_asset_name is not None:
        pairwise_asset_path = data_dir / pairwise_asset_name
        # Some pipeline outputs nest data under a "pairwise_unmixing" subfolder
        if (pairwise_asset_path / "pairwise_unmixing").exists():
            pairwise_asset_path = pairwise_asset_path / "pairwise_unmixing"
        pw_ds = create_pairwise_unmixing_dataset(
            mouse_id=mouse_id,
            pairwise_asset_path=pairwise_asset_path,
            source_dataset=dataset,
        )
        pw_ds.summary()
        if load_spots:
            spots = pw_ds.load_spots_parquet(return_removed=return_removed)
    else:
        print("No pairwise_unmixing asset found in catalog record — skipping.")
        pw_ds = None

    return dataset, pw_ds, spots
