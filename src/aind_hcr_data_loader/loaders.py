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
    coreg_cells_only: bool = False,
    catalog_path: Path | None = None,
    pairwise_only: bool = False,
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
        Pass ``coreg_cells_only=True`` to load only spots whose ``cell_id`` is
        present in ``dataset.load_coreg_table()['hcr_id']``.

    pairwise_only : bool
        When ``True``, skip building the full multi-round ``HCRDataset`` and
        work only from the pairwise-unmixing asset. Use this when only the
        pairwise asset is mounted: the default path calls
        ``create_hcr_dataset_from_schema`` first, which reads a
        ``processing_manifest.json`` from *every* round folder in the catalog
        record and raises ``FileNotFoundError`` if any round asset is not
        mounted. ``create_pairwise_unmixing_dataset`` reads everything it needs
        from the pairwise asset itself; ``source_dataset`` (the full dataset) is
        only used to delegate zarr/segmentation/centroid access, which callers
        that only touch the unmixed cell/spot tables never need. In this mode
        the round assets are also not attached, ``dataset`` is returned as
        ``None``, and zarr/segmentation calls on ``pw_ds`` will raise.
    """
    mouse_id = str(mouse_id)

    if catalog_path is None:
        catalog_path = Path(
            f"/src/ophys-mfish-dataset-catalog/mice/{mouse_id}.json"
            #f"/opt/venv/src/ophys-mfish-dataset-catalog/mice/{mouse_id}.json"
        )

    if pairwise_only and coreg_cells_only:
        raise ValueError(
            "coreg_cells_only=True requires the full dataset, which is not built "
            "when pairwise_only=True."
        )

    # ── attach & load ────────────────────────────────────────────────────────
    record = MouseRecord.from_json_file(catalog_path)
    # Don't attach the round assets in pairwise-only mode — they aren't needed.
    results = attach_mouse_record_to_workstation(
        record, include_rounds=not pairwise_only
    )
    print_attach_results(results)

    if pairwise_only:
        dataset = None
    else:
        dataset = create_hcr_dataset_from_schema(catalog_path, data_dir)
        dataset.summary()

    # ── pairwise unmixing (optional) ─────────────────────────────────────────
    pairwise_asset_name = record.derived_assets.get("pairwise_unmixing")
    spots = None

    if pairwise_only and pairwise_asset_name is None:
        raise FileNotFoundError(
            f"pairwise_only=True but the catalog record {catalog_path} has no "
            "'derived_assets.pairwise_unmixing' entry."
        )

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
            coreg_cell_ids = None
            if coreg_cells_only:
                coreg_df = dataset.load_coreg_table()
                if "hcr_id" not in coreg_df.columns:
                    raise ValueError(
                        "Coreg table does not contain expected 'hcr_id' column."
                    )
                coreg_cell_ids = coreg_df["hcr_id"].dropna().unique().tolist()
                print(
                    f"Loading spots for coregistered cells only "
                    f"(n={len(coreg_cell_ids)} unique hcr_id values)."
                )
            spots = pw_ds.load_spots_parquet(
                return_removed=return_removed,
                cell_ids=coreg_cell_ids,
            )
    else:
        print("No pairwise_unmixing asset found in catalog record — skipping.")
        pw_ds = None

    return dataset, pw_ds, spots
