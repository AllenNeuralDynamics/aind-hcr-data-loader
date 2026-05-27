# -*- coding: utf-8 -*-
"""
CZ-stack / HCR co-registration asset loader.

The ``czstack_hcr_coreg`` asset has the following layout::

    782149_2025-05-01_ctl-czstack-hcr-coreg_2026-03-03_00-00-00/
        782149_coreg_table.csv
        782149_2025-05-01_czstack_cell_centroids.csv

The coreg asset is loaded as an *optional attribute* on ``HCRDataset``
(``dataset.czstack_coreg_files``).  It is attached automatically when the
catalog record contains a ``"czstack_hcr_coreg"`` key in ``derived_assets``,
or it can be attached manually::

    from aind_hcr_data_loader.coreg_dataset import create_coreg_files

    dataset.czstack_coreg_files = create_coreg_files(
        '/root/capsule/data/782149_2025-05-01_ctl-czstack-hcr-coreg_2026-03-03_00-00-00'
    )

    # Then load via the dataset method:
    coreg_df = dataset.load_coreg_table()

Public API
----------
``CoregFiles``
    Dataclass holding the two resolved file paths.

``create_coreg_files``
    Factory: resolves paths from the asset folder and returns a ``CoregFiles``
    object.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class CoregFiles:
    """
    Paths to the key outputs of the czstack-hcr co-registration pipeline.

    Attributes
    ----------
    coreg_table : Path or None
        ``{mouse_id}_coreg_table.csv`` — per-cell co-registration match table.
    czstack_cell_centroids : Path or None
        ``{mouse_id}_*_czstack_cell_centroids.csv`` — CZ-stack cell centroid
        coordinates.
    asset_path : Path or None
        Root folder of the coreg asset (kept for reference / repr).
    """

    coreg_table: Optional[Path] = None
    czstack_cell_centroids: Optional[Path] = None
    asset_path: Optional[Path] = None

    def __repr__(self) -> str:
        ct = "✓" if self.coreg_table and self.coreg_table.exists() else "✗"
        cc = (
            "✓"
            if self.czstack_cell_centroids and self.czstack_cell_centroids.exists()
            else "✗"
        )
        name = self.asset_path.name if self.asset_path else "unknown"
        return f"CoregFiles(asset='{name}', coreg_table={ct}, czstack_centroids={cc})"


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def create_coreg_files(asset_path) -> CoregFiles:
    """
    Resolve file paths for a czstack-hcr coreg asset and return a
    ``CoregFiles`` object.

    Parameters
    ----------
    asset_path : str or Path
        Root of the coreg asset, e.g.
        ``/root/capsule/data/782149_2025-05-01_ctl-czstack-hcr-coreg_2026-03-03_00-00-00``.

    Returns
    -------
    CoregFiles

    Raises
    ------
    FileNotFoundError
        If ``asset_path`` does not exist.
    """
    asset_path = Path(asset_path)

    if not asset_path.exists():
        raise FileNotFoundError(f"Coreg asset not found: {asset_path}")

    def _check(p: Path) -> Optional[Path]:
        return p if p.exists() else None

    # mouse_id is the first token in the folder name
    mouse_id = asset_path.name.split("_")[0]

    coreg_table = _check(asset_path / f"{mouse_id}_coreg_table.csv")

    # Centroids file includes the acquisition date; glob for it
    centroids_candidates = sorted(
        asset_path.glob(f"{mouse_id}_*czstack_cell_centroids.csv")
    )
    czstack_cell_centroids = centroids_candidates[0] if centroids_candidates else None

    return CoregFiles(
        coreg_table=coreg_table,
        czstack_cell_centroids=czstack_cell_centroids,
        asset_path=asset_path,
    )


# ---------------------------------------------------------------------------
# Free-function loader
# ---------------------------------------------------------------------------


def load_coreg_table(coreg_files: CoregFiles) -> pd.DataFrame:
    """
    Load the co-registration match table from a ``CoregFiles`` object.

    Parameters
    ----------
    coreg_files : CoregFiles

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If ``coreg_table`` path is ``None`` or the file does not exist.
    """
    path = coreg_files.coreg_table
    if path is None or not path.exists():
        raise FileNotFoundError(
            f"coreg_table not found: {path}\n"
            "Make sure the coreg asset is attached to the dataset "
            "(dataset.czstack_coreg_files)."
        )
    return pd.read_csv(path)
