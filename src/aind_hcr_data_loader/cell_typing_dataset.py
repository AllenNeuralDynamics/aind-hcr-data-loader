# -*- coding: utf-8 -*-
"""
Cell-typing asset loader.

This module provides a dataclass and helpers for locating and loading data
produced by the cell-typing pipeline.  The cell-typing asset has the following
layout::

    HCR_782149_cell-typing_2026-03-09_12-00-00/
        all_cells/
            HCR_782149_pairwise-unmixing_2026-03-06_12-00-00/   ← one subfolder
                mapped_data/
                    basic_results.csv        ← taxonomy cell types (4-line comment header)
                    mapped_cellxgene.h5ad    ← AnnData file
        inhibitory_cells/
            ...
        output/
            ...

The cell-typing asset is loaded as an *optional attribute* on ``HCRDataset``
(``dataset.cell_typing_files``).  It is attached automatically when the mouse
config contains a ``"cell_typing"`` key, or it can be attached manually::

    from aind_hcr_data_loader.cell_typing_dataset import create_cell_typing_files

    dataset.cell_typing_files = create_cell_typing_files(
        cell_typing_asset_path='/root/capsule/data/HCR_782149_cell-typing_2026-03-09_12-00-00'
    )

    # Then load via the dataset methods:
    df   = dataset.load_taxonomy_cell_types()
    path = dataset.load_taxonomy_cell_types_h5ad()

Public API
----------
``CellTypingFiles``
    Dataclass holding the two resolved file paths.

``create_cell_typing_files``
    Factory: resolves paths and returns a ``CellTypingFiles`` object.

``load_taxonomy_cell_types``
    Free function: reads ``basic_results.csv`` (skipping 4 comment lines).

``load_taxonomy_cell_types_h5ad``
    Free function: returns the ``Path`` to ``mapped_cellxgene.h5ad``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class CellTypingFiles:
    """
    Paths to the key outputs of the cell-typing pipeline.

    Attributes
    ----------
    basic_results : Path or None
        ``mapped_data/basic_results.csv`` — per-cell taxonomy assignments.
        The file has 4 comment lines (``#…``) before the CSV header; use
        ``load_taxonomy_cell_types()`` to read it correctly.
    mapped_cellxgene_h5ad : Path or None
        ``mapped_data/mapped_cellxgene.h5ad`` — AnnData file with cell ×
        gene expression and taxonomy embeddings.
    asset_path : Path or None
        Root folder of the cell-typing asset (kept for reference / repr).
    """

    basic_results: Optional[Path] = None
    mapped_cellxgene_h5ad: Optional[Path] = None
    asset_path: Optional[Path] = None

    def __repr__(self) -> str:
        br = "✓" if self.basic_results and self.basic_results.exists() else "✗"
        h5 = (
            "✓"
            if self.mapped_cellxgene_h5ad and self.mapped_cellxgene_h5ad.exists()
            else "✗"
        )
        name = self.asset_path.name if self.asset_path else "unknown"
        return f"CellTypingFiles(asset='{name}', basic_results={br}, h5ad={h5})"


# ---------------------------------------------------------------------------
# Free-function loaders
# ---------------------------------------------------------------------------


def load_taxonomy_cell_types(cell_typing_files: CellTypingFiles) -> pd.DataFrame:
    """
    Load ``basic_results.csv`` from the cell-typing asset.

    The file begins with 4 comment lines (``#…``) that describe the taxonomy
    hierarchy and algorithm used.  These are skipped automatically.

    Parameters
    ----------
    cell_typing_files : CellTypingFiles

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by ``cell_id`` with columns:
        ``hierarchy_consistent``, ``class_label``, ``class_name``,
        ``class_bootstrapping_probability``, ``subclass_label``,
        ``subclass_name``, ``subclass_bootstrapping_probability``,
        ``supertype_label``, ``supertype_name``,
        ``supertype_bootstrapping_probability``, ``cluster_label``,
        ``cluster_name``, ``cluster_alias``,
        ``cluster_bootstrapping_probability``.

    Raises
    ------
    FileNotFoundError
        If ``basic_results`` path is ``None`` or the file does not exist.
    """
    path = cell_typing_files.basic_results
    if path is None or not path.exists():
        raise FileNotFoundError(
            f"taxonomy cell types file not found: {path}\n"
            "Make sure the cell-typing asset is attached to the dataset "
            "(dataset.cell_typing_files)."
        )
    df = pd.read_csv(path, skiprows=4, index_col="cell_id")
    return df


def load_taxonomy_cell_types_h5ad(cell_typing_files: CellTypingFiles) -> Path:
    """
    Return the path to ``mapped_cellxgene.h5ad``.

    The caller is responsible for loading the file (e.g. with
    ``anndata.read_h5ad(path)``).

    Parameters
    ----------
    cell_typing_files : CellTypingFiles

    Returns
    -------
    Path

    Raises
    ------
    FileNotFoundError
        If ``mapped_cellxgene_h5ad`` path is ``None`` or the file does not
        exist.
    """
    path = cell_typing_files.mapped_cellxgene_h5ad
    if path is None or not path.exists():
        raise FileNotFoundError(
            f"mapped_cellxgene.h5ad not found: {path}\n"
            "Make sure the cell-typing asset is attached to the dataset "
            "(dataset.cell_typing_files)."
        )
    return path


# ---------------------------------------------------------------------------
# Internal discovery helper
# ---------------------------------------------------------------------------


def _discover_mapped_data(
    cell_typing_asset_path: Path,
    pairwise_asset_name: Optional[str] = None,
) -> Path:
    """
    Locate the ``mapped_data/`` folder inside a cell-typing asset.

    The expected structure is::

        <cell_typing_asset_path>/
            all_cells/
                <pairwise_asset_name>/    ← auto-discovered if not given
                    mapped_data/

    Parameters
    ----------
    cell_typing_asset_path : Path
        Root of the cell-typing asset.
    pairwise_asset_name : str, optional
        Name of the pairwise-unmixing subfolder inside ``all_cells/``.
        When ``None`` the first (and usually only) subdirectory is used.

    Returns
    -------
    Path
        Absolute path to the ``mapped_data/`` directory.

    Raises
    ------
    FileNotFoundError
        If ``all_cells/`` does not exist or no subfolder is found.
    """
    all_cells_dir = cell_typing_asset_path / "all_cells"
    if not all_cells_dir.exists():
        raise FileNotFoundError(
            f"Expected 'all_cells/' directory not found in: {cell_typing_asset_path}"
        )

    if pairwise_asset_name is not None:
        pairwise_dir = all_cells_dir / pairwise_asset_name
        if not pairwise_dir.exists():
            raise FileNotFoundError(
                f"Pairwise asset subfolder '{pairwise_asset_name}' not found in: {all_cells_dir}"
            )
    else:
        # Auto-discover: take the first subdirectory
        subdirs = [d for d in sorted(all_cells_dir.iterdir()) if d.is_dir()]
        if not subdirs:
            raise FileNotFoundError(
                f"No subdirectory found inside 'all_cells/' in: {cell_typing_asset_path}"
            )
        pairwise_dir = subdirs[0]
        if len(subdirs) > 1:
            import warnings
            warnings.warn(
                f"Multiple subdirectories found under 'all_cells/'; using '{pairwise_dir.name}'. "
                "Pass pairwise_asset_name= to create_cell_typing_files() to be explicit.",
                stacklevel=3,
            )

    mapped_data_dir = pairwise_dir / "mapped_data"
    if not mapped_data_dir.exists():
        raise FileNotFoundError(
            f"Expected 'mapped_data/' directory not found in: {pairwise_dir}"
        )

    return mapped_data_dir


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def create_cell_typing_files(
    cell_typing_asset_path,
    pairwise_asset_name: Optional[str] = None,
) -> CellTypingFiles:
    """
    Resolve file paths for a cell-typing asset and return a
    ``CellTypingFiles`` object.

    Parameters
    ----------
    cell_typing_asset_path : str or Path
        Root of the cell-typing asset, e.g.
        ``/root/capsule/data/HCR_782149_cell-typing_2026-03-09_12-00-00``.
    pairwise_asset_name : str, optional
        Name of the pairwise-unmixing subfolder inside ``all_cells/``.
        When ``None`` the single subdirectory is auto-discovered.

    Returns
    -------
    CellTypingFiles

    Examples
    --------
    >>> from aind_hcr_data_loader.cell_typing_dataset import create_cell_typing_files
    >>> files = create_cell_typing_files(
    ...     '/root/capsule/data/HCR_782149_cell-typing_2026-03-09_12-00-00'
    ... )
    >>> files
    CellTypingFiles(asset='HCR_782149_cell-typing_2026-03-09_12-00-00',
                    basic_results=✓, h5ad=✓)
    """
    cell_typing_asset_path = Path(cell_typing_asset_path)

    if not cell_typing_asset_path.exists():
        raise FileNotFoundError(
            f"Cell-typing asset not found: {cell_typing_asset_path}"
        )

    mapped_data_dir = _discover_mapped_data(cell_typing_asset_path, pairwise_asset_name)

    def _check(p: Path) -> Optional[Path]:
        return p if p.exists() else None

    return CellTypingFiles(
        basic_results=_check(mapped_data_dir / "basic_results.csv"),
        mapped_cellxgene_h5ad=_check(mapped_data_dir / "mapped_cellxgene.h5ad"),
        asset_path=cell_typing_asset_path,
    )
