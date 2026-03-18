# -*- coding: utf-8 -*-
"""
Pairwise-unmixing dataset loader.

This module provides classes and a factory function for loading data produced
by the pairwise spectral-unmixing pipeline.  The pairwise unmixing asset has a
different directory layout from the standard ``_processed_`` assets:

    HCR_782149_pairwise-unmixing_2026-03-06_12-00-00/
        782149_R1/
            mixed_spots_R1.pkl
            unmixed_spots_R1_minDist_1.pkl
            mixed_cell_by_gene.csv
            unmixed_cell_by_gene.csv
            mixed_cell_by_gene_filtered.csv       
            unmixed_cell_by_gene_filtered.csv     
            R1_ratios_matrix.csv
            R1_reassignment_matrix.csv
            R1_reassignment_matrix_norm_rows.csv
            R1_spot_fate_matrix.csv
            R1_loss_history.pkl
            ds_config.json                        <- contains processing manifest
            unmixing_config.json                  <- pairwise-specific params
            *.png                                 <- diagnostic plots
        782149_R2/
            ...
        unmixed_cell_by_gene_all_rounds.csv
        mixed_cell_by_gene_all_rounds.csv
        inhibitory_cells_unmixed/
            unmixed_inhibitory_cells.csv
            unmixed_cluster_labels.csv
            unmixed_sorted_cell_ids.csv
            *.png
        inhibitory_cells_mixed/
            mixed_inhibitory_cells.csv
            mixed_cluster_labels.csv
            mixed_sorted_cell_ids.csv
            *.png

Public API
----------
``PairwiseUnmixingDiagnostics``
    Dataclass holding paths to per-round diagnostic files.

``InhibitoryCellAnalysis``
    Dataclass holding paths to the top-level inhibitory-cell analysis products.

``PairwiseUnmixingRound``
    Represents one round of pairwise-unmixing output.  Reuses the parent
    ``HCRRound`` interface so it can slot into the ``HCRDataset.rounds`` dict.

``PairwiseUnmixingDataset``
    Subclass of ``HCRDataset``.  Populates ``self.rounds`` with
    ``PairwiseUnmixingRound`` objects (zarr / segmentation fields are ``None``)
    and adds pairwise-specific attributes and methods.  If a
    ``source_dataset`` is supplied, zarr / segmentation / metadata calls are
    transparently delegated to it.

``create_pairwise_unmixing_dataset``
    Factory function: auto-discovers round sub-folders, parses ``ds_config.json``
    for processing manifests, and returns a ready-to-use
    ``PairwiseUnmixingDataset``.

Usage example
-------------
    from pathlib import Path
    from aind_hcr_data_loader.pairwise_dataset import create_pairwise_unmixing_dataset
    from aind_hcr_data_loader.hcr_dataset import create_hcr_dataset_from_config

    data_dir = Path('/root/capsule/data')

    # Optional: load the original processed dataset for zarr / segmentation access
    src_ds = create_hcr_dataset_from_config(
        '782149',
        config_path='/root/capsule/code/MOUSE_HCR_CONFIG.json'
    )

    pw_ds = create_pairwise_unmixing_dataset(
        mouse_id='782149',
        pairwise_asset_path=data_dir / 'HCR_782149_pairwise-unmixing_2026-03-06_12-00-00',
        source_dataset=src_ds,
    )

    # Use inherited HCRDataset methods
    cxg = pw_ds.create_cell_gene_matrix(unmixed=True)

    # Pairwise-specific: pre-built all-rounds table
    cxg_all = pw_ds.load_aggregated_cxg(unmixed=True)

    # Per-round filtered CxG (first-class product)
    filt = pw_ds.rounds['R3'].load_filtered_cxg(unmixed=True)

    # Diagnostics
    fate = pw_ds.rounds['R5'].load_spot_fate_matrix()

    # Inhibitory-cell analysis
    inhib = pw_ds.load_inhibitory_cells(unmixed=True)

    # Segmentation delegated to source_dataset
    mask = pw_ds.load_segmentation_mask('R1')
"""

import json
import pickle as pkl
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from aind_hcr_data_loader.hcr_dataset import (
    HCRDataset,
    HCRRound,
    SegmentationFiles,
    SpotFiles,
    TileAlignmentFiles,
    ZarrDataFiles,
    MetadataFiles,
)


# ---------------------------------------------------------------------------
# Dataclasses for pairwise-specific products
# ---------------------------------------------------------------------------


@dataclass
class PairwiseUnmixingDiagnostics:
    """
    Paths to per-round diagnostic files produced by the pairwise unmixing
    pipeline.

    Attributes
    ----------
    ratios_matrix : Path or None
        ``R{N}_ratios_matrix.csv`` — channel-pair bleed-through ratio matrix.
    reassignment_matrix : Path or None
        ``R{N}_reassignment_matrix.csv`` — spot reassignment counts.
    reassignment_matrix_norm_rows : Path or None
        ``R{N}_reassignment_matrix_norm_rows.csv`` — row-normalised version.
    spot_fate_matrix : Path or None
        ``R{N}_spot_fate_matrix.csv`` — fixed / reassigned / removed counts.
    loss_history : Path or None
        ``R{N}_loss_history.pkl`` — training-loss history for the unmixing
        optimisation.
    plots : dict
        Mapping of plot label → ``Path`` for PNG diagnostic figures.
    """

    ratios_matrix: Optional[Path] = None
    reassignment_matrix: Optional[Path] = None
    reassignment_matrix_norm_rows: Optional[Path] = None
    spot_fate_matrix: Optional[Path] = None
    loss_history: Optional[Path] = None
    plots: Dict[str, Path] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def load_ratios_matrix(self) -> pd.DataFrame:
        """Load ``R{N}_ratios_matrix.csv`` into a DataFrame."""
        if self.ratios_matrix is None or not self.ratios_matrix.exists():
            raise FileNotFoundError(f"Ratios matrix not found: {self.ratios_matrix}")
        return pd.read_csv(self.ratios_matrix, index_col=0)

    def load_reassignment_matrix(self, normalised: bool = False) -> pd.DataFrame:
        """
        Load the reassignment matrix.

        Parameters
        ----------
        normalised : bool
            If ``True`` load the row-normalised version.
        """
        path = self.reassignment_matrix_norm_rows if normalised else self.reassignment_matrix
        if path is None or not path.exists():
            raise FileNotFoundError(f"Reassignment matrix not found: {path}")
        return pd.read_csv(path, index_col=0)

    def load_spot_fate_matrix(self) -> pd.DataFrame:
        """Load ``R{N}_spot_fate_matrix.csv`` into a DataFrame."""
        if self.spot_fate_matrix is None or not self.spot_fate_matrix.exists():
            raise FileNotFoundError(f"Spot fate matrix not found: {self.spot_fate_matrix}")
        return pd.read_csv(self.spot_fate_matrix)

    def load_loss_history(self):
        """Load the pickled loss-history object."""
        if self.loss_history is None or not self.loss_history.exists():
            raise FileNotFoundError(f"Loss history not found: {self.loss_history}")
        with open(self.loss_history, "rb") as f:
            return pkl.load(f)

    def __repr__(self) -> str:
        available = [
            name
            for name, path in [
                ("ratios_matrix", self.ratios_matrix),
                ("reassignment_matrix", self.reassignment_matrix),
                ("reassignment_matrix_norm_rows", self.reassignment_matrix_norm_rows),
                ("spot_fate_matrix", self.spot_fate_matrix),
                ("loss_history", self.loss_history),
            ]
            if path is not None and path.exists()
        ]
        return (
            f"PairwiseUnmixingDiagnostics("
            f"available={available}, plots={list(self.plots.keys())})"
        )


@dataclass
class InhibitoryCellAnalysis:
    """
    Paths to the top-level inhibitory-cell analysis outputs that sit at the
    root of the pairwise unmixing asset.

    Both ``unmixed`` and ``mixed`` variants are captured.

    Attributes
    ----------
    unmixed_inhibitory_cells : Path or None
        ``inhibitory_cells_unmixed/unmixed_inhibitory_cells.csv``
    unmixed_cluster_labels : Path or None
        ``inhibitory_cells_unmixed/unmixed_cluster_labels.csv``
    unmixed_sorted_cell_ids : Path or None
        ``inhibitory_cells_unmixed/unmixed_sorted_cell_ids.csv``
    unmixed_plots : dict
        PNG plots in the ``inhibitory_cells_unmixed/`` folder.
    mixed_inhibitory_cells : Path or None
    mixed_cluster_labels : Path or None
    mixed_sorted_cell_ids : Path or None
    mixed_plots : dict
    """

    unmixed_inhibitory_cells: Optional[Path] = None
    unmixed_cluster_labels: Optional[Path] = None
    unmixed_sorted_cell_ids: Optional[Path] = None
    unmixed_plots: Dict[str, Path] = field(default_factory=dict)

    mixed_inhibitory_cells: Optional[Path] = None
    mixed_cluster_labels: Optional[Path] = None
    mixed_sorted_cell_ids: Optional[Path] = None
    mixed_plots: Dict[str, Path] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def load_inhibitory_cells(self, unmixed: bool = True) -> pd.DataFrame:
        """
        Load the inhibitory-cell table.

        Parameters
        ----------
        unmixed : bool
            ``True`` → load from the ``unmixed`` analysis folder.
        """
        path = (
            self.unmixed_inhibitory_cells if unmixed else self.mixed_inhibitory_cells
        )
        if path is None or not path.exists():
            raise FileNotFoundError(f"Inhibitory cells file not found: {path}")
        df = pd.read_csv(path, index_col=0)
        df.index.name = "cell_id"
        return df

    def load_cluster_labels(self, unmixed: bool = True) -> pd.DataFrame:
        """Load cluster-label assignments."""
        path = (
            self.unmixed_cluster_labels if unmixed else self.mixed_cluster_labels
        )
        if path is None or not path.exists():
            raise FileNotFoundError(f"Cluster labels file not found: {path}")
        return pd.read_csv(path, index_col=0)

    def load_sorted_cell_ids(self, unmixed: bool = True) -> pd.DataFrame:
        """Load the sorted-cell-id list."""
        path = (
            self.unmixed_sorted_cell_ids if unmixed else self.mixed_sorted_cell_ids
        )
        if path is None or not path.exists():
            raise FileNotFoundError(f"Sorted cell IDs file not found: {path}")
        return pd.read_csv(path, index_col=0)

    def __repr__(self) -> str:
        u_ok = self.unmixed_inhibitory_cells is not None and self.unmixed_inhibitory_cells.exists()
        m_ok = self.mixed_inhibitory_cells is not None and self.mixed_inhibitory_cells.exists()
        return (
            f"InhibitoryCellAnalysis("
            f"unmixed={'✓' if u_ok else '✗'}, "
            f"mixed={'✓' if m_ok else '✗'})"
        )


# ---------------------------------------------------------------------------
# PairwiseUnmixingRound
# ---------------------------------------------------------------------------


class PairwiseUnmixingRound(HCRRound):
    """
    One round of pairwise-unmixing output.

    Extends ``HCRRound`` so it slots naturally into ``HCRDataset.rounds``.
    Zarr, segmentation, tile-alignment, and spot-detection attributes are
    ``None``; accessing them raises a descriptive error unless a
    ``source_dataset`` is available on the parent ``PairwiseUnmixingDataset``.

    Extra attributes
    ----------------
    ds_config : dict
        Full content of ``ds_config.json`` (contains the embedded processing
        manifest and per-round gene dict).
    unmixing_config : dict
        Content of ``unmixing_config.json`` (pairwise-specific parameters:
        channel pairs, thresholds, unmixing method, …).
    diagnostics : PairwiseUnmixingDiagnostics
        Paths to diagnostic artefacts for this round.
    """

    def __init__(
        self,
        round_key: str,
        name: str,
        spot_files: SpotFiles,
        ds_config: dict,
        unmixing_config: dict,
        diagnostics: PairwiseUnmixingDiagnostics,
        processing_manifest: dict,
        parent_dataset: Optional["PairwiseUnmixingDataset"] = None,
    ):
        # Build stub ZarrDataFiles / SegmentationFiles so HCRRound.__init__
        # does not crash.
        stub_zarr = ZarrDataFiles(fused={})
        super().__init__(
            round_key=round_key,
            name=name,
            spot_files=spot_files,
            zarr_files=stub_zarr,
            segmentation_files=None,
            spot_detection_files={},
            processing_manifest=processing_manifest,
            tile_alignment_files=None,
            metadata_files=None,
            parent_dataset=parent_dataset,
        )
        self.ds_config = ds_config
        self.unmixing_config = unmixing_config
        self.diagnostics = diagnostics

    # ------------------------------------------------------------------
    # Filtered CxG (first-class product)
    # ------------------------------------------------------------------

    def load_filtered_cxg(self, unmixed: bool = True) -> pd.DataFrame:
        """
        Load the filtered cell-by-gene table produced by the pairwise pipeline.

        The filtered table includes a ``round_chan_gene`` column
        (e.g. ``R1-488-GFP``) in addition to ``cell_id``, ``gene``, and
        ``spot_count``.

        Parameters
        ----------
        unmixed : bool
            ``True`` → load ``unmixed_cell_by_gene_filtered.csv``.
            ``False`` → load ``mixed_cell_by_gene_filtered.csv``.

        Returns
        -------
        pd.DataFrame
        """
        path = (
            self.spot_files.unmixed_cxg_filtered
            if unmixed
            else self.spot_files.mixed_cxg_filtered
        )
        if path is None or not path.exists():
            label = "unmixed" if unmixed else "mixed"
            raise FileNotFoundError(
                f"Filtered {label} CxG not found for round {self.round_key}: {path}"
            )
        return pd.read_csv(path)

    # ------------------------------------------------------------------
    # Diagnostics convenience pass-throughs
    # ------------------------------------------------------------------

    def load_ratios_matrix(self) -> pd.DataFrame:
        """Load the channel-pair ratios matrix for this round."""
        return self.diagnostics.load_ratios_matrix()

    def load_reassignment_matrix(self, normalised: bool = False) -> pd.DataFrame:
        """Load the spot-reassignment matrix for this round."""
        return self.diagnostics.load_reassignment_matrix(normalised=normalised)

    def load_spot_fate_matrix(self) -> pd.DataFrame:
        """Load the spot-fate summary (fixed / reassigned / removed) for this round."""
        return self.diagnostics.load_spot_fate_matrix()

    def load_loss_history(self):
        """Load the optimisation loss history for this round."""
        return self.diagnostics.load_loss_history()

    # ------------------------------------------------------------------
    # Channel → gene map (derived from ds_config)
    # ------------------------------------------------------------------

    def get_spot_channel_gene_map(self) -> Dict[str, str]:
        """
        Return a ``{channel: gene}`` mapping derived from ``ds_config``.

        Overrides the parent implementation to read from ``ds_config``
        instead of a standalone processing manifest, since pairwise rounds
        embed their manifest inside ``ds_config.json``.
        """
        gene_dict = self.processing_manifest.get("gene_dict", {})
        return {ch: info["gene"] for ch, info in gene_dict.items()}

    # ------------------------------------------------------------------
    # Override zarr / segmentation to give helpful errors / delegation
    # ------------------------------------------------------------------

    def load_zarr_channel(self, channel, data_type="fused", pyramid_level=0):
        """
        Delegate to the source dataset if available, otherwise raise a clear error.
        """
        parent: "PairwiseUnmixingDataset" = self.parent_dataset  # type: ignore[assignment]
        if parent is not None and parent.source_dataset is not None:
            return parent.source_dataset.rounds[self.round_key].load_zarr_channel(
                channel, data_type, pyramid_level
            )
        raise AttributeError(
            f"Round {self.round_key} has no zarr data (pairwise unmixing asset). "
            "Pass a source_dataset to create_pairwise_unmixing_dataset() to enable "
            "zarr / segmentation access."
        )

    def load_segmentation_mask(self, resolution_key="0"):
        """Delegate to source dataset if available."""
        parent: "PairwiseUnmixingDataset" = self.parent_dataset  # type: ignore[assignment]
        if parent is not None and parent.source_dataset is not None:
            return parent.source_dataset.rounds[self.round_key].load_segmentation_mask(
                resolution_key
            )
        raise AttributeError(
            f"Round {self.round_key} has no segmentation data (pairwise unmixing asset). "
            "Pass a source_dataset to create_pairwise_unmixing_dataset() to enable access."
        )

    def load_cell_centroids(self):
        """Delegate to source dataset if available."""
        parent: "PairwiseUnmixingDataset" = self.parent_dataset  # type: ignore[assignment]
        if parent is not None and parent.source_dataset is not None:
            return parent.source_dataset.rounds[self.round_key].load_cell_centroids()
        raise AttributeError(
            f"Round {self.round_key} has no centroid data (pairwise unmixing asset). "
            "Pass a source_dataset to create_pairwise_unmixing_dataset() to enable access."
        )

    # ------------------------------------------------------------------
    # repr / dir
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        chan_gene = self.get_spot_channel_gene_map()
        return (
            f"PairwiseUnmixingRound(round_key='{self.round_key}', "
            f"name='{self.name}', "
            f"channels={list(chan_gene.keys())}, "
            f"genes={list(chan_gene.values())})"
        )

    def __dir__(self):
        base = list(super().__dir__())
        extra = [
            "ds_config",
            "unmixing_config",
            "diagnostics",
            "load_filtered_cxg",
            "load_ratios_matrix",
            "load_reassignment_matrix",
            "load_spot_fate_matrix",
            "load_loss_history",
        ]
        return base + [e for e in extra if e not in base]


# ---------------------------------------------------------------------------
# PairwiseUnmixingDataset
# ---------------------------------------------------------------------------


class PairwiseUnmixingDataset(HCRDataset):
    """
    Subclass of ``HCRDataset`` for pairwise-unmixing assets.

    ``self.rounds`` is populated with ``PairwiseUnmixingRound`` objects so
    **all** inherited ``HCRDataset`` methods (``load_all_rounds_spots_mp``,
    ``create_cell_gene_matrix``, ``get_cell_info``, etc.) work without
    modification.

    Zarr / segmentation / metadata calls are transparently delegated to
    ``source_dataset`` when it is provided.

    Parameters
    ----------
    rounds : dict
        ``{round_key: PairwiseUnmixingRound}`` — populated by the factory.
    mouse_id : str
    pairwise_asset_path : Path
        Root folder of the pairwise unmixing asset.
    aggregated_cxg_unmixed : Path or None
        Path to ``unmixed_cell_by_gene_all_rounds.csv``.
    aggregated_cxg_mixed : Path or None
        Path to ``mixed_cell_by_gene_all_rounds.csv``.
    inhibitory_analysis : InhibitoryCellAnalysis or None
    source_dataset : HCRDataset or None
        The original processed ``HCRDataset``.  Used for zarr / segmentation
        delegation.
    metadata : dict, optional
    """

    def __init__(
        self,
        rounds: Dict[str, "PairwiseUnmixingRound"],
        mouse_id: str = None,
        pairwise_asset_path: Path = None,
        aggregated_cxg_unmixed: Optional[Path] = None,
        aggregated_cxg_mixed: Optional[Path] = None,
        inhibitory_analysis: Optional[InhibitoryCellAnalysis] = None,
        source_dataset: Optional[HCRDataset] = None,
        metadata: dict = None,
    ):
        super().__init__(
            rounds=rounds,
            mouse_id=mouse_id,
            metadata=metadata,
        )
        self.pairwise_asset_path = Path(pairwise_asset_path) if pairwise_asset_path else None
        self.aggregated_cxg_unmixed = aggregated_cxg_unmixed
        self.aggregated_cxg_mixed = aggregated_cxg_mixed
        self.inhibitory_analysis = inhibitory_analysis
        self.source_dataset = source_dataset

    # ------------------------------------------------------------------
    # Pairwise-specific loaders
    # ------------------------------------------------------------------

    def load_aggregated_cxg(self, unmixed: bool = True) -> pd.DataFrame:
        """
        Load the pre-built all-rounds cell-by-gene table.

        This is the ``*_cell_by_gene_all_rounds.csv`` file at the root of the
        pairwise unmixing asset.  Each column is a ``{round}-{channel}-{gene}``
        label (e.g. ``R3-561-Crh``); each row is a ``cell_id``.

        Parameters
        ----------
        unmixed : bool
            ``True`` → load unmixed table; ``False`` → mixed.

        Returns
        -------
        pd.DataFrame
            Wide-format cell × gene matrix (``cell_id`` as index).
        """
        path = self.aggregated_cxg_unmixed if unmixed else self.aggregated_cxg_mixed
        label = "unmixed" if unmixed else "mixed"
        if path is None or not path.exists():
            raise FileNotFoundError(
                f"Aggregated {label} CxG not found: {path}"
            )
        df = pd.read_csv(path, index_col=0)
        df.index.name = "cell_id"
        return df

    def load_inhibitory_cells(self, unmixed: bool = True) -> pd.DataFrame:
        """
        Load the inhibitory-cell table from the top-level analysis folder.

        Parameters
        ----------
        unmixed : bool
            ``True`` → ``inhibitory_cells_unmixed/`` folder.

        Returns
        -------
        pd.DataFrame
        """
        if self.inhibitory_analysis is None:
            raise ValueError("No inhibitory cell analysis found in this dataset.")
        return self.inhibitory_analysis.load_inhibitory_cells(unmixed=unmixed)

    def load_cluster_labels(self, unmixed: bool = True) -> pd.DataFrame:
        """Load cluster-label assignments from the inhibitory-cell analysis."""
        if self.inhibitory_analysis is None:
            raise ValueError("No inhibitory cell analysis found in this dataset.")
        return self.inhibitory_analysis.load_cluster_labels(unmixed=unmixed)

    def load_sorted_cell_ids(self, unmixed: bool = True) -> pd.DataFrame:
        """Load sorted-cell-id list from the inhibitory-cell analysis."""
        if self.inhibitory_analysis is None:
            raise ValueError("No inhibitory cell analysis found in this dataset.")
        return self.inhibitory_analysis.load_sorted_cell_ids(unmixed=unmixed)

    def load_unmixing_diagnostics(self, round_key: str) -> PairwiseUnmixingDiagnostics:
        """
        Return the ``PairwiseUnmixingDiagnostics`` object for a given round.

        Parameters
        ----------
        round_key : str
            e.g. ``'R3'``

        Returns
        -------
        PairwiseUnmixingDiagnostics
        """
        if round_key not in self.rounds:
            raise ValueError(f"Round {round_key} not found in dataset.")
        rnd = self.rounds[round_key]
        if not isinstance(rnd, PairwiseUnmixingRound):
            raise TypeError(f"Round {round_key} is not a PairwiseUnmixingRound.")
        return rnd.diagnostics

    # ------------------------------------------------------------------
    # Delegation of zarr / segmentation to source_dataset
    # ------------------------------------------------------------------

    def load_zarr_channel(self, round_key, channel, data_type="fused", pyramid_level=0):
        """Delegate to source_dataset if available."""
        if self.source_dataset is not None:
            return self.source_dataset.load_zarr_channel(
                round_key, channel, data_type, pyramid_level
            )
        raise AttributeError(
            "This PairwiseUnmixingDataset has no source_dataset. "
            "Pass source_dataset= to create_pairwise_unmixing_dataset() for zarr access."
        )

    def load_segmentation_mask(self, round_key, resolution_key="0"):
        """Delegate to source_dataset if available."""
        if self.source_dataset is not None:
            return self.source_dataset.load_segmentation_mask(round_key, resolution_key)
        raise AttributeError(
            "This PairwiseUnmixingDataset has no source_dataset. "
            "Pass source_dataset= to create_pairwise_unmixing_dataset() for segmentation access."
        )

    def load_cell_centroids(self, round_key):
        """Delegate to source_dataset if available."""
        if self.source_dataset is not None:
            return self.source_dataset.load_cell_centroids(round_key)
        raise AttributeError(
            "This PairwiseUnmixingDataset has no source_dataset. "
            "Pass source_dataset= to create_pairwise_unmixing_dataset() for centroid access."
        )

    # ------------------------------------------------------------------
    # summary / repr / dir
    # ------------------------------------------------------------------

    def summary(self):
        """Print a summary of the pairwise unmixing dataset."""
        print("Pairwise Unmixing Dataset")
        print("=" * 40)
        if self.mouse_id:
            print(f"Mouse ID : {self.mouse_id}")
        if self.pairwise_asset_path:
            print(f"Asset    : {self.pairwise_asset_path.name}")
        print(f"Rounds   : {', '.join(self.get_rounds())}")
        print()
        for rk, rnd in self.rounds.items():
            cgmap = rnd.get_spot_channel_gene_map()
            has_filt_u = (
                rnd.spot_files.unmixed_cxg_filtered is not None
                and rnd.spot_files.unmixed_cxg_filtered.exists()
            )
            has_filt_m = (
                rnd.spot_files.mixed_cxg_filtered is not None
                and rnd.spot_files.mixed_cxg_filtered.exists()
            )
            print(
                f"  {rk}: genes={list(cgmap.values())}, "
                f"filtered_cxg=[unmixed={'✓' if has_filt_u else '✗'}, "
                f"mixed={'✓' if has_filt_m else '✗'}]"
            )
        print()
        agg_u = self.aggregated_cxg_unmixed and self.aggregated_cxg_unmixed.exists()
        agg_m = self.aggregated_cxg_mixed and self.aggregated_cxg_mixed.exists()
        print(f"Aggregated CxG : unmixed={'✓' if agg_u else '✗'}, mixed={'✓' if agg_m else '✗'}")
        if self.inhibitory_analysis:
            print(f"Inhibitory analysis : {self.inhibitory_analysis}")
        if self.source_dataset is not None:
            print(f"Source dataset : {self.source_dataset!r}")

    def __repr__(self) -> str:
        rounds_list = list(self.rounds.keys())
        src = repr(self.source_dataset) if self.source_dataset else "None"
        return (
            f"PairwiseUnmixingDataset(mouse_id='{self.mouse_id}', "
            f"rounds={rounds_list}, "
            f"source_dataset={src})"
        )

    def __dir__(self):
        base = list(super().__dir__())
        extra = [
            "pairwise_asset_path",
            "aggregated_cxg_unmixed",
            "aggregated_cxg_mixed",
            "inhibitory_analysis",
            "source_dataset",
            "load_aggregated_cxg",
            "load_inhibitory_cells",
            "load_cluster_labels",
            "load_sorted_cell_ids",
            "load_unmixing_diagnostics",
        ]
        return base + [e for e in extra if e not in base]


# ---------------------------------------------------------------------------
# Internal file-discovery helpers
# ---------------------------------------------------------------------------


def _check(path: Path) -> Optional[Path]:
    """Return ``path`` if it exists, else ``None``."""
    return path if path.exists() else None


def _discover_round_subfolders(
    pairwise_asset_path: Path, mouse_id: str
) -> Dict[str, Path]:
    """
    Auto-discover ``{mouse_id}_R{N}`` sub-folders and return a
    ``{round_key: folder_path}`` dict ordered by round number.

    Parameters
    ----------
    pairwise_asset_path : Path
        Root of the pairwise unmixing asset.
    mouse_id : str
        e.g. ``'782149'``.

    Returns
    -------
    dict
        e.g. ``{'R1': Path('…/782149_R1'), 'R2': Path('…/782149_R2'), …}``
    """
    pattern = re.compile(rf"^{re.escape(mouse_id)}_R(\d+)$")
    found: List[tuple] = []
    for child in pairwise_asset_path.iterdir():
        if not child.is_dir():
            continue
        m = pattern.match(child.name)
        if m:
            found.append((int(m.group(1)), f"R{m.group(1)}", child))

    if not found:
        raise FileNotFoundError(
            f"No round sub-folders matching '{mouse_id}_R{{N}}' found in "
            f"{pairwise_asset_path}"
        )

    found.sort(key=lambda t: t[0])
    return {rk: fp for _, rk, fp in found}


def _parse_spot_files(round_key: str, round_folder: Path) -> SpotFiles:
    """
    Build a ``SpotFiles`` object for one pairwise-unmixing round folder.

    Pairwise spot files use the round number from the key (e.g. ``R3`` → 3)
    in their filenames.
    """
    rnum = round_key[1:]  # 'R3' → '3'

    unmixed_spots = next(round_folder.glob(f"unmixed_spots_R{rnum}*.pkl"), None)
    mixed_spots = next(round_folder.glob(f"mixed_spots_R{rnum}.pkl"), None)

    return SpotFiles(
        unmixed_cxg=_check(round_folder / "unmixed_cell_by_gene.csv"),
        mixed_cxg=_check(round_folder / "mixed_cell_by_gene.csv"),
        unmixed_spots=unmixed_spots,
        mixed_spots=mixed_spots,
        spot_unmixing_stats=_check(round_folder / "spot_unmixing_stats.csv"),
        ratios_file=_check(round_folder / f"{round_key}_ratios.txt"),
        unmixed_cxg_filtered=_check(round_folder / "unmixed_cell_by_gene_filtered.csv"),
        mixed_cxg_filtered=_check(round_folder / "mixed_cell_by_gene_filtered.csv"),
    )


def _parse_diagnostics(round_key: str, round_folder: Path) -> PairwiseUnmixingDiagnostics:
    """Build a ``PairwiseUnmixingDiagnostics`` for one round folder."""
    rk = round_key  # e.g. 'R3'

    plots: Dict[str, Path] = {}
    for png in round_folder.glob("*.png"):
        plots[png.stem] = png

    return PairwiseUnmixingDiagnostics(
        ratios_matrix=_check(round_folder / f"{rk}_ratios_matrix.csv"),
        reassignment_matrix=_check(round_folder / f"{rk}_reassignment_matrix.csv"),
        reassignment_matrix_norm_rows=_check(
            round_folder / f"{rk}_reassignment_matrix_norm_rows.csv"
        ),
        spot_fate_matrix=_check(round_folder / f"{rk}_spot_fate_matrix.csv"),
        loss_history=_check(round_folder / f"{rk}_loss_history.pkl"),
        plots=plots,
    )


def _parse_inhibitory_analysis(
    pairwise_asset_path: Path,
) -> InhibitoryCellAnalysis:
    """Build an ``InhibitoryCellAnalysis`` from the top-level analysis folders."""
    u_dir = pairwise_asset_path / "inhibitory_cells_unmixed"
    m_dir = pairwise_asset_path / "inhibitory_cells_mixed"

    def _plots(folder: Path) -> Dict[str, Path]:
        if not folder.exists():
            return {}
        return {p.stem: p for p in folder.glob("*.png")}

    return InhibitoryCellAnalysis(
        unmixed_inhibitory_cells=_check(u_dir / "unmixed_inhibitory_cells.csv"),
        unmixed_cluster_labels=_check(u_dir / "unmixed_cluster_labels.csv"),
        unmixed_sorted_cell_ids=_check(u_dir / "unmixed_sorted_cell_ids.csv"),
        unmixed_plots=_plots(u_dir),
        mixed_inhibitory_cells=_check(m_dir / "mixed_inhibitory_cells.csv"),
        mixed_cluster_labels=_check(m_dir / "mixed_cluster_labels.csv"),
        mixed_sorted_cell_ids=_check(m_dir / "mixed_sorted_cell_ids.csv"),
        mixed_plots=_plots(m_dir),
    )


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def create_pairwise_unmixing_dataset(
    mouse_id: str,
    pairwise_asset_path: Path,
    source_dataset: Optional[HCRDataset] = None,
    config_path: Optional[Path] = None,
) -> PairwiseUnmixingDataset:
    """
    Build a ``PairwiseUnmixingDataset`` from a pairwise unmixing asset folder.

    The factory auto-discovers ``{mouse_id}_R{N}`` sub-folders, reads
    ``ds_config.json`` from each to recover the processing manifest and gene
    dict, and assembles ``PairwiseUnmixingRound`` objects that slot into the
    parent ``HCRDataset.rounds`` dict.

    Parameters
    ----------
    mouse_id : str
        Mouse identifier, e.g. ``'782149'``.
    pairwise_asset_path : Path
        Absolute path to the root of the pairwise unmixing asset, e.g.
        ``/root/capsule/data/HCR_782149_pairwise-unmixing_2026-03-06_12-00-00``.
    source_dataset : HCRDataset, optional
        The original processed ``HCRDataset`` for this mouse.  When provided,
        zarr / segmentation / metadata calls on the returned dataset (or on
        individual rounds) are transparently delegated to it.
    config_path : Path, optional
        Path to ``MOUSE_HCR_CONFIG.json``.  Currently used only to signal
        intent; full config-driven loading is a TODO (see note below).

    Returns
    -------
    PairwiseUnmixingDataset

    Notes
    -----
    # TODO (config-driven): read the ``"pairwise_unmixing"`` key from
    # ``MOUSE_HCR_CONFIG.json`` to auto-resolve ``pairwise_asset_path`` when
    # ``config_path`` is supplied but ``pairwise_asset_path`` is omitted.
    # Expected config structure::
    #
    #     "782149": {
    #         "rounds": { ... },
    #         "pairwise_unmixing": {
    #             "asset": "HCR_782149_pairwise-unmixing_2026-03-06_12-00-00"
    #         }
    #     }
    """
    pairwise_asset_path = Path(pairwise_asset_path)

    if not pairwise_asset_path.exists():
        raise FileNotFoundError(
            f"Pairwise unmixing asset not found: {pairwise_asset_path}"
        )

    # ------------------------------------------------------------------ #
    # 1. Discover round sub-folders                                        #
    # ------------------------------------------------------------------ #
    round_folders = _discover_round_subfolders(pairwise_asset_path, mouse_id)
    print(f"Found {len(round_folders)} pairwise round(s): {list(round_folders.keys())}")

    # ------------------------------------------------------------------ #
    # 2. Build PairwiseUnmixingRound objects                              #
    # ------------------------------------------------------------------ #
    rounds: Dict[str, PairwiseUnmixingRound] = {}

    for round_key, round_folder in round_folders.items():
        # Load ds_config.json — contains the embedded processing manifest
        ds_config_path = round_folder / "ds_config.json"
        if not ds_config_path.exists():
            warnings.warn(
                f"ds_config.json not found for round {round_key} in {round_folder}. "
                "Processing manifest will be empty."
            )
            ds_config = {}
        else:
            with open(ds_config_path, "r") as f:
                ds_config = json.load(f)

        # The processing manifest is embedded under the "manifest" key
        processing_manifest: dict = ds_config.get("manifest", {})

        # Load unmixing_config.json — pairwise-specific parameters
        unmixing_config_path = round_folder / "unmixing_config.json"
        if not unmixing_config_path.exists():
            warnings.warn(
                f"unmixing_config.json not found for round {round_key} in {round_folder}."
            )
            unmixing_config = {}
        else:
            with open(unmixing_config_path, "r") as f:
                unmixing_config = json.load(f)

        spot_files = _parse_spot_files(round_key, round_folder)
        diagnostics = _parse_diagnostics(round_key, round_folder)

        rounds[round_key] = PairwiseUnmixingRound(
            round_key=round_key,
            name=round_folder.name,
            spot_files=spot_files,
            ds_config=ds_config,
            unmixing_config=unmixing_config,
            diagnostics=diagnostics,
            processing_manifest=processing_manifest,
            parent_dataset=None,  # set below after dataset is created
        )

    # ------------------------------------------------------------------ #
    # 3. Top-level aggregated CxG                                         #
    # ------------------------------------------------------------------ #
    aggregated_cxg_unmixed = _check(
        pairwise_asset_path / "unmixed_cell_by_gene_all_rounds.csv"
    )
    aggregated_cxg_mixed = _check(
        pairwise_asset_path / "mixed_cell_by_gene_all_rounds.csv"
    )

    # ------------------------------------------------------------------ #
    # 4. Inhibitory-cell analysis                                         #
    # ------------------------------------------------------------------ #
    inhibitory_analysis = _parse_inhibitory_analysis(pairwise_asset_path)

    # ------------------------------------------------------------------ #
    # 5. Assemble dataset                                                  #
    # ------------------------------------------------------------------ #
    dataset = PairwiseUnmixingDataset(
        rounds=rounds,
        mouse_id=mouse_id,
        pairwise_asset_path=pairwise_asset_path,
        aggregated_cxg_unmixed=aggregated_cxg_unmixed,
        aggregated_cxg_mixed=aggregated_cxg_mixed,
        inhibitory_analysis=inhibitory_analysis,
        source_dataset=source_dataset,
    )

    # Back-fill parent_dataset reference on each round
    for rnd in dataset.rounds.values():
        rnd.parent_dataset = dataset

    print(f"PairwiseUnmixingDataset ready: {dataset!r}")
    return dataset
