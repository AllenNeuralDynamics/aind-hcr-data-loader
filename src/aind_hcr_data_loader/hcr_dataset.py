# -*- coding: utf-8 -*-
"""
# Create from existing round_dict
dataset = create_hcr_dataset(round_dict, data_dir, mouse_id="747667")

# Or create directly from config
dataset = create_hcr_dataset_from_config("747667")

# Overview
dataset.summary()

# Get cell info
cell_info = dataset.get_cell_info('R1')

# Create cell-gene matrix from all rounds
cxg_matrix = dataset.create_cell_gene_matrix(unmixed=True)

# Lazy Load specific zarr channel
channel_data = dataset.load_zarr_channel('R1', '405')

# Get channel-gene mapping
channel_genes = dataset.create_channel_gene_table()

"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class SpotFiles:
    """
    Data class to hold paths to spot files for unmixed and mixed cell by gene data,
    unmixed and mixed spots, and spot unmixing statistics.
    """

    unmixed_cxg: Path
    mixed_cxg: Path
    unmixed_spots: Path
    mixed_spots: Path
    spot_unmixing_stats: Path
    processing_manifest: Path = None  # Optional, for processing manifest if available


@dataclass
class ZarrDataFiles:
    """
    Data class to hold paths to Zarr data files for fused, corrected, and raw datasets.
    """

    fused: Dict[str, Path]
    corrected: Dict[str, Path] = None
    raw: Dict[str, Path] = None

    def __post_init__(self):
        """Initialize empty dictionaries for corrected and raw if not provided."""
        if self.corrected is None:
            self.corrected = {}
        if self.raw is None:
            self.raw = {}

    def get_channels(self):
        """Get list of available channels."""
        return list(self.fused.keys())

    def has_channel(self, channel):
        """Check if a specific channel exists in fused data."""
        return channel in self.fused


@dataclass
class SpotDetection:
    """
    Data class to hold paths to spot detection files for each channel.
    Contains spots.npy files and channel vs spots comparison files.
    """

    channel: str
    spots_file: Path
    stats_files: Dict[str, Path]


@dataclass
class TileAlignmentFiles:
    """Dataclass to hold paths to tile alignment XML files."""

    raw_single_xml: Optional[Path] = None
    raw_single_tile_subset_xml: Optional[Path] = None
    raw_spot_xml: Optional[Path] = None
    raw_spot_tile_subset_xml: Optional[Path] = None
    pc_xml: Optional[Path] = None
    ip_xml: Optional[Path] = None


@dataclass
class SegmentationFiles:
    """
    Data class to hold paths to segmentation files.
    """

    segmentation_masks: Dict[
        str, Path
    ]  # Dictionary mapping resolution keys to segmentation mask paths
    cell_centroids: Path = None  # Path to cell centroids file

    def __post_init__(self):
        """Initialize empty dictionary if not provided."""
        if self.segmentation_masks is None:
            self.segmentation_masks = {}

    def get_resolutions(self):
        """Get list of available resolution keys."""
        return list(self.segmentation_masks.keys())

    def has_resolution(self, resolution_key):
        """Check if a specific resolution exists."""
        return resolution_key in self.segmentation_masks


class HCRRound:
    """
    A class representing a single round of HCR data, containing all associated files and methods
    for working with that round's data.
    """

    def __init__(
        self,
        round_key: str,
        name: str,
        spot_files: "SpotFiles",
        zarr_files: "ZarrDataFiles",
        segmentation_files: "SegmentationFiles",
        spot_detection_files: Dict[str, "SpotDetection"],
        processing_manifest: dict,
        tile_alignment_files: "TileAlignmentFiles" = None,
    ):
        """
        Initialize HCRRound.

        Parameters:
        -----------
        round_key : str
            The identifier for this round (e.g., 'R1', 'R2')
        name : str
            Name of the round. Folder name where data is stored.
        spot_files : SpotFiles
            Spot files for this round
        zarr_files : ZarrDataFiles
            Zarr files for this round
        segmentation_files : SegmentationFiles, optional
            Segmentation files for this round
        spot_detection_files : Dict[str, SpotDetection], optional
            Spot detection files for this round, mapping channel to SpotDetection
        processing_manifest : dict
            Processing manifest data for this round
        tile_alignment_files : TileAlignmentFiles, optional
            Tile alignment files for this round
        """
        self.round_key = round_key
        self.name = name
        self.spot_files = spot_files
        self.zarr_files = zarr_files
        self.segmentation_files = segmentation_files
        self.spot_detection_files = spot_detection_files
        self.processing_manifest = processing_manifest
        self.tile_alignment_files = tile_alignment_files

    def get_channels(self, data_type="fused"):
        """
        Get list of available channels for this round.

        Parameters:
        -----------
        data_type : str
            Type of data ('fused', 'corrected', 'raw')

        Returns:
        --------
        list
            List of available channels for this round
        """
        return self.zarr_files.get_channels()

    def has_channel(self, channel):
        """Check if a specific channel exists in this round."""
        return self.zarr_files.has_channel(channel)

    def load_zarr_channel(self, channel, data_type="fused", pyramid_level=0):
        """
        Load a specific channel's zarr data for this round.

        Parameters:
        -----------
        channel : str
            Channel identifier
        data_type : str
            Type of data ('fused', 'corrected', 'raw')
        pyramid_level : int
            Pyramid level (0-5), appended to zarr path

        Returns:
        --------
        dask.array.Array
            Loaded zarr array as dask array
        """
        import dask.array as da
        import zarr

        # make py level int
        pyramid_level = int(pyramid_level)

        data_dict = getattr(self.zarr_files, data_type)

        if channel not in data_dict:
            raise ValueError(
                f"Channel {channel} not found in {data_type} data for round {self.round_key}"
            )

        # Validate pyramid level
        if not isinstance(pyramid_level, int) or pyramid_level < 0 or pyramid_level > 5:
            raise ValueError(
                f"Pyramid level must be an integer between 0 and 5, got {pyramid_level}"
            )

        zarr_path = data_dict[channel]
        # Open zarr array at specified pyramid level
        zarr_array = zarr.open(zarr_path, mode="r")[pyramid_level]
        # Convert to dask array for efficient chunked computation
        dask_array = da.from_array(zarr_array, chunks=zarr_array.chunks)
        return dask_array

    def get_segmentation_resolutions(self):
        """Get available segmentation resolutions for this round."""
        if self.segmentation_files is None:
            return []
        return self.segmentation_files.get_resolutions()

    def load_segmentation_mask(self, resolution_key="0"):
        """
        Load segmentation mask for this round at specified resolution.

        Parameters:
        -----------
        resolution_key : str
            Resolution identifier ('0' for segmentation_mask.zarr, '2' for segmentation_mask_orig_res.zarr)

        Returns:
        --------
        zarr.Array
            Loaded segmentation mask
        """
        import zarr

        if self.segmentation_files is None:
            raise ValueError(f"No segmentation files available for round {self.round_key}")

        if resolution_key not in self.segmentation_files.segmentation_masks:
            valid_keys = ", ".join(self.segmentation_files.get_resolutions())
            raise ValueError(
                f"Resolution {resolution_key} not found for round {self.round_key}, valid keys are: {valid_keys}"
            )

        mask_path = self.segmentation_files.segmentation_masks[resolution_key]

        if resolution_key == "0":
            # Load the standard segmentation mask
            return zarr.open(mask_path, mode="r")["0"]
        elif resolution_key == "2":
            # sometimes the original resolution mask is stored in a group....
            zarr_file = zarr.open(mask_path, mode="r")
            if isinstance(zarr_file, zarr.Array):
                return zarr_file
            elif isinstance(zarr_file, zarr.Group):
                if "0" in zarr_file:
                    return zarr_file["0"]
                else:
                    raise ValueError(f"No '0' array found in {mask_path}")

    def load_cell_centroids(self):
        """
        Load cell centroids for this round.

        Returns:
        --------
        numpy.ndarray
            Array of cell centroids
        """
        if self.segmentation_files is None:
            raise ValueError(f"No segmentation files available for round {self.round_key}")

        if (
            self.segmentation_files.cell_centroids is None
            or not self.segmentation_files.cell_centroids.exists()
        ):
            raise ValueError(f"Cell centroids file not found for round {self.round_key}")

        return np.load(self.segmentation_files.cell_centroids)

    def get_cell_info(self):
        """
        Get cell information for this round.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing cell_id, volume, and centroid coordinates
        """
        # Read data for this round
        df = pd.read_csv(self.spot_files.unmixed_cxg)

        # Keep only the columns we want
        cols_to_keep = ["cell_id", "volume", "x_centroid", "y_centroid", "z_centroid"]
        df_cells = df[cols_to_keep].drop_duplicates()

        return df_cells

    def __dir__(self):
        """
        Return a list of valid attributes and methods for this HCRRound.

        This enables better tab completion and introspection.
        Excludes dunder methods and separates attributes from methods.
        """
        # Public attributes specific to HCRRound
        round_attrs = [
            "round_key",
            "name",
            "spot_files",
            "zarr_files",
            "processing_manifest",
            "segmentation_files",
            "spot_detection_files",
        ]

        # Public methods specific to HCRRound
        round_methods = [
            "get_channels",
            "has_channel",
            "load_zarr_channel",
            "get_segmentation_resolutions",
            "load_segmentation_mask",
            "load_cell_centroids",
            "get_cell_info",
        ]

        # Combine attributes first, then methods for organized display
        return round_attrs + round_methods

    def __repr__(self):
        """Return a string representation of the HCRRound object."""
        channels = self.get_channels()
        seg_resolutions = self.get_segmentation_resolutions()
        name_str = f", name='{self.name}'" if self.name else ""
        return (
            f"HCRRound(round_key='{self.round_key}'{name_str}, "
            f"channels={channels}, "
            f"segmentation_resolutions={seg_resolutions})"
        )


class HCRDataset:
    """
    Unified class that contains HCRRound objects for an HCR dataset.
    Provides convenient methods for accessing and working with the complete dataset.
    """

    def __init__(
        self,
        rounds: Dict[str, HCRRound] = None,
        mouse_id: str = None,
        metadata: dict = None,
        dataset_names=None,
    ):
        """
        Initialize HCRDataset.

        Parameters:
        -----------
        rounds : Dict[str, HCRRound], optional
            Dictionary mapping round keys to HCRRound objects
        mouse_id : str, optional
            Mouse ID for metadata
        metadata : dict, optional
            Additional metadata
        dataset_names : optional
            Dataset names (for backward compatibility)
        """
        self.mouse_id = mouse_id
        self.metadata = metadata
        self.dataset_names = dataset_names

        # Initialize rounds
        self.rounds = rounds or {}

        self._validate_rounds()

    def _validate_rounds(self):
        """Validate that rounds have consistent data."""
        if not self.rounds:
            return

        # Check for missing processing manifests
        for round_key, round_obj in self.rounds.items():
            if round_obj.processing_manifest is None:
                print(f"Warning: Processing manifest for round {round_key} is None")

    def get_rounds(self):
        """Get list of available rounds."""
        return list(self.rounds.keys())

    def get_channels(self, round_key=None):
        """
        Get available channels for a specific round or all rounds.

        Parameters:
        -----------
        round_key : str, optional
            Specific round to get channels for. If None, returns dict of all rounds.

        Returns:
        --------
        list or dict
            List of channels for specific round, or dict mapping rounds to channel lists
        """
        if round_key:
            if round_key not in self.rounds:
                raise ValueError(f"Round {round_key} not found")
            return self.rounds[round_key].get_channels()
        else:
            return {k: round_obj.get_channels() for k, round_obj in self.rounds.items()}

    def has_round(self, round_key):
        """Check if dataset contains a specific round."""
        return round_key in self.rounds

    def get_cell_info(self, round_key="R1"):
        """
        Get cell information from a specific round.

        Parameters:
        -----------
        round_key : str
            Round to extract cell info from (default: 'R1')

        Returns:
        --------
        pd.DataFrame
            DataFrame containing cell_id, volume, and centroid coordinates
        """
        if round_key not in self.rounds:
            raise ValueError(f"Round {round_key} not found")

        return self.rounds[round_key].get_cell_info()

    def create_cell_gene_matrix(self, unmixed=True, rounds=None):
        """
        Create cell-gene matrix from specified rounds.

        Parameters:
        -----------
        unmixed : bool
            Whether to use unmixed or mixed data
        rounds : list, optional
            Specific rounds to include. If None, uses all rounds.

        Returns:
        --------
        pd.DataFrame
            Cell-gene matrix
        """
        if rounds is None:
            spot_files = {k: round_obj.spot_files for k, round_obj in self.rounds.items()}
        else:
            spot_files = {k: self.rounds[k].spot_files for k in rounds if k in self.rounds}

        # Load all dataframes once and identify duplicates
        all_genes_by_round = {}
        dataframes = {}  # Store dataframes to avoid re-reading

        for round_key in spot_files.keys():
            if unmixed:
                # Read the unmixed cell-by-gene data
                df = pd.read_csv(spot_files[round_key].unmixed_cxg)
            else:
                df = pd.read_csv(spot_files[round_key].mixed_cxg)

            # Store the dataframe and genes for this round
            dataframes[round_key] = df
            all_genes_by_round[round_key] = set(df["gene"].unique())
            print(f"Round {round_key} has these genes: {df['gene'].unique()}")

        # Find genes that appear in multiple rounds
        all_genes = set()
        for genes in all_genes_by_round.values():
            all_genes.update(genes)

        duplicate_genes = set()
        for gene in all_genes:
            rounds_with_gene = [
                round_key for round_key, genes in all_genes_by_round.items() if gene in genes
            ]
            if len(rounds_with_gene) > 1:
                duplicate_genes.add(gene)
                print(f"Gene '{gene}' appears in rounds: {', '.join(rounds_with_gene)}")
        print(f"Total duplicate genes found: {len(duplicate_genes)}")

        # Process dataframes with appropriate gene naming
        all_rounds_data = []

        for round_key, df in dataframes.items():
            # Create a proper copy to avoid SettingWithCopyWarning
            df_processed = df[["cell_id", "gene", "spot_count"]].copy()

            # Only append round name for genes that appear in multiple rounds
            df_processed.loc[:, "gene"] = df_processed["gene"].apply(
                lambda x: f"{x}_{round_key}" if x in duplicate_genes else x
            )

            # Append to list
            all_rounds_data.append(df_processed)

        # Concatenate all rounds
        stacked_df = pd.concat(all_rounds_data, ignore_index=True)

        # Pivot to get cell_id as index and genes as columns
        pivot_df = stacked_df.pivot(index="cell_id", columns="gene", values="spot_count")

        # Fill NaN values with 0 (genes not detected in certain cells)
        pivot_df = pivot_df.fillna(0)

        return pivot_df

    # TODO: may need dask?
    def load_zarr_channel(self, round_key, channel, data_type="fused", pyramid_level=0):
        """
        Load a specific channel's zarr data.

        Parameters:
        -----------
        round_key : str
            Round identifier
        channel : str
            Channel identifier
        data_type : str
            Type of data ('fused', 'corrected', 'raw')
        pyramid_level : int
            Pyramid level (0-5), appended to zarr path

        Returns:
        --------
        dask.array.Array
            Loaded zarr array as dask array
        """
        if round_key not in self.rounds:
            raise ValueError(f"Round {round_key} not found")

        return self.rounds[round_key].load_zarr_channel(channel, data_type, pyramid_level)

    # WIP: need to make parquet conversion first
    # def query_spots(self, round_key, cell_ids, spot_type='mixed', columns=None):
    #     """
    #     Query spots for specific cells (assuming parquet conversion).

    #     Parameters:
    #     -----------
    #     round_key : str
    #         Round identifier
    #     cell_ids : list
    #         List of cell IDs to query
    #     spot_type : str
    #         'mixed' or 'unmixed'
    #     columns : list, optional
    #         Specific columns to load

    #     Returns:
    #     --------
    #     pd.DataFrame
    #         Filtered spots data
    #     """
    #     if round_key not in self.spot_files:
    #         raise ValueError(f"Round {round_key} not found")

    #     spots_file = getattr(self.spot_files[round_key], f"{spot_type}_spots")
    #     parquet_file = spots_file.with_suffix('.parquet')

    #     if parquet_file.exists():
    #         return query_spots_by_cell_ids(parquet_file, cell_ids, columns)
    #     else:
    #         # Fallback to pickle loading
    #         import pickle as pkl
    #         with open(spots_file, 'rb') as f:
    #             data = pkl.load(f)
    #         if isinstance(data, pd.DataFrame):
    #             return data[data['cell_id'].isin(cell_ids)]
    #        return data

    def create_channel_gene_table(self, spots_only=True):
        """Create channel-gene mapping table from processing manifests."""
        processing_manifests = {
            k: round_obj.processing_manifest for k, round_obj in self.rounds.items()
        }
        return create_channel_gene_table_from_manifests(
            processing_manifests, spots_only=spots_only
        )

    def get_segmentation_resolutions(self, round_key=None):
        """
        Get available segmentation resolutions for a specific round or all rounds.

        Parameters:
        -----------
        round_key : str, optional
            Specific round to get resolutions for. If None, returns dict of all rounds.

        Returns:
        --------
        list or dict
            List of resolution keys for specific round, or dict mapping rounds to resolution lists
        """
        if round_key:
            if round_key not in self.rounds:
                return []
            return self.rounds[round_key].get_segmentation_resolutions()
        else:
            return {
                k: round_obj.get_segmentation_resolutions() for k, round_obj in self.rounds.items()
            }

    def load_segmentation_mask(self, round_key, resolution_key="0"):
        """
        Load segmentation mask for a specific round and resolution.

        Parameters:
        -----------
        round_key : str
            Round identifier
        resolution_key : str
            Resolution identifier ('0' for segmentation_mask.zarr, '2' for segmentation_mask_orig_res.zarr)

        Returns:
        --------
        zarr.Array
            Loaded segmentation mask
        """
        if round_key not in self.rounds:
            raise ValueError(f"Round {round_key} not found")

        return self.rounds[round_key].load_segmentation_mask(resolution_key)

    def load_cell_centroids(self, round_key):
        """
        Load cell centroids for a specific round.

        Parameters:
        -----------
        round_key : str
            Round identifier

        Returns:
        --------
        numpy.ndarray
            Array of cell centroids
        """
        if round_key not in self.rounds:
            raise ValueError(f"Round {round_key} not found")

        return self.rounds[round_key].load_cell_centroids()

    def _print_basic_info(self):
        """Print basic dataset information."""
        print("HCR Dataset Summary")
        print("==================")
        if self.mouse_id:
            print(f"Mouse ID: {self.mouse_id}")
        print(f"Rounds: {', '.join(self.get_rounds())}")
        print("\nChannels by round:")
        for round_key, channels in self.get_channels().items():
            print(f"  {round_key}: {', '.join(channels)}")

    def _print_segmentation_info(self):
        """Print segmentation file information."""
        segmentation_rounds = {
            k: round_obj
            for k, round_obj in self.rounds.items()
            if round_obj.segmentation_files is not None
        }
        if not segmentation_rounds:
            return
        print("\nSegmentation files by round:")
        for round_key, round_obj in segmentation_rounds.items():
            resolutions = round_obj.get_segmentation_resolutions()
            centroids_exist = (
                round_obj.segmentation_files.cell_centroids
                and round_obj.segmentation_files.cell_centroids.exists()
            )
            print(
                f"  {round_key}: resolutions {', '.join(resolutions)}, centroids: {'✓' if centroids_exist else '✗'}"
            )

    def _print_spot_detection_info(self):
        """Print spot detection information."""
        detection_rounds = {
            k: round_obj for k, round_obj in self.rounds.items() if round_obj.spot_detection_files
        }
        if not detection_rounds:
            return
        print("\nSpot detection files by round:")
        for round_key, round_obj in detection_rounds.items():
            channels = list(round_obj.spot_detection_files.keys())
            print(f"  {round_key}: channels {', '.join(channels)}")
            for channel, spot_detection in round_obj.spot_detection_files.items():
                spots_exist = spot_detection.spots_file and spot_detection.spots_file.exists()
                stats_count = len([f for f in spot_detection.stats_files.values() if f.exists()])
                print(
                    f"    {channel}: spots {'✓' if spots_exist else '✗'}, stats files: {stats_count}"
                )

    def _print_tile_alignment_info(self):
        """Print tile alignment file information."""
        alignment_rounds = {
            k: round_obj
            for k, round_obj in self.rounds.items()
            if round_obj.tile_alignment_files is not None
        }
        if not alignment_rounds:
            return
        print("\nTile alignment files by round:")
        for round_key, round_obj in alignment_rounds.items():
            xml_files = round_obj.tile_alignment_files
            print(f"  {round_key}:")
            for file_type, file_path in xml_files.__dict__.items():
                if file_path:
                    print(f"    {file_type}: {file_path.name}")

    def _print_file_status(self):
        """Print file existence status."""
        print("\nFile Status:")
        for round_key, round_obj in self.rounds.items():
            print(f"  {round_key}:")

            spot_files_exist = [
                f
                for f in [round_obj.spot_files.mixed_spots, round_obj.spot_files.unmixed_spots]
                if f and f.exists()
            ]
            print(f"    Spot files: {len(spot_files_exist)} of 2 exist")

            zarr_files_exist = [f for f in round_obj.zarr_files.fused.values() if f.exists()]
            print(
                f"    Zarr files: {len(zarr_files_exist)} of {len(round_obj.zarr_files.fused)} exist"
            )

            if round_obj.segmentation_files:
                mask_count = len(
                    [
                        f
                        for f in round_obj.segmentation_files.segmentation_masks.values()
                        if f.exists()
                    ]
                )
                total_masks = len(round_obj.segmentation_files.segmentation_masks)
                print(f"    Segmentation: {mask_count} of {total_masks} masks exist")

            if round_obj.spot_detection_files:
                detections = [
                    sd
                    for sd in round_obj.spot_detection_files.values()
                    if sd.spots_file and sd.spots_file.exists()
                ]
                detection_count = len(detections)
                print(
                    f"    Spot detection: {detection_count} of {len(round_obj.spot_detection_files)} channels exist"
                )

            if round_obj.tile_alignment_files:
                xml_files = round_obj.tile_alignment_files
                print(
                    f"    Tile alignment files: {', '.join(f.name for f in xml_files.__dict__.values() if f)}"
                )

    def summary(self):
        """Print a summary of the dataset."""
        self._print_basic_info()
        self._print_segmentation_info()
        self._print_spot_detection_info()
        self._print_tile_alignment_info()
        self._print_file_status()

    def __dir__(self):
        """
        Return a list of valid attributes and methods for this HCRDataset.

        This enables better tab completion and introspection.
        Excludes dunder methods and separates attributes from methods.
        """
        # Public attributes specific to HCRDataset
        dataset_attrs = ["rounds", "mouse_id", "metadata", "dataset_names"]

        # Public methods specific to HCRDataset
        dataset_methods = [
            "get_rounds",
            "get_channels",
            "has_round",
            "get_cell_info",
            "create_cell_gene_matrix",
            "load_zarr_channel",
            "create_channel_gene_table",
            "get_segmentation_resolutions",
            "load_segmentation_mask",
            "load_cell_centroids",
            "summary",
        ]

        # Combine attributes first, then methods for organized display
        return dataset_attrs + dataset_methods

    def __repr__(self):
        """Return a string representation of the HCRDataset object."""
        rounds_list = list(self.rounds.keys())
        total_channels = sum(len(round_obj.get_channels()) for round_obj in self.rounds.values())
        return (
            f"HCRDataset(mouse_id='{self.mouse_id}', "
            f"rounds={rounds_list}, "
            f"total_channels={total_channels})"
        )


# ------------------------------------------------------------------------------------------------
# Helper functions for creating HCRDataset
# ------------------------------------------------------------------------------------------------


def create_hcr_dataset(round_dict: dict, data_dir: Path, mouse_id: str = None):
    """
    Create a complete HCRDataset from round dictionary and data directory.

    Parameters:
    -----------
    round_dict : dict
        Dictionary mapping round keys to folder names
    data_dir : Path
        Path to the directory containing round folders
    mouse_id : str, optional
        Mouse ID for metadata

    Returns:
    --------
    HCRDataset
        Complete dataset object
    """
    spot_files = get_spot_files(round_dict, data_dir)
    zarr_files = get_zarr_files(round_dict, data_dir)
    processing_manifests = get_processing_manifests(round_dict, data_dir)
    segmentation_files = get_segmentation_files(round_dict, data_dir)
    spot_detection_files = get_spot_detection_files(round_dict, data_dir)
    tile_alignment_files = get_tile_alignment_files(round_dict, data_dir)

    # Create HCRRound objects
    rounds = {}
    for round_key, folder_name in round_dict.items():
        rounds[round_key] = HCRRound(
            round_key=round_key,
            name=folder_name,
            spot_files=spot_files[round_key],
            zarr_files=zarr_files[round_key],
            processing_manifest=processing_manifests.get(round_key, {}),
            segmentation_files=segmentation_files.get(round_key),
            spot_detection_files=spot_detection_files.get(round_key, {}),
            tile_alignment_files=tile_alignment_files.get(round_key),
        )

    # Load metadata if available
    metadata = None
    if mouse_id:
        try:
            metadata = load_mouse_config(mouse_id=mouse_id)
        except FileNotFoundError:
            print(f"Could not load metadata for mouse {mouse_id}")

    return HCRDataset(
        rounds=rounds,
        mouse_id=mouse_id,
        metadata=metadata,
    )


def create_hcr_dataset_from_config(
    mouse_id: str = "747667", data_dir: Path = None, config_path: Path = None
) -> HCRDataset:
    """
    Create HCRDataset directly from mouse configuration.

    Parameters:
    -----------
    mouse_id : str
        Mouse ID to load
    data_dir : Path, optional
        Override data directory from config

    Returns:
    --------
    HCRDataset
        Complete dataset object
    """
    config = load_mouse_config(config_path=config_path, mouse_id=mouse_id)
    round_dict = config["rounds"]

    if data_dir is None:
        data_dir = Path(config.get("data_dir", "../data"))

    return create_hcr_dataset(round_dict, data_dir, mouse_id)


def load_mouse_config(mouse_id: str, config_path: Path = None) -> dict:
    """
    Load mouse configuration from JSON file.

    Parameters:
    -----------
    config_path : Path, optional
        Path to the mouse configuration JSON file. If None, uses default location.
    mouse_id : str
        Mouse ID to load configuration for.

    Returns:
    --------
    dict
        Configuration dictionary containing rounds and metadata for the specified mouse.
    """
    if config_path is None:
        config_path = Path(__file__).parent / "MOUSE_HCR_CONFIG.json"

    with open(config_path, "r") as f:
        config = json.load(f)

    if mouse_id not in config:
        raise ValueError(f"Mouse ID {mouse_id} not found in configuration")

    return config[mouse_id]


def get_cell_info_r1(spot_files, round_key="R1"):
    """
    Get unique cell IDs and their spatial information from round 1.

    Parameters:
    -----------
    spot_files : dict
        Dictionary mapping round keys to SpotFiles objects

    Returns:
    --------
    pd.DataFrame
        DataFrame containing cell_id, volume, and centroid coordinates from R1
    """
    # Read R1 data
    df_r1 = pd.read_csv(spot_files[round_key].unmixed_cxg)

    # Keep only the columns we want
    cols_to_keep = ["cell_id", "volume", "x_centroid", "y_centroid", "z_centroid"]
    df_cells = df_r1[cols_to_keep].drop_duplicates()

    return df_cells


def create_channel_gene_table(spot_files: dict, spots_only=True) -> pd.DataFrame:
    """
    Create a table of Channel, Gene, and Round from the "gene_dict" key in the processing manifest for each round.

    Parameters:
    -----------
    spot_files : dict
        Dictionary mapping round keys to SpotFiles objects.

    Returns:
    --------
    pd.DataFrame
        DataFrame containing columns: Channel, Gene, and Round.
    """
    data = []

    for round_key, spot_file in spot_files.items():
        if spot_file.processing_manifest:
            manifest = load_processing_manifest(spot_file.processing_manifest)
            gene_dict = manifest.get("gene_dict", {})

            for channel, details in gene_dict.items():
                data.append(
                    {"Channel": channel, "Gene": details.get("gene", ""), "Round": round_key}
                )

    # sort by round then channel
    data.sort(key=lambda x: (x["Round"], x["Channel"]))

    if spots_only:
        # drop Channel = 405 and Gene = Syto59
        data = [
            entry
            for entry in data
            if not (entry["Channel"] == "405" and entry["Gene"] == "Syto59")
        ]
    # for duplicate genes, append the round name to the gene
    for entry in data:
        if entry["Gene"] in [d["Gene"] for d in data if d["Round"] != entry["Round"]]:
            entry["Gene"] += f"-{entry['Round']}"
    return pd.DataFrame(data)


def create_channel_gene_table_from_manifests(
    processing_manifests: Dict[str, dict], spots_only=True
) -> pd.DataFrame:
    """
    Create a table of Channel, Gene, and Round from the "gene_dict" key in the processing manifests for each round.

    Parameters:
    -----------
    processing_manifests : Dict[str, dict]
        Dictionary mapping round keys to processing manifest dictionaries.
    spots_only : bool, optional
        If True, exclude Channel=405 and Gene=Syto59 entries (default: True)

    Returns:
    --------
    pd.DataFrame
        DataFrame containing columns: Channel, Gene, and Round.
    """
    data = []

    for round_key, manifest in processing_manifests.items():
        gene_dict = manifest.get("gene_dict", {})

        for channel, details in gene_dict.items():
            data.append({"Channel": channel, "Gene": details.get("gene", ""), "Round": round_key})

    # Sort by round then channel
    data.sort(key=lambda x: (x["Round"], x["Channel"]))

    if spots_only:
        # Drop Channel = 405 and Gene = Syto59
        data = [
            entry
            for entry in data
            if not (entry["Channel"] == "405" and entry["Gene"] == "Syto59")
        ]

    # For duplicate genes, append the round name to the gene
    for entry in data:
        if entry["Gene"] in [d["Gene"] for d in data if d["Round"] != entry["Round"]]:
            entry["Gene"] += f"-{entry['Round']}"

    return pd.DataFrame(data)


# ------------------------------------------------------------------------------------------------
# File retrieval functions
# ------------------------------------------------------------------------------------------------


def get_segmentation_files(round_dict: dict, data_dir: Path):
    """
    Get SegmentationFiles for each round based on a dictionary mapping round keys to folder names.

    Parameters:
    -----------
    round_dict : dict
        Dictionary mapping round keys (e.g., 'R1', 'R2') to folder names containing the data.
    data_dir : Path
        Path to the directory containing the round folders.

    Returns:
    --------
    dict
        Dictionary mapping round keys to SegmentationFiles objects containing paths to segmentation files.
    """
    segmentation_files = {}

    for key, folder in round_dict.items():
        folder_path = data_dir / folder / "cell_body_segmentation"

        # Look for segmentation mask files
        segmentation_masks = {}

        # Check for segmentation_mask.zarr (resolution key '0')
        mask_path = folder_path / "segmentation_mask.zarr"
        if mask_path.exists():
            segmentation_masks["0"] = mask_path

        # Check for segmentation_mask_orig_res.zarr (resolution key '2')
        mask_orig_res_path = folder_path / "segmentation_mask_orig_res.zarr"
        # Also check for alternate name: segmentation_mask_transformed_level_2.zarr
        mask_transformed_level2_path = folder_path / "segmentation_mask_transformed_level_2.zarr"

        if mask_orig_res_path.exists():
            segmentation_masks["2"] = mask_orig_res_path
        elif mask_transformed_level2_path.exists():
            segmentation_masks["2"] = mask_transformed_level2_path

        # Check for cell centroids
        centroids_path = folder_path / "cell_centroids.npy"
        if not centroids_path.exists():
            centroids_path = None

        segmentation_files[key] = SegmentationFiles(
            segmentation_masks=segmentation_masks, cell_centroids=centroids_path
        )

    return segmentation_files


def get_tile_alignment_files(round_dict: dict, data_dir: Path):
    """
    Get TileAlignmentFiles for each round based on a dictionary mapping round keys to folder names.
    Parameters:
    -----------
    round_dict : dict
        Dictionary mapping round keys (e.g., 'R1', 'R2') to folder names containing the data.
    data_dir : Path
        Path to the directory containing the round folders.
    Returns:
    --------
    dict
        Dictionary mapping round keys to TileAlignmentFiles objects containing paths to alignment files.
    """
    tile_alignment_files = {}
    for key, folder in round_dict.items():
        stitching_path = data_dir / folder / "image_tile_fusing" / "metadata" / "stitching"

        def check_exist(path):
            """check exist"""
            return path if path.exists() else None

        tile_alignment_files[key] = TileAlignmentFiles(
            raw_single_xml=check_exist(
                stitching_path / "stitching_single_channel_updated_remote.xml"
            ),
            raw_single_tile_subset_xml=check_exist(
                stitching_path / "stitching_single_channel_updated_tile_subset_remote.xml"
            ),
            raw_spot_xml=check_exist(
                stitching_path / "stitching_spot_channels_updated_remote.xml"
            ),
            raw_spot_tile_subset_xml=check_exist(
                stitching_path / "stitching_spot_channels_updated_tile_subset_remote.xml"
            ),
            pc_xml=check_exist(stitching_path / "phase_correlation_stitching" / "bigstitcher.xml"),
            ip_xml=check_exist(stitching_path / "interest_point_stitching" / "bigstitcher_0.xml"),
        )
    return tile_alignment_files


def get_spot_detection_files(round_dict: dict, data_dir: Path):
    """
    Get SpotDetection objects for each round and channel based on a dictionary mapping round keys to folder names.

    Parameters:
    -----------
    round_dict : dict
        Dictionary mapping round keys (e.g., 'R1', 'R2') to folder names containing the data.
    data_dir : Path
        Path to the directory containing the round folders.

    Returns:
    --------
    dict
        Dictionary mapping round keys to dictionaries of channel keys to SpotDetection objects.
        Structure: {round_key: {channel: SpotDetection, ...}, ...}
    """
    spot_detection_files = {}

    for round_key, folder in round_dict.items():
        spot_detection_path = data_dir / folder / "image_spot_detection"

        if not spot_detection_path.exists():
            print(f"Warning: spot detection folder not found for round {round_key}")
            continue

        round_channels = {}

        # Find all channel directories
        channel_dirs = [
            d
            for d in spot_detection_path.iterdir()
            if d.is_dir() and d.name.startswith("channel_") and d.name.endswith("_spots")
        ]

        for channel_spots_dir in channel_dirs:
            # Extract channel number from directory name (e.g., 'channel_488_spots' -> '488')
            channel = channel_spots_dir.name.replace("channel_", "").replace("_spots", "")

            # Get spots.npy file
            spots_file = channel_spots_dir / "spots.npy"

            # Get corresponding stats directory
            channel_stats_dir = spot_detection_path / f"channel_{channel}_stats"

            # Find all channel vs spots comparison files
            stats_files = {}
            if channel_stats_dir.exists():
                for stats_file in channel_stats_dir.glob(
                    "image_data_channel_*_versus_spots_*.csv"
                ):
                    # Extract the comparison channel from filename
                    # e.g., 'image_data_channel_488_versus_spots_514.csv' -> '514'
                    filename_parts = stats_file.stem.split("_")
                    if len(filename_parts) >= 6:  # Expected format has at least 6 parts
                        comparison_channel = filename_parts[
                            -1
                        ]  # Last part is the comparison channel
                        stats_files[comparison_channel] = stats_file

            # Create SpotDetection object
            round_channels[channel] = SpotDetection(
                channel=channel, spots_file=spots_file, stats_files=stats_files
            )

        spot_detection_files[round_key] = round_channels

    return spot_detection_files


def get_spot_files(round_dict: dict, data_dir: Path):
    """Get SpotFiles for each round based on a dictionary mapping round keys to folder names.
    Parameters:
    -----------
    round_dict : dict
        Dictionary mapping round keys (e.g., 'R1', 'R2') to folder names containing the data.
    data_dir : Path
        Path to the directory containing the round folders.
    Returns:
    --------
    dict
        Dictionary mapping round keys to SpotFiles objects containing paths to relevant files.
    """
    # Build a dict mapping round keys to RoundFiles
    spot_files = {}
    for key, folder in round_dict.items():
        folder_path = data_dir / folder / "image_spot_spectral_unmixing"
        unmixed_cxg = folder_path / "unmixed_cell_by_gene.csv"
        mixed_cxg = folder_path / "mixed_cell_by_gene.csv"
        # Expect only one file for each pattern
        unmixed_spots = next(folder_path.absolute().glob("unmixed_spots_*.pkl"), None)
        mixed_spots = next(folder_path.absolute().glob("mixed_spots_*.pkl"), None)
        stats = folder_path / "spot_unmixing_stats.csv"
        spot_files[key] = SpotFiles(
            unmixed_cxg=unmixed_cxg,
            mixed_cxg=mixed_cxg,
            unmixed_spots=unmixed_spots,
            mixed_spots=mixed_spots,
            spot_unmixing_stats=stats,
        )

        processing_manifest = data_dir / folder / "derived" / "processing_manifest.json"
        if processing_manifest.exists():
            spot_files[key].processing_manifest = processing_manifest

    # # Check if all required files exist
    # for key, files in spot_files.items():
    #     if not all(file.exists() for file in files.__dict__.values()):
    #         raise FileNotFoundError(f"Missing required files for round {key} in {data_dir}")
    return spot_files


def get_zarr_files(round_dict: dict, data_dir: Path):
    """
    Get ZarrDataFiles for each round based on a dictionary mapping round keys to folder names.

    Parameters:
    -----------
    round_dict : dict
        Dictionary mapping round keys (e.g., 'R1', 'R2') to folder names containing the data.
    data_dir : Path
        Path to the directory containing the round folders.

    Returns:
    --------
    dict
        Dictionary mapping round keys to ZarrDataFiles objects containing paths to zarr files.
    """
    zarr_files = {}

    for key, folder in round_dict.items():
        folder_path = data_dir / folder / "image_tile_fusing"

        # Find fused zarr files
        fused_dir = folder_path / "fused"
        fused_channels = {}

        if fused_dir.exists():
            # Look for channel_*.zarr files
            for zarr_file in fused_dir.glob("channel_*.zarr"):
                # Extract channel number from filename (e.g., "channel_405.zarr" -> "405")
                channel = zarr_file.stem.split("_")[1]
                fused_channels[channel] = zarr_file

        # Initialize corrected and raw as empty dicts (can be populated later)
        corrected_channels = {}
        raw_channels = {}

        # Look for corrected zarr files if directory exists
        corrected_dir = folder_path / "corrected"
        if corrected_dir.exists():
            for zarr_file in corrected_dir.glob("channel_*.zarr"):
                channel = zarr_file.stem.split("_")[1]
                corrected_channels[channel] = zarr_file

        # Look for raw zarr files if directory exists
        raw_dir = folder_path / "raw"
        if raw_dir.exists():
            for zarr_file in raw_dir.glob("channel_*.zarr"):
                channel = zarr_file.stem.split("_")[1]
                raw_channels[channel] = zarr_file

        zarr_files[key] = ZarrDataFiles(
            fused=fused_channels,
            corrected=corrected_channels if corrected_channels else {},
            raw=raw_channels if raw_channels else {},
        )

    return zarr_files


def get_all_files(round_dict: dict, data_dir: Path):
    """
    Get both SpotFiles and ZarrDataFiles for each round.

    Parameters:
    -----------
    round_dict : dict
        Dictionary mapping round keys to folder names.
    data_dir : Path
        Path to the directory containing the round folders.

    Returns:
    --------
    tuple
        (spot_files, zarr_files) - dictionaries mapping round keys to respective file objects
    """
    spot_files = get_spot_files(round_dict, data_dir)
    zarr_files = get_zarr_files(round_dict, data_dir)

    return spot_files, zarr_files


# ------------------------------------------------------------------------------------------------
# Loading functions
# ------------------------------------------------------------------------------------------------


def load_zarr_channel(zarr_files, round_key, channel, data_type="fused"):
    """
    Load a specific channel's zarr data.

    Parameters:
    -----------
    zarr_files : dict
        Dictionary mapping round keys to ZarrDataFiles objects
    round_key : str
        Round identifier (e.g., 'R1')
    channel : str
        Channel identifier (e.g., '405')
    data_type : str
        Type of data to load ('fused', 'corrected', 'raw')

    Returns:
    --------
    zarr.Array
        Loaded zarr array
    """
    import zarr

    if round_key not in zarr_files:
        raise ValueError(f"Round {round_key} not found in zarr_files")

    files = zarr_files[round_key]
    data_dict = getattr(files, data_type)

    if channel not in data_dict:
        raise ValueError(f"Channel {channel} not found in {data_type} data for round {round_key}")

    zarr_path = data_dict[channel]
    return zarr.open(zarr_path, mode="r")


def load_processing_manifest(manifest_path: Path) -> dict:
    """
    Load the processing_manifest.json file into a dictionary.

    Parameters:
    -----------
    manifest_path : Path
        Path to the processing_manifest.json file.

    Returns:
    --------
    dict
        Dictionary containing the contents of the processing_manifest.json file.
    """
    with open(manifest_path, "r") as file:
        return json.load(file)


def get_processing_manifests(round_dict: dict, data_dir: Path):
    """
    Get processing manifests for each round based on a dictionary mapping round keys to folder names.

    Parameters:
    -----------
    round_dict : dict
        Dictionary mapping round keys (e.g., 'R1', 'R2') to folder names containing the data.
    data_dir : Path
        Path to the directory containing the round folders.

    Returns:
    --------
    dict
        Dictionary mapping round keys to loaded processing manifest dictionaries.

    Raises:
    -------
    AssertionError
        If any processing manifest is not found
    """
    processing_manifests = {}

    for key, folder in round_dict.items():
        manifest_path = data_dir / folder / "derived" / "processing_manifest.json"

        if not manifest_path.exists():
            raise FileNotFoundError(f"Processing manifest not found at {manifest_path}")

        processing_manifests[key] = load_processing_manifest(manifest_path)

    return processing_manifests
