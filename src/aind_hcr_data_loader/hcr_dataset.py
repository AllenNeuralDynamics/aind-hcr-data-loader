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
import multiprocessing as mp
import pickle as pkl
import warnings
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import pandas as pd


# ------------------------------------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------------------------------------

def parse_genotype(genotype_string: str) -> dict:
    """
    Parse a genotype string into structured components.
    
    Parameters:
    -----------
    genotype_string : str
        Genotype string like 'Slc32a1-IRES-Cre/wt;Oi1(TIT2L-jGCaMP8s-WPRE-ICL-IRES-tTA2)/wt'
        
    Returns:
    --------
    dict
        Dictionary containing parsed genotype components:
        - driver: First part before semicolon
        - reporter: Second part after semicolon  
        - gcamp: Boolean indicating if gcamp is present in reporter
        - genotype_short: Short form like 'Slc32a1-Oi1'
    """
    if not genotype_string or not isinstance(genotype_string, str):
        return {
            'driver': None,
            'reporter': None,
            'gcamp': None,
            'genotype_short': None
        }
    
    # Split by semicolon
    parts = genotype_string.split(';')
    
    driver = parts[0].strip() if len(parts) > 0 else None
    reporter = parts[1].strip() if len(parts) > 1 else None
    
    # Extract gcamp substring from reporter (case insensitive)
    gcamp = None
    if reporter:
        import re
        # Look for gcamp followed by any characters (typically version like 8s, 7f, etc.)
        gcamp_match = re.search(r'(jGCaMP\w*)', reporter, re.IGNORECASE)
        if gcamp_match:
            gcamp = gcamp_match.group(1)
    
    # Create short genotype
    genotype_short = None
    if driver and reporter:
        # Extract first part of driver (before first '-' or '/')
        driver_parts = driver.split('-')
        driver_short = driver_parts[0] if driver_parts else driver
        
        # Extract reporter part before parentheses
        if '(' in reporter:
            reporter_short = reporter.split('(')[0]
        else:
            reporter_short = reporter.split('-')[0] if '-' in reporter else reporter.split('/')[0]
        
        genotype_short = f"{driver_short}-{reporter_short}"
    
    return {
        'driver': driver,
        'reporter': reporter,
        'gcamp': gcamp,
        'genotype_short': genotype_short
    }


# ------------------------------------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------------------------------------


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
    ratios_file: Path = None  # Optional, for ratios if available
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
class MetadataFiles:
    """
    Data class to hold paths to metadata JSON files for an HCR dataset round.
    """
    
    acquisition: Optional[Path] = None
    data_description: Optional[Path] = None
    metadata_nd: Optional[Path] = None
    procedures: Optional[Path] = None
    processing: Optional[Path] = None
    subject: Optional[Path] = None
    
    def __post_init__(self):
        """Initialize after creation."""
        pass
    
    def get_available_metadata(self):
        """Get list of available metadata files."""
        available = []
        for attr_name, file_path in self.__dict__.items():
            if file_path is not None and file_path.exists():
                available.append(attr_name)
        return available
    
    def has_metadata(self, metadata_type):
        """Check if a specific metadata type exists."""
        file_path = getattr(self, metadata_type, None)
        return file_path is not None and file_path.exists()


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
    metrics_path: Path = None

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
        metadata_files: "MetadataFiles" = None,
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
        metadata_files : MetadataFiles, optional
            Metadata JSON files for this round
        """
        self.round_key = round_key
        self.name = name
        self.spot_files = spot_files
        self.zarr_files = zarr_files
        self.segmentation_files = segmentation_files
        self.spot_detection_files = spot_detection_files
        self.processing_manifest = processing_manifest
        self.tile_alignment_files = tile_alignment_files
        self.metadata_files = metadata_files

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

    def get_cell_info(self, source="mixed_cxg"):
        """
        Get cell information.

        segmentation:
            contains every mask from segmentation,
            columns: cell_id, x_centroid, y_centroid, z_centroid
        mixed_cxg:
            contains subset of masks, that had a detected spot
            columns: cell_id, volume, x_centroid, y_centroid, z_centroid

        Parameters:
        -----------
        source : str
            Source of cell information ('mixed_cxg' or 'unmixed_cxg')

        Returns:
        --------
        pd.DataFrame
            DataFrame containing cell_id, volume, and centroid coordinates
        """
        # Read data for this round
        if source not in ["mixed_cxg", "unmixed_cxg", "segmentation"]:
            raise ValueError("Source must be 'mixed_cxg', 'unmixed_cxg', or 'segmentation'")

        if source == "unmixed_cxg":
            print(f"Loading unmixed cxg for round {self.round_key}")
            try:
                df = pd.read_csv(self.spot_files.unmixed_cxg)
            except Exception as e:
                print(f"Warning: Error reading unmixed cxg file: {e}")
                return pd.DataFrame()  # Return empty DataFrame if file does not exist

            # add warning, getting cell info from unmixed cxg
            warnings.warn(
                "Getting cell info from unmixed cxg file. Does not include all segmentation masks."
            )

            # Keep only the columns we want
            cols_to_keep = ["cell_id", "volume", "x_centroid", "y_centroid", "z_centroid"]
            df_cells = df[cols_to_keep].drop_duplicates()
        elif source == "mixed_cxg":
            print(f"Loading mixed cxg for round {self.round_key}")
            try:
                df = pd.read_csv(self.spot_files.mixed_cxg)
            except Exception as e:
                print(f"Warning: Error reading mixed cxg file: {e}")
                return pd.DataFrame()  # Return empty DataFrame if file does not exist

            # Keep only the columns we want
            cols_to_keep = ["cell_id", "volume", "x_centroid", "y_centroid", "z_centroid"]
            df_cells = df[cols_to_keep].drop_duplicates()
        elif source == "segmentation":
            # load centroids from segmentation files
            centroids = self.load_cell_centroids()
            df_cells = pd.DataFrame(
                centroids, columns=["z_centroid", "y_centroid", "x_centroid", "cell_id"]
            )
            # Retain the original cell_id values from the centroids data
        print(f"Number of cells in {source} for round {self.round_key}: {len(df_cells)}")
        return df_cells

    def load_spots(self, 
                  table_type: str ="mixed", 
                  remove_fg_bg_cols: bool = True, 
                  filter_cell_ids: Optional[List] = None):
        """
        Load spots for this round (non-multiprocessing version).

        Parameters:
        -----------
        table_type : str
            Type of spots to load ('mixed'/'mixed_spots' or 'unmixed'/unmixed_spots')
        remove_fg_bg_cols : bool
            Whether to remove columns containing 'fg' or 'bg' substrings to save space
        filter_cell_ids : list, optional
            If provided, filter spots to only include these cell IDs

        Returns:
        --------
        pd.DataFrame
            DataFrame with spots data including round column
        """
        print(f"Loading {table_type} for round {self.round_key}")

        if table_type not in ["mixed", "unmixed", "mixed_spots", "unmixed_spots"]:
            raise ValueError("table_type must be 'mixed' or 'unmixed'")

        if table_type == "mixed":
            table_type = "mixed_spots"
        else:
            table_type = "unmixed_spots"

        spot_file_path = getattr(self.spot_files, table_type)

        with open(spot_file_path, "rb") as f:
            spots_data = pd.read_pickle(f)
            spots_data["round"] = self.round_key

        # set 'chan' 
        categorical_cols = ['chan', 'round', 'unmixed_chan']

        for col in categorical_cols:
            if col in spots_data.columns:
                spots_data[col] = spots_data[col].astype('category')

        # remove cols with 'fg' or 'bg' substrings, save space
        if remove_fg_bg_cols:
            cols_to_remove = [col for col in spots_data.columns if 'fg' in col or 'bg' in col]
            spots_data = spots_data.drop(columns=cols_to_remove)

        # convert float64 to float32 to save space
        float_cols = spots_data.select_dtypes(include=['float64']).columns
        spots_data[float_cols] = spots_data[float_cols].astype('float32')

        spots_data = spots_data.drop(columns=["spot_id"])
        spots_data = spots_data.reset_index(drop=True)

        # Filter by cell_ids if provided
        if filter_cell_ids is not None:
            spots_data = spots_data[spots_data["cell_id"].isin(filter_cell_ids)].reset_index(drop=True)
            print(f"Filtered spots to {len(spots_data)} entries based on provided cell IDs")

        return spots_data

    def get_spot_channel_gene_map(self):
        """
        Extract channel-gene mapping for spot channels from this round.
        
        Returns:
        --------
        dict
            Mapping of channel to gene name for spot channels only
        """
        spot_channels = self.processing_manifest['spot_channels']
        gene_dict = self.processing_manifest['gene_dict']

        # make sure spot_channels are str, ints in dummy manifest
        spot_channels = [str(ch) for ch in spot_channels]
        
        # Create channel-gene mapping for spot channels only
        spot_channel_gene_map = {}
        for channel in spot_channels:
            if channel in gene_dict:
                gene_name = gene_dict[channel]['gene']
                spot_channel_gene_map[channel] = gene_name
        
        return spot_channel_gene_map

    def load_metadata(self, metadata_type: str) -> dict:
        """
        Load a specific metadata JSON file.
        
        Parameters:
        -----------
        metadata_type : str
            Type of metadata to load ('acquisition', 'subject', 'processing', etc.)
            
        Returns:
        --------
        dict
            Loaded metadata dictionary
        """
        import json
        
        if self.metadata_files is None:
            raise ValueError(f"No metadata files available for round {self.round_key}")
            
        if not self.metadata_files.has_metadata(metadata_type):
            available = self.metadata_files.get_available_metadata()
            raise ValueError(
                f"Metadata type '{metadata_type}' not found for round {self.round_key}. "
                f"Available: {available}"
            )
            
        metadata_path = getattr(self.metadata_files, metadata_type)
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def get_subject_metadata(self, fields: list = None) -> dict:
        """
        Extract specific fields from subject metadata.
        
        Parameters:
        -----------
        fields : list, optional
            List of fields to extract. If None, extracts common fields.
            
        Returns:
        --------
        dict
            Dictionary containing requested subject metadata fields with parsed genotype
        """
        if fields is None:
            fields = ['genotype', 'date_of_birth', 'sex', 'subject_id']
            
        try:
            subject_data = self.load_metadata('subject')
            result = {}
            for field in fields:
                if field in subject_data:
                    result[field] = subject_data[field]
                else:
                    result[field] = None
            
            # Parse genotype if present
            if 'genotype' in result and result['genotype']:
                parsed_genotype = parse_genotype(result['genotype'])
                result.update(parsed_genotype)
                
            return result
        except ValueError:
            # Return None values if subject metadata not available
            return {field: None for field in fields}
    
    def get_metadata_summary(self) -> dict:
        """
        Get a summary of available metadata and key information.
        
        Returns:
        --------
        dict
            Summary of metadata information
        """
        summary = {
            'available_metadata': [],
            'subject_info': {},
            'round_key': self.round_key
        }
        
        if self.metadata_files is not None:
            summary['available_metadata'] = self.metadata_files.get_available_metadata()
            
            # Try to get subject info
            try:
                summary['subject_info'] = self.get_subject_metadata()
            except Exception:
                summary['subject_info'] = {}
                
        return summary

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
            "metadata_files",
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
            "get_spot_channel_gene_map",
            "load_metadata",
            "get_subject_metadata",
            "get_metadata_summary",
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
        self.metadata = metadata or {}
        self.dataset_names = dataset_names

        # Initialize rounds
        self.rounds = rounds or {}

        self._validate_rounds()
        self._extract_subject_metadata()

    def _validate_rounds(self):
        """Validate that rounds have consistent data."""
        if not self.rounds:
            return

        # Check for missing processing manifests
        for round_key, round_obj in self.rounds.items():
            if round_obj.processing_manifest is None:
                print(f"Warning: Processing manifest for round {round_key} is None")

    def _extract_subject_metadata(self):
        """Extract and store subject metadata in the dataset metadata."""
        try:
            subject_metadata = self.get_subject_metadata_summary()
            if any(v is not None for v in subject_metadata.values()):
                self.metadata.update(subject_metadata)
        except Exception:
            pass  # Continue if metadata extraction fails

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

    def get_cell_info(self, source="mixed_cxg"):
        """
        Get cell information from a specific round.

        Usually just get R1 cell ids.

        Parameters:
        -----------
        round_key : str
            Round to extract cell info from (default: 'R1')

        Returns:
        --------
        pd.DataFrame
            DataFrame containing cell_id, volume, and centroid coordinates
        """

        if source == "segmentation":
            # only R1 has segmentation centroids
            return self.rounds["R1"].get_cell_info(source=source)

        if source == "mixed_cxg" or source == "unmixed_cxg":
            # Concatenate cell info from all rounds
            all_cells = []
            for r_key, round_obj in self.rounds.items():
                cells = round_obj.get_cell_info(source=source)
                # cells['round'] = r_key
                all_cells.append(cells)

                # get the unique cell ids
            all_cells_df = pd.concat(all_cells, ignore_index=True)
            all_cells_df = all_cells_df.drop_duplicates(subset=["cell_id"]).reset_index(drop=True)

            return all_cells_df

    def load_all_rounds_spots_mp(self, table_type="mixed_spots"):
        """
        Load all spots from the dataset in parallel.

        Parameters:
        -----------
        table_type : str
            Type of spots to load ('mixed_spots' or 'unmixed_spots')

        Returns:
        --------
        pd.DataFrame
            DataFrame containing all spots from all rounds
        """
        # Create list of (round_key, round_obj) tuples for multiprocessing
        round_items = list(self.rounds.items())

        # Use multiprocessing to load spots from all rounds
        pool = mp.Pool(processes=min(len(self.rounds), mp.cpu_count()))
        process_round = partial(_load_spots_for_round, table_type=table_type)
        all_spots_list = pool.map(process_round, round_items)
        pool.close()
        pool.join()

        # Concatenate all DataFrames
        all_spots = pd.concat(all_spots_list, ignore_index=True)
        print(f"Number of {table_type}: {len(all_spots):.3e}")

        return all_spots

    def load_all_rounds_spots_coreg(self, ophys_mfish_match_df, table_type="mixed_spots"):
        """Wrapper function around load_all_rounds_spots_mp to filter for coregistered spots.
        Parameters:
        - dataset: The HCR dataset containing the rounds and mixed spots.
        - ophys_mfish_match_df: DataFrame containing coregistered spots with 'ls_id' column.
        - table_type: Type of spots to load, default is 'mixed_spots'.
        Returns:
        - DataFrame containing coregistered mixed spots.
        """
        # Load all mixed spots
        all_mixed_spots = self.load_all_rounds_spots_mp(table_type=table_type)
        # coreg_spots are in ophys_mfish_match_df 'ls_id' col, need to match 'cell_id' col in all_mixed_spots
        coreg_spots = all_mixed_spots[
            all_mixed_spots["cell_id"].isin(ophys_mfish_match_df["ls_id"])
        ]
        print(f"Number of coregistered mixed spots: {len(coreg_spots)}")
        return coreg_spots

    def create_cell_gene_matrix(self, unmixed=True, rounds=None, sort_order='round_chan'):
        """
        Create cell-gene matrix from specified rounds.

        Parameters:
        -----------
        unmixed : bool
            Whether to use unmixed or mixed data
        rounds : list, optional
            Specific rounds to include. If None, uses all rounds.
        sort_order : str
            'auto' to sort genes alphabetically, 'round_chan' to keep round order, or None for no sorting

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
            pm = self.rounds[round_key].processing_manifest
            try:
                if unmixed:
                    # Read the unmixed cell-by-gene data
                    df = pd.read_csv(spot_files[round_key].unmixed_cxg)
                else:
                    df = pd.read_csv(spot_files[round_key].mixed_cxg)
            except FileNotFoundError:
                print(f"Warning: Spot file for round {round_key} does not exist.")
                continue

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

    def create_channel_gene_table(self, spots_only=True, label_duplicate_genes=False):
        """Create channel-gene mapping table from processing manifests."""
        processing_manifests = {
            k: round_obj.processing_manifest for k, round_obj in self.rounds.items()
        }
        return create_channel_gene_table_from_manifests(
            processing_manifests,
            spots_only=spots_only,
            label_duplicate_genes=label_duplicate_genes,
        )

    def get_subject_metadata_summary(self) -> dict:
        """
        Get subject metadata summary across all rounds, with preference for R1.
        
        Returns:
        --------
        dict
            Subject metadata dictionary with genotype, date_of_birth, sex, subject_id
        """
        # Try to get subject metadata from R1 first, then any available round
        round_keys = ['R1'] + [k for k in self.rounds.keys() if k != 'R1']
        
        for round_key in round_keys:
            if round_key in self.rounds:
                try:
                    subject_metadata = self.rounds[round_key].get_subject_metadata()
                    if any(v is not None for v in subject_metadata.values()):
                        return subject_metadata
                except Exception:
                    continue
                    
        # Return empty if no metadata found
        return {'genotype': None, 'date_of_birth': None, 'sex': None, 'subject_id': None}

    def get_metadata_summary_all_rounds(self) -> dict:
        """
        Get metadata summary for all rounds in the dataset.
        
        Returns:
        --------
        dict
            Dictionary with round keys and their metadata summaries
        """
        summary = {}
        for round_key, round_obj in self.rounds.items():
            summary[round_key] = round_obj.get_metadata_summary()
        return summary

    def load_metadata_from_round(self, round_key: str, metadata_type: str) -> dict:
        """
        Load specific metadata from a specific round.
        
        Parameters:
        -----------
        round_key : str
            Round identifier
        metadata_type : str
            Type of metadata ('subject', 'acquisition', etc.)
            
        Returns:
        --------
        dict
            Loaded metadata dictionary
        """
        if round_key not in self.rounds:
            raise ValueError(f"Round {round_key} not found")
            
        return self.rounds[round_key].load_metadata(metadata_type)

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
            "get_subject_metadata_summary",
            "get_metadata_summary_all_rounds",
            "load_metadata_from_round",
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


def _load_spots_for_round(round_item, table_type="mixed_spots"):
    """
    Helper function for multiprocessing to load spots for a single round.

    Parameters:
    -----------
    round_item : tuple
        Tuple of (round_key, round_obj)
    table_type : str
        Type of spots to load ('mixed_spots' or 'unmixed_spots')

    Returns:
    --------
    pd.DataFrame
        DataFrame with spots data including round column
    """
    round_key, round_obj = round_item
    print(f"Loading {table_type} for round {round_key}: {round_obj.name}\n")

    # Get the appropriate spot file path
    spot_file_path = getattr(round_obj.spot_files, table_type)

    # check if path exists
    if spot_file_path is None or not spot_file_path.exists():
        print(f"Warning: Spot file {spot_file_path} does not exist for round {round_key}")
        return pd.DataFrame()  # Return empty DataFrame if file does not exist

    with open(spot_file_path, "rb") as f:
        spots_data = pkl.load(f)
        spots_data["round"] = round_key

    return spots_data


def create_hcr_dataset(round_dict: dict, 
                       data_dir: Path, 
                       mouse_id: str = None, 
                       config_path: Path = None) -> HCRDataset:
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
    metadata_files = get_metadata_files(round_dict, data_dir)

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
            metadata_files=metadata_files.get(round_key),
        )

    # Load metadata if available
    metadata = None
    if mouse_id:
        try:
            print('mouse_id:', mouse_id)
            config = load_mouse_config(mouse_id=str(mouse_id),config_path=config_path)
            metadata = config.get("metadata", {})
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

    return create_hcr_dataset(round_dict, data_dir, mouse_id,config_path=config_path)


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
                    {"channel": channel, "gene": details.get("gene", ""), "round": round_key}
                )

    # sort by round then channel
    data.sort(key=lambda x: (x["round"], x["channel"]))

    if spots_only:
        # drop Channel = 405 and Gene = Syto59
        data = [
            entry
            for entry in data
            if not (entry["channel"] == "405" and entry["gene"] == "Syto59")
        ]
    # for duplicate genes, append the round name to the gene
    for entry in data:
        if entry["gene"] in [d["gene"] for d in data if d["round"] != entry["round"]]:
            entry["gene"] += f"-{entry['round']}"
    return pd.DataFrame(data)


def create_channel_gene_table_from_manifests(
    processing_manifests: Dict[str, dict], spots_only=True, label_duplicate_genes=True
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
    if label_duplicate_genes:
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

        if mask_transformed_level2_path.exists() and mask_orig_res_path.exists():
            print(
                "WARNING: multiple types of segmentation masks found. Double check the data is"
                "spots/segmentation data is correct for you. Defaulting to transformed masks."
            )
            segmentation_masks["2"] = mask_transformed_level2_path
        elif mask_transformed_level2_path.exists():
            segmentation_masks["2"] = mask_transformed_level2_path
        elif mask_orig_res_path.exists():
            segmentation_masks["2"] = mask_orig_res_path

        # Check for cell centroids
        centroids_path = folder_path / "cell_centroids.npy"
        if not centroids_path.exists():
            centroids_path = None

        # check for metrics.pkl
        metrics_path = folder_path / "metrics.pickle"
        if not metrics_path.exists():
            metrics_path = None

        segmentation_files[key] = SegmentationFiles(
            segmentation_masks=segmentation_masks,
            cell_centroids=centroids_path,
            metrics_path=metrics_path,
        )

    return segmentation_files


def get_metadata_files(round_dict: dict, data_dir: Path):
    """
    Get MetadataFiles for each round based on a dictionary mapping round keys to folder names.

    Parameters:
    -----------
    round_dict : dict
        Dictionary mapping round keys (e.g., 'R1', 'R2') to folder names containing the data.
    data_dir : Path
        Path to the directory containing the round folders.

    Returns:
    --------
    dict
        Dictionary mapping round keys to MetadataFiles objects containing paths to metadata JSON files.
    """
    metadata_files = {}

    for key, folder in round_dict.items():
        metadata_path = data_dir / folder / "metadata"
        
        def check_exist(path):
            """Helper function to check if file exists and return path or None."""
            return path if path.exists() else None

        # Check for all standard metadata files
        metadata_files[key] = MetadataFiles(
            acquisition=check_exist(metadata_path / "acquisition.json"),
            data_description=check_exist(metadata_path / "data_description.json"),
            metadata_nd=check_exist(metadata_path / "metadata.nd.json"),
            procedures=check_exist(metadata_path / "procedures.json"),
            processing=check_exist(metadata_path / "processing.json"),
            subject=check_exist(metadata_path / "subject.json"),
        )

    return metadata_files


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
        ratios_file = next(folder_path.absolute().glob("*_ratios.txt"), None)
        spot_files[key] = SpotFiles(
            unmixed_cxg=unmixed_cxg,
            mixed_cxg=mixed_cxg,
            unmixed_spots=unmixed_spots,
            mixed_spots=mixed_spots,
            spot_unmixing_stats=stats,
            ratios_file=ratios_file,
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
