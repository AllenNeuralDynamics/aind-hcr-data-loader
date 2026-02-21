"""
For now keep separate since some of the filters are fairly custom and rely on metrics generated
off-pipeline. Could be added to the data_loader when things are more standard in the future
10/10/2025



"""
#import soma_classifier_simple as soma

import aind_hcr_data_loader.classifiers.simple_soma_1.functions as soma

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import aind_hcr_qc.viz.tile_alignment as ta

import pickle as pkl
import json
import pickle

from aind_hcr_data_loader.hcr_dataset import HCRDataset
from aind_hcr_qc.utils.utils import saveable_plot

def roi_filter_comprehensive(
    ds: HCRDataset,
    round_key: str = "R1",
    metrics_base_path: str = "/root/capsule/scratch/shape_metrics",
    soma_threshold: float = 0.8,
    edge_xy_threshold: float = 10.0,
    edge_z_threshold: float = 10.0,
    overlap_threshold: float = 0.1,
    verbose: bool = True
):
    """
    Apply comprehensive ROI filtering pipeline.
    
    Filters ROIs based on multiple quality criteria:
    1. Cell size (volume) - removes outliers
    2. Soma classifier - identifies non-soma cells
    3. Edge ROIs - identifies cells at tissue boundaries
    4. Tile overlap boundaries - identifies cells in stitching artifacts
    
    Parameters
    ----------
    ds : HCRDataset
        The HCR dataset object
    round_key : str, default='R1'
        Round to use for metrics (typically R1 has most complete set)
    metrics_base_path : str, default='/root/capsule/scratch/shape_metrics'
        Base path to shape metrics directory
    soma_threshold : float, default=0.8
        Confidence threshold for soma classifier
    edge_xy_threshold : float, default=10.0
        Distance threshold for XY edge detection (pixels)
    edge_z_threshold : float, default=10.0
        Distance threshold for Z edge detection (pixels)
    overlap_threshold : float, default=0.1
        Overlap fraction threshold for tile boundary filtering
    verbose : bool, default=True
        Print detailed progress information
        
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'filtered_ids': np.ndarray - All ROI IDs to filter out
        - 'soma_filter_ids': np.ndarray - IDs filtered by soma classifier
        - 'edge_filter_ids': np.ndarray - IDs filtered by edge classifier
        - 'overlap_filter_ids': np.ndarray - IDs filtered by tile overlap
        - 'volume_filtered_ids': np.ndarray - IDs passing volume filter
        - 'soma_classifier_df': pd.DataFrame - Full soma classification results
        - 'edge_classifier_df': pd.DataFrame - Full edge classification results
        - 'metrics_df': pd.DataFrame - Upscaled metrics dataframe
        - 'n_total': int - Total ROIs before filtering
        - 'n_filtered': int - Total ROIs filtered out
        - 'n_kept': int - Total ROIs kept after filtering
    """
    
    if verbose:
        print("="*80)
        print("ROI FILTERING PIPELINE")
        print("="*80)
    
    # -------------------------------------------------------------------------
    # 1. Load metrics data
    # -------------------------------------------------------------------------
    dataset_name = ds.rounds[round_key].name
    metrics_path = Path(metrics_base_path) / dataset_name / "seg_shape_metrics_pyr2.parquet"
    
    if verbose:
        print(f"\n[1/5] Loading metrics from: {metrics_path}")
    
    metrics_df = pd.read_parquet(metrics_path)
    n_total = metrics_df.shape[0]
    
    if verbose:
        print(f"      Total ROIs loaded: {n_total:,}")
    
    # -------------------------------------------------------------------------
    # 2. Volume filter
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n[2/5] Applying volume filter...")
    
    metrics_df, feat_cols = soma.build_features(metrics_df)
    metrics_df = soma.volume_filter(metrics_df)
    volume_filtered_ids = metrics_df.cell_id.values
    
    if verbose:
        n_volume_kept = len(volume_filtered_ids)
        n_volume_filtered = n_total - n_volume_kept
        print(f"      Kept: {n_volume_kept:,} ROIs")
        print(f"      Filtered out: {n_volume_filtered:,} ROIs (volume outliers)")
    
    # -------------------------------------------------------------------------
    # 3. Soma classifier
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n[3/5] Running soma classifier (threshold={soma_threshold})...")
    
    bundle = soma.load_soma_classifier()
    soma_classifier_df = soma.predict_soma_labels(
        df=metrics_df,
        pipeline=bundle['pipeline'],
        feat_cols=bundle['feat_cols'],
        threshold=soma_threshold
    )
    
    # Get IDs for soma vs non-soma
    soma_ids = soma_classifier_df.loc[soma_classifier_df.predicted_soma == True, 'cell_id'].values
    non_soma_ids = soma_classifier_df.loc[soma_classifier_df.predicted_soma == False, 'cell_id'].values
    
    if verbose:
        print(f"      Soma ROIs: {len(soma_ids):,}")
        print(f"      Non-soma ROIs: {len(non_soma_ids):,} (to filter)")
    
    # -------------------------------------------------------------------------
    # 4. Edge ROI classifier
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n[4/5] Running edge ROI classifier...")
        print(f"      XY threshold: {edge_xy_threshold} pixels")
        print(f"      Z threshold: {edge_z_threshold} pixels")
    
    edge_classifier = EdgeROIClassifier(
        xy_distance_threshold=edge_xy_threshold,
        z_distance_threshold=edge_z_threshold
    )
    
    edge_classified_df = edge_classifier.classify(metrics_df, verbose=False)
    edge_roi_ids = edge_classified_df.loc[edge_classified_df.is_edge_roi == True, 'cell_id'].values
    core_roi_ids = edge_classified_df.loc[edge_classified_df.is_edge_roi == False, 'cell_id'].values
    
    if verbose:
        print(f"      Core ROIs: {len(core_roi_ids):,}")
        print(f"      Edge ROIs: {len(edge_roi_ids):,} (to filter)")
    
    # -------------------------------------------------------------------------
    # 5. Tile boundary overlap filter
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n[5/5] Running tile boundary overlap filter...")
    
    # This method upscales coordinates to pyramid level 0
    metrics_upscaled_df, overlap_roi_ids = filter_tile_boundary_rois(
        ds, round_key, overlap_threshold=overlap_threshold
    )
    
    if verbose:
        print(f"      ROIs in tile overlaps: {len(overlap_roi_ids):,} (to filter)")
    
    # -------------------------------------------------------------------------
    # 6. Combine all filters
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n{'='*80}")
        print("COMBINING FILTERS")
        print("="*80)
    
    # Combine all IDs to filter out (union of all filters)
    all_filtered_ids = np.unique(np.concatenate([
        non_soma_ids,
        edge_roi_ids,
        overlap_roi_ids
    ]))
    
    n_filtered = len(all_filtered_ids)
    n_kept = n_total - n_filtered
    
    if verbose:
        print(f"\nFilter breakdown:")
        print(f"  Non-soma:        {len(non_soma_ids):>8,}")
        print(f"  Edge ROIs:       {len(edge_roi_ids):>8,}")
        print(f"  Tile overlaps:   {len(overlap_roi_ids):>8,}")
        print(f"  " + "-"*30)
        print(f"  Total unique:    {n_filtered:>8,} (filtered out)")
        print(f"  Kept:            {n_kept:>8,} ({100*n_kept/n_total:.1f}%)")
        print(f"\n{'='*80}\n")
    
    # -------------------------------------------------------------------------
    # 7. Return comprehensive results
    # -------------------------------------------------------------------------
    results = {
        'filtered_ids': all_filtered_ids,
        'soma_filter_ids': non_soma_ids,
        'edge_filter_ids': edge_roi_ids,
        'overlap_filter_ids': overlap_roi_ids,
        'volume_filtered_ids': volume_filtered_ids,
        'soma_classifier_df': soma_classifier_df,
        'edge_classifier_df': edge_classified_df,
        'metrics_df': metrics_upscaled_df,
        'n_total': n_total,
        'n_filtered': n_filtered,
        'n_kept': n_kept,
    }
    
    return results

def filter_tile_boundary_rois(
    ds: HCRDataset,
    round_key: str,
    overlap_threshold: float = 0.1
):
    """
    Filter ROIs that fall within tile overlap regions.
    
    Note: Metrics must be from GPU loader which has correct bboxes.
    This function upscales coordinates from pyramid level 2 to level 0.
    
    Parameters
    ----------
    ds : HCRDataset
        The HCR dataset object
    round_key : str
        The round key (e.g., 'R1', 'R2')
    overlap_threshold : float, default=0.1
        Fraction of ROI overlap required to be filtered (0-1)
    
    Returns
    -------
    filtered_df : pd.DataFrame
        Upscaled metrics dataframe
    filtered_ids : np.ndarray
        IDs of ROIs in overlap regions
    """
    pc_xml = ds.rounds[round_key].tile_alignment_files.pc_xml
    stitched_xml = ta.parse_bigstitcher_xml(pc_xml)
    pairs = ta.get_all_adjacent_pairs(stitched_xml["tile_names"], include_diagonals=False)
    print(f"Found {len(pairs)} adjacent tile pairs")

    overlap_regions = ta.calculate_overlap_regions(stitched_xml, pairs)
    overlap_bbox_array = ta.get_overlap_bbox_array_from_dict(stitched_xml, pairs)

    # load metrics and upscale
    metrics_path = Path(f"/root/capsule/scratch/shape_metrics/{ds.rounds[round_key].name}/seg_shape_metrics_pyr2.parquet")
    print(f"Loading metrics from {metrics_path}")
    df = pd.read_parquet(metrics_path)
    centroid_cols = ['centroid_y', 'centroid_x']
    bbox_cols = ['bbox_min_y', 'bbox_min_x', 'bbox_max_y', 'bbox_max_x']
    df_up = df.copy()

    # Upscale coordinates from pyramid level 2 to level 0
    # Note: Different scaling factors apply
    # - Centroids: 16x (full resolution)
    # - Bboxes: 4x (quarter resolution due to how they're stored)
    df_up[centroid_cols] = df_up[centroid_cols] * 16
    df_up[bbox_cols] = df_up[bbox_cols] * 4 

    filtered_df, filtered_ids = ta.filter_rois_in_overlap_regions(
        df_up, 
        overlap_regions, 
        overlap_threshold=overlap_threshold,
        id_col='cell_id',
        bbox_cols=['bbox_min_x', 'bbox_max_x', 'bbox_min_y', 'bbox_max_y']
    )

    return filtered_df, filtered_ids


# Probably can go in tile_alignment
def load_tile_overlaps(ds, round_key):
    """
    Load tile alignment data and calculate overlap regions.
    
    Parameters
    ----------
    ds : HCRDataset
        The HCR dataset object
    round_key : str
        The round key (e.g., 'R1', 'R2')
    
    Returns
    -------
    tuple
        (stitched_xml, pairs, overlap_regions, overlap_bbox_array)
    """
    
    pc_xml = ds.rounds[round_key].tile_alignment_files.pc_xml
    stitched_xml = ta.parse_bigstitcher_xml(pc_xml)
    pairs = ta.get_all_adjacent_pairs(stitched_xml["tile_names"], include_diagonals=False)
    print(f"Found {len(pairs)} adjacent tile pairs")
    
    overlap_regions = ta.calculate_overlap_regions(stitched_xml, pairs)
    overlap_bbox_array = ta.get_overlap_bbox_array_from_dict(stitched_xml, pairs)
    
    return stitched_xml, pairs, overlap_regions, overlap_bbox_array


def filter_cell_info(cell_info, q1=0.2, q2=0.95):
    """
    Filter cells based on volume quantiles.
    
    Parameters
    ----------
    cell_info : pd.DataFrame
        DataFrame with cell information including 'volume' column
    q1 : float
        Lower quantile threshold (default: 0.2)
    q2 : float
        Upper quantile threshold (default: 0.95)
    
    Returns
    -------
    pd.DataFrame
        Filtered cell info dataframe
    """
    n_cells = cell_info.shape[0]
    cell_info = cell_info[
        (cell_info["volume"] > cell_info["volume"].quantile(q1))
        & (cell_info["volume"] < cell_info["volume"].quantile(q2))
    ]
    print(f"Kept {cell_info.shape[0]} cells out of {n_cells} total cells, based on volume quantiles {q1} and {q2}.")
    return cell_info


# ---
# Edge filter
# ---

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Literal
class EdgeROIClassifier:
    """
    Classify ROIs as edge or core based on distance to tissue boundaries.
    
    Uses percentile-based boundary detection that is robust to outliers:
    - XY plane: Simple bounding box based on percentiles
    - Z planes (ZX, ZY): Percentile refinement that creates smooth curves
      following tissue shape, ignoring top/bottom boundaries (handled by XY)
    
    Attributes
    ----------
    xy_distance_threshold : float
        Distance threshold for XY boundaries (pixels)
    z_distance_threshold : float
        Distance threshold for Z boundaries (pixels)
    xy_percentile : float
        Percentile for XY bounding box
    z_n_slices : int
        Number of slices for Z percentile computation
    z_percentile_low : float
        Lower percentile for Z boundaries
    z_percentile_high : float
        Upper percentile for Z boundaries
    z_smooth_window : int
        Smoothing window size for Z boundaries
    z_smooth_method : str
        Smoothing method ('savgol' or 'median')
    savgol_order : int
        Polynomial order for Savitzky-Golay filter
    
    Examples
    --------
    >>> classifier = EdgeROIClassifier(
    ...     xy_distance_threshold=10,
    ...     z_distance_threshold=100
    ... )
    >>> classified_df = classifier.classify(rois_df)
    >>> fig = classifier.plot(classified_df, orientation='ZX')
    """
    
    def __init__(
        self,
        xy_distance_threshold: float = 10.0,
        z_distance_threshold: float = 10.0,
        xy_percentile: float = 1.0,
        z_n_slices: int = 250,
        z_percentile_low: float = 1.0,
        z_percentile_high: float = 99.0,
        z_smooth_window: int = 8,
        z_smooth_method: Literal['savgol', 'median'] = 'savgol',
        savgol_order: int = 3
    ):
        """
        Initialize EdgeROIClassifier.
        
        Parameters
        ----------
        xy_distance_threshold : float, default=10
            Distance from XY boundary to classify as edge (pixels)
        z_distance_threshold : float, default=10
            Distance from Z boundary to classify as edge (pixels)
        xy_percentile : float, default=1
            Percentile for XY bounding box (0-100)
        z_n_slices : int, default=250
            Number of slices for Z percentile computation
        z_percentile_low : float, default=1
            Lower percentile for Z boundaries (0-100)
        z_percentile_high : float, default=99
            Upper percentile for Z boundaries (0-100)
        z_smooth_window : int, default=8
            Smoothing window size (must be odd for savgol)
        z_smooth_method : {'savgol', 'median'}, default='savgol'
            Smoothing method for Z boundaries
        savgol_order : int, default=3
            Polynomial order for Savitzky-Golay filter
        """
        self.xy_distance_threshold = xy_distance_threshold
        self.z_distance_threshold = z_distance_threshold
        self.xy_percentile = xy_percentile
        self.z_n_slices = z_n_slices
        self.z_percentile_low = z_percentile_low
        self.z_percentile_high = z_percentile_high
        self.z_smooth_window = z_smooth_window
        self.z_smooth_method = z_smooth_method
        self.savgol_order = savgol_order
        
    def _estimate_xy_boundary(self, df: pd.DataFrame) -> Tuple[np.ndarray, dict]:
        """
        Estimate XY boundary using simple percentile-based bounding box.
        
        Parameters
        ----------
        df : pd.DataFrame
            ROI dataframe with centroid_x, centroid_y columns
            
        Returns
        -------
        distances : np.ndarray
            Signed distances to XY boundary
        boundary_info : dict
            Boundary metadata
        """
        x_min, x_max = np.percentile(
            df['centroid_x'], 
            [self.xy_percentile, 100 - self.xy_percentile]
        )
        y_min, y_max = np.percentile(
            df['centroid_y'], 
            [self.xy_percentile, 100 - self.xy_percentile]
        )
        
        x = df['centroid_x'].values
        y = df['centroid_y'].values
        
        # Distance to edges
        dist_to_x_edge = np.minimum(x - x_min, x_max - x)
        dist_to_y_edge = np.minimum(y - y_min, y_max - y)
        dist_xy = np.minimum(dist_to_x_edge, dist_to_y_edge)
        
        # Mark outside points as negative distance
        outside_x = (x < x_min) | (x > x_max)
        outside_y = (y < y_min) | (y > y_max)
        dist_xy[outside_x | outside_y] = -np.abs(dist_xy[outside_x | outside_y])
        
        boundary_info = {
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max,
            'method': 'bounding_box'
        }
        
        return dist_xy, boundary_info
    
    def _estimate_z_boundary(
        self, 
        df: pd.DataFrame, 
        z_col: str = 'centroid_z',
        slice_col: str = 'centroid_x'
    ) -> Tuple[np.ndarray, dict]:
        """
        Estimate Z boundary using percentile refinement (left/right only).
        
        This method computes Z percentiles for each X (or Y) position,
        creating vertical boundary curves that ignore top/bottom edges
        (already handled by XY filtering).
        
        Parameters
        ----------
        df : pd.DataFrame
            ROI dataframe
        z_col : str
            Z coordinate column name
        slice_col : str
            Slicing axis column (X or Y)
            
        Returns
        -------
        distances : np.ndarray
            Signed distances to Z boundary
        boundary_info : dict
            Boundary metadata including slice_centers, z_low, z_high
        """
        z = df[z_col].values
        slice_vals = df[slice_col].values
        
        # Create slices along X or Y
        slice_min, slice_max = slice_vals.min(), slice_vals.max()
        slice_edges = np.linspace(slice_min, slice_max, self.z_n_slices + 1)
        slice_centers = (slice_edges[:-1] + slice_edges[1:]) / 2
        
        # Compute Z percentiles for each X/Y slice
        z_low = np.zeros(self.z_n_slices)
        z_high = np.zeros(self.z_n_slices)
        
        for i in range(self.z_n_slices):
            mask = (slice_vals >= slice_edges[i]) & (slice_vals < slice_edges[i + 1])
            if mask.sum() > 0:
                z_slice = z[mask]
                z_low[i] = np.percentile(z_slice, self.z_percentile_low)
                z_high[i] = np.percentile(z_slice, self.z_percentile_high)
            else:
                z_low[i] = np.nan
                z_high[i] = np.nan
        
        # Fill NaN values
        valid_mask = ~np.isnan(z_low)
        if valid_mask.sum() > 0:
            z_low = np.interp(slice_centers, slice_centers[valid_mask], z_low[valid_mask])
            z_high = np.interp(slice_centers, slice_centers[valid_mask], z_high[valid_mask])
        
        # Smooth the curves
        if self.z_smooth_method == 'savgol':
            window = min(self.z_smooth_window, len(slice_centers))
            if window % 2 == 0:
                window -= 1
            window = max(window, self.savgol_order + 2)
            
            z_low_smooth = savgol_filter(z_low, window, self.savgol_order)
            z_high_smooth = savgol_filter(z_high, window, self.savgol_order)
        elif self.z_smooth_method == 'median':
            z_low_smooth = median_filter(z_low, size=self.z_smooth_window)
            z_high_smooth = median_filter(z_high, size=self.z_smooth_window)
        else:
            z_low_smooth = z_low
            z_high_smooth = z_high
        
        # Compute distances
        z_low_interp = np.interp(slice_vals, slice_centers, z_low_smooth)
        z_high_interp = np.interp(slice_vals, slice_centers, z_high_smooth)
        
        dist_to_low = z - z_low_interp
        dist_to_high = z_high_interp - z
        dist = np.minimum(dist_to_low, dist_to_high)
        
        # Mark outside points as negative
        outside = (z < z_low_interp) | (z > z_high_interp)
        dist[outside] = -np.abs(dist[outside])
        
        boundary_info = {
            'slice_centers': slice_centers,
            'z_low': z_low_smooth,
            'z_high': z_high_smooth,
            'slice_col': slice_col,
            'z_col': z_col,
            'method': 'percentile_z_only'
        }
        
        return dist, boundary_info
    
    def classify(
        self, 
        df: pd.DataFrame,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Classify ROIs as edge or core based on boundary distances.
        
        Parameters
        ----------
        df : pd.DataFrame
            ROI dataframe with centroid_x, centroid_y, centroid_z columns
        verbose : bool, default=True
            Print classification summary
            
        Returns
        -------
        df_classified : pd.DataFrame
            Copy of input with added columns:
            - distance_to_edge_xy : float
            - distance_to_edge_zx : float
            - distance_to_edge_zy : float
            - min_distance_to_edge : float
            - is_edge_roi : bool
            - edge_zone : {'core', 'edge'}
            
            Also has attributes:
            - _boundary_info : dict with keys 'xy', 'zx', 'zy'
            - _thresholds : dict with keys 'xy', 'z'
        """
        df_out = df.copy()
        
        if verbose:
            print("=== Percentile-Based Edge ROI Classification ===")
            print(f"\nXY Boundary (simple bounding box):")
            print(f"  Percentile: {self.xy_percentile}")
            print(f"  Distance threshold: {self.xy_distance_threshold} px")
        
        # XY plane - simple bounding box
        dist_xy, boundary_xy = self._estimate_xy_boundary(df)
        
        if verbose:
            print(f"  X range: [{boundary_xy['x_min']:.1f}, {boundary_xy['x_max']:.1f}]")
            print(f"  Y range: [{boundary_xy['y_min']:.1f}, {boundary_xy['y_max']:.1f}]")
            
            print(f"\nZ Boundaries (percentile refinement, left/right only):")
            print(f"  Slices: {self.z_n_slices}")
            print(f"  Percentiles: [{self.z_percentile_low}, {self.z_percentile_high}]")
            print(f"  Smoothing: {self.z_smooth_method}, window={self.z_smooth_window}")
            print(f"  Distance threshold: {self.z_distance_threshold} px")
            print(f"  Note: Top/bottom boundaries handled by XY filtering")
        
        # ZX plane - Z boundaries as function of X
        dist_zx, boundary_zx = self._estimate_z_boundary(
            df, z_col='centroid_z', slice_col='centroid_x'
        )
        
        # ZY plane - Z boundaries as function of Y
        dist_zy, boundary_zy = self._estimate_z_boundary(
            df, z_col='centroid_z', slice_col='centroid_y'
        )
        
        # Store distances
        df_out['distance_to_edge_xy'] = dist_xy
        df_out['distance_to_edge_zx'] = dist_zx
        df_out['distance_to_edge_zy'] = dist_zy
        df_out['min_distance_to_edge'] = np.minimum(
            np.minimum(dist_xy, dist_zx), dist_zy
        )
        
        # Apply plane-specific thresholds
        is_edge_xy = dist_xy < self.xy_distance_threshold
        is_edge_zx = dist_zx < self.z_distance_threshold
        is_edge_zy = dist_zy < self.z_distance_threshold
        
        # ROI is edge if it's edge in ANY plane
        df_out['is_edge_roi'] = is_edge_xy | is_edge_zx | is_edge_zy
        df_out['edge_zone'] = 'core'
        df_out.loc[df_out['is_edge_roi'], 'edge_zone'] = 'edge'
        
        # Store boundary info as attributes
        df_out._boundary_info = {
            'xy': boundary_xy,
            'zx': boundary_zx,
            'zy': boundary_zy
        }
        df_out._thresholds = {
            'xy': self.xy_distance_threshold,
            'z': self.z_distance_threshold
        }
        
        # Summary
        if verbose:
            n_total = len(df_out)
            n_core = (df_out['edge_zone'] == 'core').sum()
            n_edge = (df_out['edge_zone'] == 'edge').sum()
            
            print(f"\n=== Classification Summary ===")
            print(f"Total ROIs: {n_total:,}")
            print(f"Core ROIs: {n_core:,} ({100*n_core/n_total:.1f}%)")
            print(f"Edge ROIs: {n_edge:,} ({100*n_edge/n_total:.1f}%)")
            
            n_edge_xy = is_edge_xy.sum()
            n_edge_zx = is_edge_zx.sum()
            n_edge_zy = is_edge_zy.sum()
            print(f"\nEdge ROIs by plane:")
            print(f"  XY: {n_edge_xy:,} ({100*n_edge_xy/n_total:.1f}%)")
            print(f"  ZX: {n_edge_zx:,} ({100*n_edge_zx/n_total:.1f}%)")
            print(f"  ZY: {n_edge_zy:,} ({100*n_edge_zy/n_total:.1f}%)")
        
        return df_out
    
    @saveable_plot()
    def plot(
        self,
        df: pd.DataFrame,
        orientation: Literal['XY', 'ZX', 'ZY'] = 'ZX',
        distance_threshold: Optional[float] = None,
        figsize: Tuple[int, int] = (18, 5)
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Visualize boundary detection and edge classification.
        
        Parameters
        ----------
        df : pd.DataFrame
            Classified dataframe from classify()
        orientation : {'XY', 'ZX', 'ZY'}, default='ZX'
            Viewing plane
        distance_threshold : float, optional
            Distance threshold to show. If None, uses stored threshold
        figsize : tuple, default=(18, 5)
            Figure size
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object
        axes : np.ndarray
            Array of 3 axes objects
        """
        if not hasattr(df, '_boundary_info'):
            raise ValueError(
                "DataFrame must have _boundary_info attribute. "
                "Did you forget to call classify()?"
            )
        
        orientation_key = orientation.lower()
        boundary_info = df._boundary_info[orientation_key]
        
        # Get appropriate threshold
        if distance_threshold is None and hasattr(df, '_thresholds'):
            if orientation == 'XY':
                distance_threshold = df._thresholds['xy']
            else:
                distance_threshold = df._thresholds['z']
        elif distance_threshold is None:
            distance_threshold = 50
        
        # Get column names
        if orientation == 'XY':
            x_col, y_col = 'centroid_x', 'centroid_y'
            dist_col = 'distance_to_edge_xy'
        elif orientation == 'ZX':
            x_col, y_col = 'centroid_z', 'centroid_x'
            dist_col = 'distance_to_edge_zx'
        elif orientation == 'ZY':
            x_col, y_col = 'centroid_z', 'centroid_y'
            dist_col = 'distance_to_edge_zy'
        else:
            raise ValueError(f"Unknown orientation: {orientation}")
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Plot 1: Scatter with boundary curves
        axes[0].scatter(
            df[x_col], df[y_col], 
            s=0.5, alpha=0.3, c='blue', rasterized=True
        )
        
        if boundary_info.get('method') == 'percentile_z_only':
            slice_centers = boundary_info['slice_centers']
            z_low = boundary_info['z_low']
            z_high = boundary_info['z_high']
            
            axes[0].plot(z_low, slice_centers, 'r-', linewidth=2, 
                        label='Left boundary (min Z)')
            axes[0].plot(z_high, slice_centers, 'r-', linewidth=2,
                        label='Right boundary (max Z)')
            axes[0].fill_betweenx(slice_centers, z_low, z_high, 
                                 color='red', alpha=0.1)
            axes[0].legend()
            axes[0].set_title(f'Z-Only Boundary ({orientation})\nIgnores top/bottom')
            
        elif boundary_info.get('method') == 'bounding_box':
            x_min = boundary_info.get('x_min')
            x_max = boundary_info.get('x_max')
            y_min = boundary_info.get('y_min')
            y_max = boundary_info.get('y_max')
            
            if x_min is not None:
                axes[0].axvline(x_min, color='red', linewidth=2, label='Boundary')
                axes[0].axvline(x_max, color='red', linewidth=2)
                axes[0].axhline(y_min, color='red', linewidth=2)
                axes[0].axhline(y_max, color='red', linewidth=2)
                axes[0].legend()
            axes[0].set_title(f'Bounding Box ({orientation})')
        
        axes[0].set_xlabel(x_col)
        axes[0].set_ylabel(y_col)
        axes[0].set_aspect('equal')
        
        # Plot 2: ROIs colored by edge zone
        zone_colors = {'core': 'blue', 'edge': 'red'}
        for zone, color in zone_colors.items():
            mask = df['edge_zone'] == zone
            n = mask.sum()
            axes[1].scatter(
                df.loc[mask, x_col], 
                df.loc[mask, y_col],
                s=1, alpha=0.4, c=color, 
                label=f'{zone} (n={n:,})', 
                rasterized=True
            )
        
        # Overlay boundary curves
        if boundary_info.get('method') == 'percentile_z_only':
            axes[1].plot(z_low, slice_centers, 'k-', linewidth=1, alpha=0.5)
            axes[1].plot(z_high, slice_centers, 'k-', linewidth=1, alpha=0.5)
        
        axes[1].legend(markerscale=5)
        axes[1].set_title(f'Edge Classification (threshold={distance_threshold} px)')
        axes[1].set_xlabel(x_col)
        axes[1].set_ylabel(y_col)
        axes[1].set_aspect('equal')
        
        # Plot 3: Distance distribution
        axes[2].hist(df[dist_col], bins=100, edgecolor='none', alpha=0.7)
        axes[2].axvline(distance_threshold, color='orange', linestyle='--', 
                       linewidth=2, label=f'Edge threshold ({distance_threshold} px)')
        axes[2].axvline(0, color='red', linestyle='-', linewidth=2, 
                       alpha=0.5, label='Boundary')
        axes[2].set_xlabel(f'Distance to Edge ({orientation})')
        axes[2].set_ylabel('Count')
        axes[2].set_title('Distance Distribution')
        axes[2].legend()
        axes[2].set_yscale('log')
        
        plt.tight_layout()
        return fig, axes
    
    @saveable_plot()
    def plot_all_orientations(
        self,
        df: pd.DataFrame,
        distance_threshold: Optional[float] = None,
        figsize: Tuple[int, int] = (20, 12)
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Visualize edge classification in all three orientations in a single figure.
        
        Creates a 3x3 grid showing boundary detection, classification, and distance
        distribution for ZX, ZY, and XY planes.
        
        Parameters
        ----------
        df : pd.DataFrame
            Classified dataframe from classify()
        distance_threshold : float, optional
            Distance threshold to show. If None, uses stored thresholds
        figsize : tuple, default=(20, 12)
            Figure size
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object
        axes : np.ndarray
            2D array of axes (3 rows x 3 columns)
        """
        if not hasattr(df, '_boundary_info'):
            raise ValueError(
                "DataFrame must have _boundary_info attribute. "
                "Did you forget to call classify()?"
            )
        
        # Create figure with 3 rows (one per orientation) and 3 columns
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        
        orientations = ['ZX', 'ZY', 'XY']
        zone_colors = {'core': 'blue', 'edge': 'red'}
        
        for row_idx, orientation in enumerate(orientations):
            orientation_key = orientation.lower()
            boundary_info = df._boundary_info[orientation_key]
            
            # Get appropriate threshold
            if distance_threshold is None and hasattr(df, '_thresholds'):
                if orientation == 'XY':
                    thresh = df._thresholds['xy']
                else:
                    thresh = df._thresholds['z']
            else:
                thresh = distance_threshold or 50
            
            # Get column names
            if orientation == 'XY':
                x_col, y_col = 'centroid_x', 'centroid_y'
                dist_col = 'distance_to_edge_xy'
            elif orientation == 'ZX':
                x_col, y_col = 'centroid_z', 'centroid_x'
                dist_col = 'distance_to_edge_zx'
            elif orientation == 'ZY':
                x_col, y_col = 'centroid_z', 'centroid_y'
                dist_col = 'distance_to_edge_zy'
            
            # Column 0: Scatter with boundary curves
            ax0 = axes[row_idx, 0]
            ax0.scatter(
                df[x_col], df[y_col], 
                s=0.5, alpha=0.3, c='blue', rasterized=True
            )
            
            if boundary_info.get('method') == 'percentile_z_only':
                slice_centers = boundary_info['slice_centers']
                z_low = boundary_info['z_low']
                z_high = boundary_info['z_high']
                
                ax0.plot(z_low, slice_centers, 'r-', linewidth=2)
                ax0.plot(z_high, slice_centers, 'r-', linewidth=2)
                ax0.fill_betweenx(slice_centers, z_low, z_high, 
                                 color='red', alpha=0.1)
                ax0.set_title(f'{orientation} - Boundary Detection')
                
            elif boundary_info.get('method') == 'bounding_box':
                x_min = boundary_info.get('x_min')
                x_max = boundary_info.get('x_max')
                y_min = boundary_info.get('y_min')
                y_max = boundary_info.get('y_max')
                
                if x_min is not None:
                    ax0.axvline(x_min, color='red', linewidth=2)
                    ax0.axvline(x_max, color='red', linewidth=2)
                    ax0.axhline(y_min, color='red', linewidth=2)
                    ax0.axhline(y_max, color='red', linewidth=2)
                ax0.set_title(f'{orientation} - Bounding Box')
            
            ax0.set_xlabel(x_col)
            ax0.set_ylabel(y_col)
            ax0.set_aspect('equal')
            
            # Column 1: ROIs colored by edge zone
            ax1 = axes[row_idx, 1]
            for zone, color in zone_colors.items():
                mask = df['edge_zone'] == zone
                n = mask.sum()
                ax1.scatter(
                    df.loc[mask, x_col], 
                    df.loc[mask, y_col],
                    s=1, alpha=0.4, c=color, 
                    label=f'{zone} (n={n:,})', 
                    rasterized=True
                )
            
            # Overlay boundary curves
            if boundary_info.get('method') == 'percentile_z_only':
                ax1.plot(z_low, slice_centers, 'k-', linewidth=1, alpha=0.5)
                ax1.plot(z_high, slice_centers, 'k-', linewidth=1, alpha=0.5)
            
            ax1.legend(markerscale=5, loc='best')
            ax1.set_title(f'{orientation} - Classification (t={thresh} px)')
            ax1.set_xlabel(x_col)
            ax1.set_ylabel(y_col)
            ax1.set_aspect('equal')
            
            # Column 2: Distance distribution
            ax2 = axes[row_idx, 2]
            ax2.hist(df[dist_col], bins=100, edgecolor='none', alpha=0.7)
            ax2.axvline(thresh, color='orange', linestyle='--', 
                       linewidth=2, label=f'Edge threshold ({thresh} px)')
            ax2.axvline(0, color='red', linestyle='-', linewidth=2, 
                       alpha=0.5, label='Boundary')
            ax2.set_xlabel(f'Distance to Edge')
            ax2.set_ylabel('Count')
            ax2.set_title(f'{orientation} - Distance Distribution')
            ax2.legend()
            ax2.set_yscale('log')
        
        # Add overall title
        fig.suptitle('Edge ROI Classification - All Orientations', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        return fig, axes
    
    def filter_core_rois(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter dataframe to only include core ROIs.
        
        Parameters
        ----------
        df : pd.DataFrame
            Classified dataframe from classify()
            
        Returns
        -------
        df_core : pd.DataFrame
            Filtered dataframe with only core ROIs
        """
        if 'edge_zone' not in df.columns:
            raise ValueError(
                "DataFrame does not have 'edge_zone' column. "
                "Did you forget to call classify()?"
            )
        return df[df['edge_zone'] == 'core'].copy()
    
    def get_edge_statistics(self, df: pd.DataFrame) -> dict:
        """
        Get detailed statistics about edge classification.
        
        Parameters
        ----------
        df : pd.DataFrame
            Classified dataframe from classify()
            
        Returns
        -------
        stats : dict
            Dictionary containing:
            - n_total, n_core, n_edge
            - pct_core, pct_edge
            - mean_distance, median_distance, std_distance
            - distance_by_plane (dict with xy, zx, zy statistics)
        """
        if 'edge_zone' not in df.columns:
            raise ValueError(
                "DataFrame does not have 'edge_zone' column. "
                "Did you forget to call classify()?"
            )
        
        n_total = len(df)
        n_core = (df['edge_zone'] == 'core').sum()
        n_edge = (df['edge_zone'] == 'edge').sum()
        
        stats = {
            'n_total': n_total,
            'n_core': n_core,
            'n_edge': n_edge,
            'pct_core': 100 * n_core / n_total,
            'pct_edge': 100 * n_edge / n_total,
            'mean_distance': df['min_distance_to_edge'].mean(),
            'median_distance': df['min_distance_to_edge'].median(),
            'std_distance': df['min_distance_to_edge'].std(),
            'distance_by_plane': {}
        }
        
        for plane in ['xy', 'zx', 'zy']:
            col = f'distance_to_edge_{plane}'
            if col in df.columns:
                stats['distance_by_plane'][plane] = {
                    'mean': df[col].mean(),
                    'median': df[col].median(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max()
                }
        
        return stats


# ---
# Cell x gene fitlers
# ---

def get_inhibitory_mask(cxg_df, gene_thresholds=None):
    """
    Get mask for inhibitory cells based on gene thresholds (OR logic).
    A cell is inhibitory if it has >= threshold for ANY specified gene.
    
    Parameters
    ----------
    cxg_df : pd.DataFrame
        Cell x gene DataFrame
    gene_thresholds : dict, optional
        Dictionary mapping gene names to thresholds.
        Default: {'Gad2': 50, 'Npy': 50, 'Pvalb': 50, 'Sst': 50, 'Vip': 50}
    
    Returns
    -------
    mask : pd.Series
        Boolean mask for inhibitory cells
    genes_found : list
        List of gene criteria that were applied
    """
    if gene_thresholds is None:
        gene_thresholds = {'Gad2': 50, 'Npy': 50, 'Pvalb': 50, 'Sst': 50, 'Vip': 50}
    
    mask = pd.Series(False, index=cxg_df.index)
    genes_found = []
    
    for gene, threshold in gene_thresholds.items():
        if gene in cxg_df.columns:
            gene_mask = cxg_df[gene] >= threshold
            mask = mask | gene_mask
            genes_found.append(f"{gene}>={threshold}")
    
    return mask, genes_found