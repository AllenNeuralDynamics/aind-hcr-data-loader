"""
Spot filtering functions for HCR spatial transcriptomics data.

This module provides utilities for filtering spots based on:
1. Channel intensity percentiles (vectorized, fast)
2. Spatial isolation and neighbor density (kdtree-based)
3. Spectral purity (channel separation)

The module uses a boolean column approach for memory efficiency,
adding filter flags to the DataFrame rather than creating copies.

Functions are separated into:
- Data processing functions (add columns, compute masks)
- Visualization functions (plotting)
- Convenience wrappers (end-to-end pipeline)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# NOTE: The saveable_plot decorator should be imported from aind_hcr_qc
# but we'll make it optional in case that package isn't installed
try:
    from aind_hcr_qc.utils.utils import saveable_plot
    from aind_hcr_qc.viz import spectral_unmixing as su
    _HAVE_VIZ = True
except ImportError:
    _HAVE_VIZ = False
    # Fallback decorator that does nothing
    def saveable_plot(func):
        return func

# TODO: Move clean_spots functions here
# NOTE: The compute_clean_mask_kdtree function is currently in:
#       /root/capsule/code/clean_spots.py
# 
# MIGRATION PLAN:
# 1. Copy the following functions from clean_spots.py to this module:
#    - _channel_cols()
#    - _anisotropic_scale()
#    - _nn_distance_and_counts()
#    - compute_clean_mask_kdtree()
# 
# 2. Update imports in notebooks/scripts:
#    OLD: import clean_spots
#         clean_spots.compute_clean_mask_kdtree(...)
#    NEW: from aind_hcr_data_loader import spot_filters
#         spot_filters.compute_clean_mask_kdtree(...)
# 
# 3. Keep clean_spots.py for backwards compatibility initially
#    (can be deprecated later)
#
# For now, we'll import from clean_spots if available


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def filter_top_percentile_vectorized(
    df: pd.DataFrame,
    percentile: int = 90,
    chan_col: str = 'chan',
    intensity_col_pattern: str = 'chan_{}_intensity'
) -> pd.Series:
    """
    Vectorized filtering for spots in the top percentile of their assigned channel.
    
    This is ~100x faster than row-wise apply() approach by using pandas
    vectorized operations.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with channel assignment and intensity columns
    percentile : int, default=90
        Percentile threshold (0-100). Spots with intensity >= this
        percentile for their assigned channel will pass the filter.
    chan_col : str, default='chan'
        Column name containing channel assignments (e.g., 488, 561, 647)
    intensity_col_pattern : str, default='chan_{}_intensity'
        Pattern for intensity column names. '{}' will be replaced with
        channel value from chan_col.
    
    Returns
    -------
    pd.Series
        Boolean mask (same index as df) where True indicates spot passes
        the percentile threshold for its assigned channel.
    
    Examples
    --------
    >>> # Filter for top 90% brightest spots in each channel
    >>> mask = filter_top_percentile_vectorized(spots_df, percentile=90)
    >>> bright_spots = spots_df[mask]
    
    >>> # Filter for top 50% (median and above)
    >>> mask = filter_top_percentile_vectorized(spots_df, percentile=50)
    """
    # Get unique channels
    unique_chans = df[chan_col].unique()
    
    # Initialize mask as all False
    mask = pd.Series(False, index=df.index)
    
    # For each channel, vectorized filter
    for chan in unique_chans:
        # Build the intensity column name
        chan_intensity_col = intensity_col_pattern.format(chan)
        
        if chan_intensity_col not in df.columns:
            print(f"Warning: Column {chan_intensity_col} not found, skipping channel {chan}")
            continue
        
        # Get all rows for this channel (vectorized boolean indexing)
        chan_mask = df[chan_col] == chan
        
        # Calculate the percentile threshold for this channel
        threshold = df.loc[chan_mask, chan_intensity_col].quantile(percentile / 100.0)
        
        # Vectorized comparison: mark spots that meet the threshold
        # Use bitwise OR to combine with existing mask
        mask = mask | (chan_mask & (df[chan_intensity_col] >= threshold))
    
    return mask


# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

def add_spot_filter_columns(
    spots_df: pd.DataFrame,
    percentile_all: int = 90,
    percentile_clean: int = 50,
    clean_params: Optional[Dict] = None,
    clean_spots_func = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Add boolean filter columns to spots dataframe.
    
    This function adds three boolean columns indicating which spots pass
    different filtering criteria:
    - passes_percentile_{percentile_all}: Top N% of channel intensity
    - passes_clean: Passes clean_spots kdtree filter  
    - passes_clean_percentile_{percentile_clean}: Top N% after cleaning
    
    Parameters
    ----------
    spots_df : pd.DataFrame
        Input spots dataframe (modified in-place with new columns)
    percentile_all : int, default=90
        Percentile threshold for raw spots (0-100)
    percentile_clean : int, default=50
        Percentile threshold for cleaned spots (0-100)
    clean_params : dict, optional
        Parameters for clean_spots.compute_clean_mask_kdtree().
        If None, uses sensible defaults.
    clean_spots_func : callable, optional
        Function to compute clean mask. If None, tries to import
        from clean_spots module.
    verbose : bool, default=True
        Print filtering statistics
    
    Returns
    -------
    pd.DataFrame
        Same dataframe with added boolean columns:
        - 'passes_percentile_{percentile_all}'
        - 'passes_clean'
        - 'passes_clean_percentile_{percentile_clean}'
    
    Examples
    --------
    >>> # Add filter columns with defaults
    >>> spots_df = add_spot_filter_columns(spots_df)
    >>> 
    >>> # Use custom parameters
    >>> spots_df = add_spot_filter_columns(
    ...     spots_df,
    ...     percentile_all=95,
    ...     percentile_clean=60,
    ...     clean_params={'min_top_frac': 0.6, 'r_in': 2.0}
    ... )
    >>> 
    >>> # Access filtered data
    >>> high_quality = spots_df[spots_df['passes_clean_percentile_50']]
    """
    # Default clean_spots parameters
    default_clean_params = {
        'voxel_size_xyz': (0.24, 0.24, 1.0),
        'R_iso': 0.1,
        'r_in': 1.0,
        'r_out': 8.0,
        'max_neighbors': 10,
        'min_top_frac': 0.3,
        'min_top_to_second': 0.1,
        'r_min': 0.3,
        'group_cols': None
    }
    if clean_params:
        default_clean_params.update(clean_params)
    
    # Try to get clean_spots function
    if clean_spots_func is None:
        try:
            import sys
            sys.path.insert(0, '/root/capsule/code')
            import clean_spots
            clean_spots_func = clean_spots.compute_clean_mask_kdtree
        except ImportError:
            raise ImportError(
                "Could not import clean_spots module. "
                "Please provide clean_spots_func parameter or ensure "
                "clean_spots.py is in the Python path."
            )
    
    n_total = len(spots_df)
    
    # Stage 1: Percentile filter on all spots
    col_name_p1 = f'passes_percentile_{percentile_all}'
    if verbose:
        print(f"Computing {col_name_p1}...")
    
    spots_df[col_name_p1] = filter_top_percentile_vectorized(
        spots_df, percentile=percentile_all
    )
    n_p1 = spots_df[col_name_p1].sum()
    
    # Stage 2: Clean spots filter
    if verbose:
        print("Computing passes_clean (kdtree spatial filter)...")
    
    spots_df['passes_clean'] = clean_spots_func(
        spots_df,
        **default_clean_params
    )
    n_clean = spots_df['passes_clean'].sum()
    
    # Stage 3: Percentile filter on clean spots only
    # Only calculate percentile among clean spots, then apply to full df
    if verbose:
        print(f"Computing passes_clean_percentile_{percentile_clean}...")
    
    clean_subset = spots_df[spots_df['passes_clean']]
    clean_percentile_mask = filter_top_percentile_vectorized(
        clean_subset, percentile=percentile_clean
    )
    
    # Create column with False by default
    col_name_p2 = f'passes_clean_percentile_{percentile_clean}'
    spots_df[col_name_p2] = False
    # Set True only for spots that pass both clean AND percentile
    spots_df.loc[clean_subset.index[clean_percentile_mask], col_name_p2] = True
    n_clean_p = spots_df[col_name_p2].sum()
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Spot Filtering Statistics")
        print(f"{'='*60}")
        print(f"Total spots:                            {n_total:>10,}")
        print(f"Passes {percentile_all}th percentile:               {n_p1:>10,} ({100*n_p1/n_total:>5.1f}%)")
        print(f"Passes clean filter:                    {n_clean:>10,} ({100*n_clean/n_total:>5.1f}%)")
        print(f"Passes clean + {percentile_clean}th percentile:       {n_clean_p:>10,} ({100*n_clean_p/n_total:>5.1f}%)")
        print(f"{'='*60}\n")
    
    return spots_df


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_spot_filtering_pipeline(
    spots_df: pd.DataFrame,
    mouse_id: str,
    round_key: str,
    percentile_all: int = 90,
    percentile_clean: int = 50,
    xlims: Tuple[float, float] = (0, 1000),
    ylims: Tuple[float, float] = (0, 1000),
    scale: str = 'linear',
    figsize: Tuple[float, float] = (20, 20),
    channel_label: str = 'chan',
    save: bool = False,
    output_dir: Optional[str] = None,
    filename_base: Optional[str] = None
) -> Tuple[Dict, np.ndarray]:
    """
    Create 4 separate plots showing spot filtering stages using boolean columns.
    
    NOTE: Due to how plot_filtered_intensities works (creates own figures),
    this function returns 4 SEPARATE figures instead of a combined 2x2 grid.
    
    Creates 4 individual plots:
    1. All spots (baseline)
    2. All spots at percentile threshold
    3. Clean spots only
    4. Clean spots at percentile threshold
    
    Parameters
    ----------
    spots_df : pd.DataFrame
        DataFrame with boolean filter columns created by add_spot_filter_columns()
    mouse_id : str
        Mouse identifier for titles
    round_key : str
        Round identifier for titles
    percentile_all : int, default=90
        Which percentile column to use for all spots (must match column name)
    percentile_clean : int, default=50
        Which percentile column to use for clean spots (must match column name)
    xlims : tuple, default=(0, 1000)
        X-axis limits for all subplots
    ylims : tuple, default=(0, 1000)
        Y-axis limits for all subplots
    scale : str, default='linear'
        'linear' or 'log' scale for plotting
    figsize : tuple, default=(20, 20)
        Figure size (width, height) - NOTE: currently not used, each figure has its own size
    channel_label : str, default='chan'
        Column name for channel labels
    save : bool, default=False
        Whether to save the figures
    output_dir : str, optional
        Directory to save figures. If None and save=True, uses current directory.
    filename_base : str, optional
        Base filename for saved figures (without extension).
        If None, uses "{mouse_id}_{round_key}_spot_filtering".
        Each figure gets a suffix: _all, _p{N}, _clean, _clean_p{N}
    
    Returns
    -------
    dict
        Dictionary with keys:
        - 'fig_all': Figure list for all spots
        - 'fig_percentile': Figure list for percentile-filtered spots
        - 'fig_clean': Figure list for clean spots
        - 'fig_clean_percentile': Figure list for clean + percentile spots
        - 'filepaths': List of saved file paths (if save=True)
    axes : ndarray
        Placeholder for backwards compatibility (not used)
    
    Notes
    -----
    This function requires the aind_hcr_qc package for visualization.
    The boolean columns must exist in spots_df before calling this function.
    
    Each figure in the returned dict is actually a list [figure_obj] as 
    returned by plot_filtered_intensities().
    
    Examples
    --------
    >>> # After adding filter columns
    >>> figs, _ = plot_spot_filtering_pipeline(
    ...     spots_df,
    ...     mouse_id='785054-v1',
    ...     round_key='R2',
    ...     save=True,
    ...     output_dir='/results/spot_qc'
    ... )
    >>> # Access saved paths
    >>> print(figs['filepaths'])
    """
    if not _HAVE_VIZ:
        raise ImportError(
            "aind_hcr_qc package required for visualization. "
            "Install with: pip install aind-hcr-qc"
        )
    
    # Column names
    col_p1 = f'passes_percentile_{percentile_all}'
    col_clean = 'passes_clean'
    col_p2 = f'passes_clean_percentile_{percentile_clean}'
    
    # Check required columns exist
    required_cols = [col_p1, col_clean, col_p2]
    missing = [c for c in required_cols if c not in spots_df.columns]
    if missing:
        raise ValueError(
            f"Missing required filter columns: {missing}. "
            f"Run add_spot_filter_columns() first."
        )
    
    # Create figure with 2x2 grid
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    axes = np.array([
        [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    ])
    
    # Get counts for titles
    n_total = len(spots_df)
    n_p1 = spots_df[col_p1].sum()
    n_clean = spots_df[col_clean].sum()
    n_p2 = spots_df[col_p2].sum()
    
    # Setup filenames and output directory
    if save:
        if output_dir is None:
            output_dir = "."
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if filename_base is None:
            filename_base = f"{mouse_id}_{round_key}_spot_filtering"
    
    # Create 4 separate figures using plot_filtered_intensities
    # Each call will handle its own saving via @saveable_plot decorator
    
    print("Creating individual figures for each filtering stage...")
    print(f"  [1/4] All spots (n={n_total:,})...")
    fig_all = su.plot_filtered_intensities(
        spots_df,
        plot_cell_ids=None,
        xlims=xlims,
        ylims=ylims,
        scale=scale,
        channel_label=channel_label,
        title=f"{mouse_id} - {round_key} - All Spots\nn={n_total:,}",
        save=save,
        filename=str(output_path / f"{filename_base}_all") if save else None
    )
    
    print(f"  [2/4] Top {percentile_all}th percentile (n={n_p1:,})...")
    fig_percentile = su.plot_filtered_intensities(
        spots_df[spots_df[col_p1]],
        plot_cell_ids=None,
        xlims=xlims,
        ylims=ylims,
        scale=scale,
        channel_label=channel_label,
        title=f"{mouse_id} - {round_key} - Top {percentile_all}th Percentile\n"
              f"n={n_p1:,} ({100*n_p1/n_total:.1f}%)",
        save=save,
        filename=str(output_path / f"{filename_base}_p{percentile_all}") if save else None
    )
    
    print(f"  [3/4] Clean spots (n={n_clean:,})...")
    fig_clean = su.plot_filtered_intensities(
        spots_df[spots_df[col_clean]],
        plot_cell_ids=None,
        xlims=xlims,
        ylims=ylims,
        scale=scale,
        channel_label=channel_label,
        title=f"{mouse_id} - {round_key} - Clean Spots\n"
              f"n={n_clean:,} ({100*n_clean/n_total:.1f}%)",
        save=save,
        filename=str(output_path / f"{filename_base}_clean") if save else None
    )
    
    print(f"  [4/4] Clean + top {percentile_clean}th percentile (n={n_p2:,})...")
    fig_clean_percentile = su.plot_filtered_intensities(
        spots_df[spots_df[col_p2]],
        plot_cell_ids=None,
        xlims=xlims,
        ylims=ylims,
        scale=scale,
        channel_label=channel_label,
        title=f"{mouse_id} - {round_key} - Clean + Top {percentile_clean}th Percentile\n"
              f"n={n_p2:,} ({100*n_p2/n_total:.1f}%)",
        save=save,
        filename=str(output_path / f"{filename_base}_clean_p{percentile_clean}") if save else None
    )
    
    # Collect filepaths if saved
    filepaths = []
    if save:
        filepaths = [
            str(output_path / f"{filename_base}_all.png"),
            str(output_path / f"{filename_base}_p{percentile_all}.png"),
            str(output_path / f"{filename_base}_clean.png"),
            str(output_path / f"{filename_base}_clean_p{percentile_clean}.png")
        ]
    
    # Close the empty grid figure since we're returning the individual figures
    plt.close(fig)
    
    # Return a dict with all 4 figures
    return {
        'fig_all': fig_all,
        'fig_percentile': fig_percentile,
        'fig_clean': fig_clean,
        'fig_clean_percentile': fig_clean_percentile,
        'filepaths': filepaths
    }, axes  # Keep axes for backwards compatibility but they're not used


# ============================================================================
# CONVENIENCE WRAPPER
# ============================================================================

def process_and_plot_spot_filtering(
    ds,  # HCRDataset object
    round_key: str = "R2",
    cell_id_range: Tuple[int, int] = (1, 20000),
    percentile_all: int = 90,
    percentile_clean: int = 50,
    clean_params: Optional[Dict] = None,
    save: bool = True,
    output_dir: str = "/root/capsule/scratch/spot_filtering",
    xlims: Tuple[float, float] = (0, 1000),
    ylims: Tuple[float, float] = (0, 1000),
    scale: str = 'linear',
    figsize: Tuple[float, float] = (20, 20),
    **plot_kwargs
) -> Dict:
    """
    Complete pipeline: load data, filter, and plot.
    
    This is a convenience function that combines:
    1. Data loading from HCRDataset
    2. add_spot_filter_columns() for processing
    3. plot_spot_filtering_pipeline() for visualization
    
    Parameters
    ----------
    ds : HCRDataset
        HCR dataset object
    round_key : str, default="R2"
        Round to process
    cell_id_range : tuple, default=(1, 20000)
        (min_cell_id, max_cell_id) for subsetting visualization
    percentile_all : int, default=90
        Percentile for all spots filtering
    percentile_clean : int, default=50
        Percentile for cleaned spots filtering
    clean_params : dict, optional
        Parameters for clean_spots
    save : bool, default=True
        Whether to save the figure
    output_dir : str, default="/root/capsule/scratch/spot_filtering"
        Directory for saved figures
    xlims, ylims, scale, figsize : plotting parameters
    **plot_kwargs : additional plotting arguments
    
    Returns
    -------
    dict
        {
            'spots_df': DataFrame with filter columns,
            'fig': matplotlib figure,
            'axes': subplot axes,
            'filepath': str (if saved),
            'stats': dict with filtering statistics
        }
    
    Examples
    --------
    >>> # Basic usage
    >>> results = process_and_plot_spot_filtering(
    ...     ds,
    ...     round_key="R2",
    ...     save=True
    ... )
    >>> print(f"Saved to: {results['filepath']}")
    >>> 
    >>> # Custom parameters
    >>> results = process_and_plot_spot_filtering(
    ...     ds,
    ...     round_key="R2",
    ...     cell_id_range=(1, 50000),
    ...     percentile_all=95,
    ...     percentile_clean=60,
    ...     clean_params={'min_top_frac': 0.6},
    ...     xlims=(-50, 500),
    ...     ylims=(-50, 500),
    ...     output_dir='figures/qc'
    ... )
    """
    print("="*60)
    print("Spot Filtering Pipeline")
    print("="*60)
    print(f"Dataset: {ds.mouse_id}")
    print(f"Round: {round_key}")
    print(f"Cell ID range: {cell_id_range}")
    print("="*60)
    
    # Step 1: Load spots with comprehensive ROI filtering
    print("\n[1/3] Loading spots...")
    spots_df = ds.load_all_rounds_spots_mp(
        table_type='mixed_spots',
        roi_filter_type='comprehensive'
    )
    print(f"Loaded {len(spots_df):,} spots across all rounds")
    
    # Step 2: Add filter columns
    print("\n[2/3] Adding filter columns...")
    spots_df = add_spot_filter_columns(
        spots_df,
        percentile_all=percentile_all,
        percentile_clean=percentile_clean,
        clean_params=clean_params,
        verbose=True
    )
    
    # Subset for plotting
    min_cell, max_cell = cell_id_range
    plot_cell_ids = list(range(min_cell, max_cell + 1))
    spots_df_plot = spots_df[
        (spots_df["round"] == round_key) &
        (spots_df["cell_id"].isin(plot_cell_ids))
    ].copy()
    
    print(f"\nSubset for plotting: {len(spots_df_plot):,} spots "
          f"(round={round_key}, cells {min_cell}-{max_cell})")
    
    # Step 3: Plot
    print("\n[3/3] Creating visualizations...")
    figs_dict, _ = plot_spot_filtering_pipeline(
        spots_df_plot,
        mouse_id=ds.mouse_id,
        round_key=round_key,
        percentile_all=percentile_all,
        percentile_clean=percentile_clean,
        xlims=xlims,
        ylims=ylims,
        scale=scale,
        figsize=figsize,
        save=save,
        output_dir=output_dir,
        **plot_kwargs
    )
    
    # Collect statistics
    col_p1 = f'passes_percentile_{percentile_all}'
    col_p2 = f'passes_clean_percentile_{percentile_clean}'
    
    stats = {
        'n_total': len(spots_df),
        'n_percentile_all': spots_df[col_p1].sum(),
        'n_clean': spots_df['passes_clean'].sum(),
        'n_clean_percentile': spots_df[col_p2].sum(),
        'pct_percentile_all': 100 * spots_df[col_p1].sum() / len(spots_df),
        'pct_clean': 100 * spots_df['passes_clean'].sum() / len(spots_df),
        'pct_clean_percentile': 100 * spots_df[col_p2].sum() / len(spots_df),
    }
    
    # Print saved filepaths if applicable
    if save and figs_dict.get('filepaths'):
        print("\nSaved figures:")
        for fp in figs_dict['filepaths']:
            print(f"  {fp}")
    
    print("\n" + "="*60)
    print("Pipeline complete!")
    print("="*60)
    
    return {
        'spots_df': spots_df,
        'spots_df_plot': spots_df_plot,
        'figures': figs_dict,
        'filepaths': figs_dict.get('filepaths', []),
        'stats': stats
    }
