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

def roi_filter_soma_and_overlap(ds: HCRDataset):

    # always get R1 rois, since most complete set and metrics were calculated there
    dataset_name = ds.rounds["R1"].name
    metrics_path = Path(f"/root/capsule/scratch/shape_metrics/{dataset_name}/seg_shape_metrics_pyr2.parquet")

    #
    metrics_df = pd.read_parquet(metrics_path)
    print(f"Total cells in roi metrics parquet: {metrics_df.shape[0]}")
    metrics_df, feat_cols = soma.build_features(metrics_df)
    metrics_df = soma.volume_filter(metrics_df)
    volume_ids = metrics_df.cell_id.values

    bundle = soma.load_soma_classifier()

    roi_classifier_df = soma.predict_soma_labels(
        df=metrics_df,
        pipeline=bundle['pipeline'],
        feat_cols=bundle['feat_cols'],
        threshold=0.8
    )

    # get cell_ids predicted_soma is True
    soma_ids = roi_classifier_df.loc[roi_classifier_df.predicted_soma==True].cell_id.values
    not_soma_ids = roi_classifier_df.loc[roi_classifier_df.predicted_soma==False].cell_id.values

    # boudary filter
    roi_upscale_df, ids_in_overlap = filter_tile_boundary_rois(ds,"R1")

    combined_ids = np.unique(np.concatenate([ids_in_overlap, not_soma_ids]))
    print(f"Combined unique IDs: {combined_ids.shape}")

    # Info printing
    total_before = len(ids_in_overlap) + len(not_soma_ids)
    total_after = len(combined_ids)
    n_duplicates = total_before - total_after

    print(f"IDs in overlap: {len(ids_in_overlap)}")
    print(f"IDs not soma: {len(not_soma_ids)}")
    print(f"Total before merge: {total_before}")
    print(f"Unique combined: {total_after}")
    print(f"Duplicates removed: {n_duplicates}")

    return combined_ids, roi_classifier_df, roi_upscale_df

def filter_tile_boundary_rois(ds: HCRDataset,
                              round_key: str):
    """
    Note: must supply recalculated metrics from GPU loader, which has the correct bboxes.


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

    # in order to upscale, different factors. I forget why, but centroids aren't used.
    # centroids need 16x from 2->0
    # bbox needs 4x from 2->0
    df_up[centroid_cols] = df_up[centroid_cols] * 16
    df_up[bbox_cols] = df_up[bbox_cols] * 4 

    filtered_df, filtered_ids = ta.filter_rois_in_overlap_regions(
        df_up, 
        overlap_regions, 
        overlap_threshold=0.1,
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
