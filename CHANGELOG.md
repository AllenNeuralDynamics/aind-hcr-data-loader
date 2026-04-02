# Changelog

**v0.7.0 (04/02/2026)**

*New module: `codeocean_utils.py`*
+ `create_client_from_env()` — builds a `CodeOcean` API client from environment variables
+ `MouseRecord` dataclass — typed wrapper around an ophys-mfish-dataset-catalog mouse JSON record
+ `attach_mouse_record_to_capsule()`, `attach_mouse_record_to_pipeline()`, `attach_mouse_record_to_workstation()`, `attach_mouse_record()` — attach all data assets from a mouse record to a target CodeOcean resource
+ `print_attach_results()` — human-readable summary of attach results with dry-run support
+ Exported from `__init__` for top-level access
+ `tests/test_codeocean_utils.py` — comprehensive unit tests for all public functions

*`hcr_dataset.py`*
+ Added `_load_spots_for_round()` helper to consolidate per-round spot loading with `roi_filter_type` and `remove_fg_bg_cols` support
+ Checks for CxG aggregated files in subfolders to support pipeline output layout changes

*`pairwise_dataset.py`*
+ `PairwiseDataset` inherits base class cell-typing attributes correctly

*Examples*
+ `examples/workstation_attach_and_load.ipynb` — end-to-end notebook for attaching assets and loading data on a CodeOcean workstation

**v0.6.0 (03/18/2026)**

*New modules*
+ Added `pairwise_dataset.py` — `create_pairwise_unmixing_dataset()` for loading pairwise-unmixing pipeline outputs
+ Added `cell_typing_dataset.py` — for attaching and querying cell-typing assets
+ Added `spot_filters.py` — spot-level filtering by channel-intensity percentiles, spatial isolation (kdtree), and spectral purity

*`hcr_dataset.py`*
+ Added `create_hcr_dataset_from_schema()` to construct an `HCRDataset` directly from an `ophys-mfish-dataset-catalog` JSON record
+ `create_hcr_dataset_from_config()` now auto-attaches cell-typing assets and accepts `metrics_base_path`
+ `HCRRound` gains a `parent_dataset` reference enabling advanced filtering from round-level calls
+ `HCRRound.load_spots()` gains `roi_filter_type` parameter (`'volume'` or `'comprehensive'`)
+ `HCRDataset` gains: `get_filtered_cell_ids()`, `create_cell_gene_matrix_from_spots()`, `load_taxonomy_cell_types()`, `load_taxonomy_cell_types_h5ad()`, `annotate_with_cell_types()`
+ `SpotFiles` replaces `processing_manifest` with `unmixed_cxg_filtered` and `mixed_cxg_filtered`
+ `get_spot_files()` resolves pkl files using round number from key to avoid artefact files
+ `get_processing_manifests()` checks both `derived/processing_manifest.json` and round root
+ `create_channel_gene_table_from_manifests()` adds `round_channel_gene` convenience column
+ `create_channel_gene_table()` parameter renamed `spot_files` → `processing_manifests`

*`filters.py`*
+ `roi_filter_soma_and_overlap()` renamed to `roi_filter_comprehensive()` with updated signature
+ Added `filter_cell_info()` — volume-quantile filter on cell-info DataFrames
+ Added `get_inhibitory_mask()` — boolean mask for inhibitory cells based on per-gene thresholds

*Dependencies*
+ Added `pyarrow` as a core dependency

**v0.5.1 (02/23/2026)**
+ Added `metrics_base_path` parameter to `HCRDataset` for soma shape classifier
+ Added `spot_filters.py` with vectorized percentile filtering and spot quality assessment
+ Enhanced `filters.py` with comprehensive ROI filtering pipeline
+ Added `hcr_filters.ipynb` notebook for filtering examples

**v0.4.0 (10/16/2025)**
+ Extract tile overlap boundaries
+ ROI overlap calculation and filtering, duplicate bbox plotting
+ Linear unmixing in single cell plots
+ New dye line plots and pairwise intensity plots
+ Added `constants.py` for channel colormaps
+ Gather neuroglancer links
+ Flexible figure saving via `saveable_plot` decorator

**v0.3.8 (8/11/2025)**
+ Updated how cell info is gathered from segmentation sources
+ Metrics path to segmentation files
+ Better error handling when missing spot files
