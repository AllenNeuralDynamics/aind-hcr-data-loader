# Architecture Overview

## Purpose
A Python package for efficiently loading and manipulating HCR (Hybridization Chain Reaction) data stored in S3 as OME-Zarr files. Provides lazy loading capabilities and convenient data access methods for large microscopy datasets.

## Repository Structure

```
src/aind_hcr_data_loader/
├── __init__.py          # Package initialization
├── s3_utils.py          # S3 connectivity and utilities
├── tile_data.py         # Core TileData class for individual tiles
├── hcr_dataset.py       # HCRDataset class for full datasets
└── examples/            # Usage examples and notebooks
```

## Core Components

### `s3_utils.py`
- S3 bucket connectivity checks
- Prefix existence validation
- Content listing utilities
- Uses `s3fs` for filesystem-like S3 operations

### `tile_data.py`
- `TileData` class for individual tile manipulation
- Lazy loading of Zarr data using Dask arrays
- Multi-dimensional slicing (XY, ZY, ZX orientations)
- Projection operations (max, mean, min, sum)
- Pyramid level support for multi-resolution data

### `hcr_dataset.py`
- `HCRDataset` class for managing collections of tiles
- Dataset-level operations and metadata handling
- Coordinate system management

## Development Tools
- **`pre_commit_checks.sh`**: Bash script for running all pre-commit checks
- **`pre_commit_checks.py`**: Python script for cross-platform pre-commit checks
- Both scripts run: black, isort, flake8, interrogate, and coverage tests
