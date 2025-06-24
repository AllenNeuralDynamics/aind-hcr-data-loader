# TileData Arithmetic Operations

This document describes the arithmetic operations that have been added to the `TileData` class to enable mathematical operations between compatible tile datasets.

## Overview

The `TileData` class now supports element-wise arithmetic operations between two `TileData` objects or between a `TileData` object and a scalar value. All operations ensure that the tiles are compatible (same shape and dimension order) before performing calculations.

## Supported Operations

### 1. Basic Arithmetic Operators

#### Addition (`+`, `+=`)
```python
result = tile1 + tile2        # Element-wise addition, returns new TileData
tile1 += tile2               # In-place addition
result = tile1 + 10          # Scalar addition
```

#### Subtraction (`-`, `-=`)
```python
result = tile1 - tile2        # Element-wise subtraction, returns new TileData
tile1 -= tile2               # In-place subtraction
result = tile1 - 5           # Scalar subtraction
```

#### Multiplication (`*`, `*=`)
```python
result = tile1 * tile2        # Element-wise multiplication, returns new TileData
tile1 *= tile2               # In-place multiplication
result = tile1 * 0.5         # Scalar multiplication
```

#### Division (`/`, `/=`)
```python
result = tile1 / tile2        # Element-wise division, returns new TileData
tile1 /= tile2               # In-place division
result = tile1 / 3           # Scalar division
```

### 2. Convenience Methods

#### Average
```python
result = tile1.average(tile2)     # Returns (tile1 + tile2) / 2
```

#### Sum
```python
result = tile1.sum_with(tile2)    # Same as tile1 + tile2
```

#### Difference
```python
result = tile1.difference(tile2)  # Same as tile1 - tile2
```

#### Absolute Difference
```python
result = tile1.abs_difference(tile2)  # Returns |tile1 - tile2|
```

## Compatibility Checking

Before any operation, the system verifies that two `TileData` objects are compatible:

- **Same shape**: All dimensions (Z, Y, X) must match
- **Same dimension order**: The transpose state must be identical
- **Same dimension attributes**: `z_dim`, `y_dim`, `x_dim` must all match

If tiles are incompatible, a `ValueError` is raised with detailed information about the mismatch.

## Channel Averaging Use Case

The primary motivation for these operations was to enable averaging across multiple imaging channels, as used in the `figure_tile_overlap_4_slices` function:

```python
# Average across spot channels
spots_channels = ["488", "514", "561", "594", "638"]
for ch in spots_channels:
    tile1_name_ch = tile1_name.replace('_ch_405.zarr', f'_ch_{ch}.zarr')
    tile2_name_ch = tile2_name.replace('_ch_405.zarr', f'_ch_{ch}.zarr')
    
    if ch == spots_channels[0]:
        tile1 = TileData(tile1_name_ch, bucket_name, data["dataset_path"], pyramid_level=pyramid_level)
        tile2 = TileData(tile2_name_ch, bucket_name, data["dataset_path"], pyramid_level=pyramid_level)
    else:
        tile1 += TileData(tile1_name_ch, bucket_name, data["dataset_path"], pyramid_level=pyramid_level)
        tile2 += TileData(tile2_name_ch, bucket_name, data["dataset_path"], pyramid_level=pyramid_level)

# Average the accumulated channels
tile1 /= len(spots_channels)
tile2 /= len(spots_channels)
```

## Memory Efficiency

- All operations work with Dask arrays for memory efficiency
- Results maintain the lazy evaluation characteristics of the original data
- New `TileData` objects are created for non-in-place operations, preserving the original data

## Error Handling

The system provides clear error messages for common issues:

- **Type errors**: When trying to operate with incompatible types
- **Shape mismatches**: When tiles have different dimensions
- **Dimension order mismatches**: When tiles have been transposed differently

## Examples

See the example files for comprehensive demonstrations:
- `examples/test_arithmetic_operations.py` - Basic functionality tests
- `examples/test_channel_averaging.py` - Channel averaging simulation
- `examples/arithmetic_operations_demo.py` - Complete documentation with examples

## Implementation Details

The arithmetic operations are implemented using Python's magic methods (`__add__`, `__iadd__`, etc.) and include:

- Compatibility checking via `_check_compatibility()`
- Efficient object copying via `_create_copy()`
- Support for both tile-to-tile and tile-to-scalar operations
- Proper handling of all tile metadata (shape, dimensions, etc.)
