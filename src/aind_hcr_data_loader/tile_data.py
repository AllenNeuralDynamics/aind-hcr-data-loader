"""Tile data loading and manipulation utilities for HCR datasets."""

import dask.array as da
import numpy as np


# add custom exceptions PyramidDoesNotExist
class PyramidDoesNotExist(Exception):
    """Exception raised when a requested pyramid level does not exist."""

    def __init__(self, message):
        """Initialize the exception with a message.

        Args:
            message (str): Error message describing the missing pyramid level.
        """
        super().__init__(message)


class TileData:
    """
    A class for lazily loading and manipulating tile data with flexible slicing and projection options.

    This class maintains the original dask array for memory efficiency and only computes data when needed.
    It provides methods to access data in different orientations (XY, ZY, ZX) and to perform projections.
    """

    def __init__(self, tile_name, bucket_name, dataset_path, pyramid_level=0, verbose=False):
        """
        Initialize the TileData object.

        Args:
            tile_name: Name of the tile
            bucket_name: S3 bucket name
            dataset_path: Path to dataset in bucket
            pyramid_level: Pyramid level to load (default 0)
        """
        self.tile_name = tile_name
        self.bucket_name = bucket_name
        self.dataset_path = dataset_path
        self.pyramid_level = pyramid_level
        self._data = None
        self._loaded = False
        self.dim_order = None  # Original zarr dimension order
        self.shape = None
        self.z_dim = None
        self.y_dim = None
        self.x_dim = None
        self.verbose = verbose

        self.connect()

    def _load_lazy(self):
        """Lazily load the data as a dask array without computing"""
        if not self._loaded:
            tile_array_loc = f"{self.dataset_path}{self.tile_name}/{self.pyramid_level}"
            zarr_path = f"s3://{self.bucket_name}/{tile_array_loc}"
            self._data = da.from_zarr(url=zarr_path, storage_options={"anon": False}).squeeze()
            self.shape = self._data.shape
            self.z_dim, self.y_dim, self.x_dim = self.shape
            self.dim_order = "ZYX"

            if len(self.shape) != 3:
                raise ValueError(
                    f"Tile data for {self.tile_name} must have 3 dimensions (Z, Y, X), "
                    f"got {len(self.shape)} dimensions instead."
                )

            print(
                f"Loaded tile {self.tile_name} at pyramid level {self.pyramid_level} with shape {self.shape}"
            )

    @property
    def data(self):
        """Get the full computed data"""
        self._loaded = True
        # return self._data.compute().transpose(2, 1, 0)
        return self._data.compute()

    @property
    def dask_array(self):
        """Get the full data as a dask array"""
        self._load_lazy()
        return self._data

    def transpose(self, axes=(2, 1, 0)):
        """
        Transpose the data array and update relevant attributes.

        Args:
            axes (tuple): Tuple specifying the permutation of axes.
                         Default (2, 1, 0) converts ZYX to XYZ order.

        Returns:
            self (for method chaining)
        """

        # Transpose the dask array
        self._data = self._data.transpose(axes)

        # Update shape and dimension attributes
        self.shape = self._data.shape

        # Update dimension order based on the transposition
        if axes == (2, 1, 0):
            # ZYX -> XYZ
            self.x_dim, self.y_dim, self.z_dim = self.shape
            self.dim_order = "XYZ"
        elif axes == (0, 1, 2):
            # Keep original ZYX order
            self.z_dim, self.y_dim, self.x_dim = self.shape
            self.dim_order = "ZYX"
        elif axes == (1, 0, 2):
            # ZYX -> YZX
            self.y_dim, self.z_dim, self.x_dim = self.shape
            self.dim_order = "YZX"
        elif axes == (0, 2, 1):
            # ZYX -> ZXY
            self.z_dim, self.x_dim, self.y_dim = self.shape
            self.dim_order = "ZXY"
        elif axes == (1, 2, 0):
            # ZYX -> YXZ
            self.y_dim, self.x_dim, self.z_dim = self.shape
            self.dim_order = "YXZ"
        elif axes == (2, 0, 1):
            # ZYX -> XZY
            self.x_dim, self.z_dim, self.y_dim = self.shape
            self.dim_order = "XZY"
        else:
            # Custom axes - update generically
            self.z_dim, self.y_dim, self.x_dim = self.shape
            self.dim_order = f"Custom{axes}"

        if self.verbose:
            print(f"Transposed data to shape {self.shape} with dimension order {self.dim_order}")

        return self

    # @property
    # def data_raw(self):
    #     """Get the full computed data in original zarr order """
    #     self._load_lazy()
    #     return self._data.compute()

    def connect(self):
        """Establish connection to the data source without computing"""
        self._load_lazy()
        return self

    def get_slice(self, index, orientation="xy", compute=True):
        """
        Get a 2D slice through the data in the specified orientation.

        Args:
            index: Index of the slice
            orientation: One of 'xy', 'zy', 'zx' (default 'xy')
            compute: Whether to compute the dask array (default True)

        Returns:
            2D numpy array or dask array
        """

        if orientation == "xy":
            # XY slice at specific Z
            if index >= self.z_dim:
                raise IndexError(f"Z index {index} out of bounds (max {self.z_dim-1})")
            slice_data = self._data[index, :, :]
        elif orientation == "zy":
            # ZY slice at specific X
            if index >= self.x_dim:
                raise IndexError(f"X index {index} out of bounds (max {self.x_dim-1})")
            slice_data = self._data[:, :, index]
        elif orientation == "zx":
            # ZX slice at specific Y
            if index >= self.y_dim:
                raise IndexError(f"Y index {index} out of bounds (max {self.y_dim-1})")
            slice_data = self._data[:, index, :]
        else:
            raise ValueError(f"Unknown orientation: {orientation}. Use 'xy', 'zy', or 'zx'")

        if compute:
            return slice_data.compute()
        return slice_data

    def get_slice_range(self, start, end, axis="z", compute=True):
        """
        Get a range of slices along the specified axis.

        Args:
            start: Start index (inclusive)
            end: End index (exclusive)
            axis: One of 'z', 'y', 'x' (default 'z')
            compute: Whether to compute the dask array (default True)

        Returns:
            3D numpy array or dask array
        """

        if axis == "z":
            if end > self.z_dim:
                raise IndexError(f"Z end index {end} out of bounds (max {self.z_dim})")
            slice_data = self._data[start:end, :, :]
        elif axis == "y":
            if end > self.y_dim:
                raise IndexError(f"Y end index {end} out of bounds (max {self.y_dim})")
            slice_data = self._data[:, start:end, :]
        elif axis == "x":
            if end > self.x_dim:
                raise IndexError(f"X end index {end} out of bounds (max {self.x_dim})")
            slice_data = self._data[:, :, start:end]
        else:
            raise ValueError(f"Unknown axis: {axis}. Use 'z', 'y', or 'x'")

        if compute:
            return slice_data.compute()
        return slice_data

    def project(self, axis="z", method="max", start=None, end=None, compute=True):
        """
        Project data along the specified axis using the specified method.

        Args:
            axis: One of 'z', 'y', 'x' (default 'z')
            method: One of 'max', 'mean', 'min', 'sum' (default 'max')
            start: Start index for projection range (default None = 0)
            end: End index for projection range (default None = full dimension)
            compute: Whether to compute the dask array (default True)

        Returns:
            2D numpy array or dask array
        """

        # Set default range
        if start is None:
            start = 0
        if end is None:
            if axis == "z":
                end = self.z_dim
            elif axis == "y":
                end = self.y_dim
            else:
                end = self.x_dim

        # Get the slice range
        range_data = self.get_slice_range(start, end, axis, compute=False)

        # Apply projection method
        if method == "max":
            if axis == "z":
                result = range_data.max(axis=0)
            elif axis == "y":
                result = range_data.max(axis=1)
            else:  # axis == 'x'
                result = range_data.max(axis=2)
        elif method == "mean":
            if axis == "z":
                result = range_data.mean(axis=0)
            elif axis == "y":
                result = range_data.mean(axis=1)
            else:  # axis == 'x'
                result = range_data.mean(axis=2)
        elif method == "min":
            if axis == "z":
                result = range_data.min(axis=0)
            elif axis == "y":
                result = range_data.min(axis=1)
            else:  # axis == 'x'
                result = range_data.min(axis=2)
        elif method == "sum":
            if axis == "z":
                result = range_data.sum(axis=0)
            elif axis == "y":
                result = range_data.sum(axis=1)
            else:  # axis == 'x'
                result = range_data.sum(axis=2)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'max', 'mean', 'min', or 'sum'")

        if compute:
            return result.compute()
        return result

    def get_orthogonal_views(self, z_index=None, y_index=None, x_index=None, compute=True):
        """
        Get orthogonal views (XY, ZY, ZX) at the specified indices.

        Args:
            z_index: Z index for XY view (default None = middle slice)
            y_index: Y index for ZX view (default None = middle slice)
            x_index: X index for ZY view (default None = middle slice)
            compute: Whether to compute the dask arrays (default True)

        Returns:
            dict with keys 'xy', 'zy', 'zx' containing the respective views
        """

        # Use middle slices by default
        if z_index is None:
            z_index = self.z_dim // 2
        if y_index is None:
            y_index = self.y_dim // 2
        if x_index is None:
            x_index = self.x_dim // 2

        # Get the three orthogonal views
        xy_view = self.get_slice(z_index, "xy", compute)
        zy_view = self.get_slice(x_index, "zy", compute)
        zx_view = self.get_slice(y_index, "zx", compute)

        return {"xy": xy_view, "zy": zy_view, "zx": zx_view}

    def set_pyramid_level(self, level: int):
        """
        Set the pyramid level and clear any loaded data.

        Args:
            level: New pyramid level to use

        Returns:
            self (for method chaining)
        """
        if level != self.pyramid_level:
            self.pyramid_level = level
            # Clear loaded data so it will be reloaded at new pyramid level
            self._data = None
            self._loaded = False
            # lazy load the new data
            self._load_lazy()
        return self

    def calculate_max_slice(self, level_to_use=2):
        """

        Use pyramidal level 2 and calculate the mean of the slices in all 3 dimensions,
        report back using the index for all pyramid levels.

        scale = int(2**pyramid_level)

        Help to get estimates of where lots of signal is in the tile.

        """
        self.set_pyramid_level(level_to_use)

        # first load the data
        data = self.data

        max_slices = {}
        # find index of max slice in z
        max_slice_z = data.mean(axis=0)
        max_slice_z_index = np.unravel_index(max_slice_z.argmax(), max_slice_z.shape)
        max_slice_y = data.mean(axis=1)
        max_slice_y_index = np.unravel_index(max_slice_y.argmax(), max_slice_y.shape)
        max_slice_x = data.mean(axis=2)
        max_slice_x_index = np.unravel_index(max_slice_x.argmax(), max_slice_x.shape)

        pyramid_levels = [0, 1, 2, 3, 4, 5]

        max_slices[level_to_use] = {
            "z": int(max_slice_z_index[0]),
            "y": int(max_slice_y_index[0]),
            "x": int(max_slice_x_index[0]),
        }

        # remove level_to_use from pyramid_levels
        pyramid_levels.remove(level_to_use)

        for level in pyramid_levels:
            try:
                if level_to_use >= level:
                    scale_factor = 2 ** (level_to_use - level)
                else:
                    print(f"level_to_use: {level_to_use}, level: {level}")
                    scale_factor = 1 / (2 ** (level - level_to_use))
                max_slices[level] = {
                    "z": int(max_slice_z_index[0] * scale_factor),
                    "y": int(max_slice_y_index[0] * scale_factor),
                    "x": int(max_slice_x_index[0] * scale_factor),
                }
            except PyramidDoesNotExist:
                # If the pyramid level does not exist, skip it
                pass

        # sort keys by int value
        max_slices = dict(sorted(max_slices.items(), key=lambda item: int(item[0])))

        return max_slices

    def _check_compatibility(self, other):
        """
        Check if two TileData objects are compatible for arithmetic operations.

        Args:
            other: Another TileData object

        Raises:
            ValueError: If the objects are not compatible
        """
        if not isinstance(other, TileData):
            raise TypeError(
                f"Cannot perform operation with {type(other)}. Expected TileData object."
            )

        # Check if both objects are loaded
        if not self._loaded:
            self._load_lazy()
        if not other._loaded:
            other._load_lazy()

        # Check shapes
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")

        # Check dimension order
        if self.dim_order != other.dim_order:
            raise ValueError(f"Dimension order mismatch: {self.dim_order} vs {other.dim_order}")

        # Check all dimension attributes
        if self.z_dim != other.z_dim or self.y_dim != other.y_dim or self.x_dim != other.x_dim:
            raise ValueError(
                f"Dimension mismatch: "
                f"Z({self.z_dim} vs {other.z_dim}), "
                f"Y({self.y_dim} vs {other.y_dim}), "
                f"X({self.x_dim} vs {other.x_dim})"
            )

    def __add__(self, other):
        """
        Add two TileData objects element-wise.

        Args:
            other: Another TileData object or scalar

        Returns:
            New TileData object with the sum
        """
        if isinstance(other, (int, float)):
            # Scalar addition
            result = self._create_copy()
            result._data = self._data + other
            return result

        self._check_compatibility(other)

        # Create a new TileData object with the same metadata
        result = self._create_copy()
        result._data = self._data + other._data

        return result

    def __sub__(self, other):
        """
        Subtract two TileData objects element-wise.

        Args:
            other: Another TileData object or scalar

        Returns:
            New TileData object with the difference
        """
        if isinstance(other, (int, float)):
            # Scalar subtraction
            result = self._create_copy()
            result._data = self._data - other
            return result

        self._check_compatibility(other)

        # Create a new TileData object with the same metadata
        result = self._create_copy()
        result._data = self._data - other._data

        return result

    def __mul__(self, other):
        """
        Multiply two TileData objects element-wise or by a scalar.

        Args:
            other: Another TileData object or scalar

        Returns:
            New TileData object with the product
        """
        if isinstance(other, (int, float)):
            # Scalar multiplication
            result = self._create_copy()
            result._data = self._data * other
            return result

        self._check_compatibility(other)

        # Create a new TileData object with the same metadata
        result = self._create_copy()
        result._data = self._data * other._data

        return result

    def __truediv__(self, other):
        """
        Divide two TileData objects element-wise or by a scalar.

        Args:
            other: Another TileData object or scalar

        Returns:
            New TileData object with the quotient
        """
        if isinstance(other, (int, float)):
            # Scalar division
            result = self._create_copy()
            result._data = self._data / other
            return result

        self._check_compatibility(other)

        # Create a new TileData object with the same metadata
        result = self._create_copy()
        result._data = self._data / other._data

        return result

    def __iadd__(self, other):
        """
        In-place addition.

        Args:
            other: Another TileData object or scalar

        Returns:
            Self (modified in place)
        """
        if isinstance(other, (int, float)):
            # Scalar addition
            self._data = self._data + other
            return self

        self._check_compatibility(other)
        self._data = self._data + other._data
        return self

    def __isub__(self, other):
        """
        In-place subtraction.

        Args:
            other: Another TileData object or scalar

        Returns:
            Self (modified in place)
        """
        if isinstance(other, (int, float)):
            # Scalar subtraction
            self._data = self._data - other
            return self

        self._check_compatibility(other)
        self._data = self._data - other._data
        return self

    def __imul__(self, other):
        """
        In-place multiplication.

        Args:
            other: Another TileData object or scalar

        Returns:
            Self (modified in place)
        """
        if isinstance(other, (int, float)):
            # Scalar multiplication
            self._data = self._data * other
            return self

        self._check_compatibility(other)
        self._data = self._data * other._data
        return self

    def __itruediv__(self, other):
        """
        In-place division.

        Args:
            other: Another TileData object or scalar

        Returns:
            Self (modified in place)
        """
        if isinstance(other, (int, float)):
            # Scalar division
            self._data = self._data / other
            return self

        self._check_compatibility(other)
        self._data = self._data / other._data
        return self

    def _create_copy(self):
        """
        Create a copy of this TileData object with the same metadata but no data loaded.

        Returns:
            New TileData object with same metadata
        """
        # Create a new object without calling __init__ to avoid S3 loading
        copy_obj = TileData.__new__(TileData)

        # Copy all attributes
        copy_obj.tile_name = self.tile_name
        copy_obj.bucket_name = self.bucket_name
        copy_obj.dataset_path = self.dataset_path
        copy_obj.pyramid_level = self.pyramid_level
        copy_obj.verbose = self.verbose
        copy_obj.shape = self.shape
        copy_obj.z_dim = self.z_dim
        copy_obj.y_dim = self.y_dim
        copy_obj.x_dim = self.x_dim
        copy_obj.dim_order = self.dim_order
        copy_obj._loaded = True
        copy_obj._data = None  # Will be set by the calling method

        return copy_obj

    def average(self, other):
        """
        Compute the element-wise average of two TileData objects.

        Args:
            other: Another TileData object

        Returns:
            New TileData object with the average
        """
        return (self + other) / 2

    def sum_with(self, other):
        """
        Compute the element-wise sum of two TileData objects.

        Args:
            other: Another TileData object

        Returns:
            New TileData object with the sum
        """
        return self + other

    def difference(self, other):
        """
        Compute the element-wise difference of two TileData objects.

        Args:
            other: Another TileData object

        Returns:
            New TileData object with the difference (self - other)
        """
        return self - other

    def abs_difference(self, other):
        """
        Compute the element-wise absolute difference of two TileData objects.

        Args:
            other: Another TileData object

        Returns:
            New TileData object with the absolute difference
        """
        self._check_compatibility(other)

        result = self._create_copy()
        result._data = da.abs(self._data - other._data)

        return result
