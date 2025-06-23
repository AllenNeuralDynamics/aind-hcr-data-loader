import dask.array as da
import numpy as np


class TileData:
    """
    A class for lazily loading and manipulating tile data with flexible slicing and projection options.

    This class maintains the original dask array for memory efficiency and only computes data when needed.
    It provides methods to access data in different orientations (XY, ZY, ZX) and to perform projections.
    """

    def __init__(self, tile_name, bucket_name, dataset_path, pyramid_level=0):
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

    def _load_lazy(self):
        """Lazily load the data as a dask array without computing"""
        if not self._loaded:
            tile_array_loc = f"{self.dataset_path}{self.tile_name}/{self.pyramid_level}"
            zarr_path = f"s3://{self.bucket_name}/{tile_array_loc}"
            self._data = da.from_zarr(url=zarr_path, storage_options={"anon": False}).squeeze()
            self._loaded = True

            # Store shape information
            self.shape = self._data.shape
            # Assuming zarr is stored in (z,y,x) order
            self.z_dim, self.y_dim, self.x_dim = self.shape

    @property
    def data(self):
        """Get the full computed data in (x,y,z) order"""
        self._load_lazy()
        return self._data.compute().transpose(2, 1, 0)

    @property
    def data_raw(self):
        """Get the full computed data in original (z,y,x) order"""
        self._load_lazy()
        return self._data.compute()

    @property
    def dask_array(self):
        """Get the underlying dask array without computing"""
        self._load_lazy()
        return self._data

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
        self._load_lazy()

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
        self._load_lazy()

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
        self._load_lazy()

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
        self._load_lazy()

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
        return self

    def calculate_max_slice(self, level_to_use=2):
        """

        Use pyramidal level 3 and calulate the mean of the slices in all 3 dimensions,
        report back using the index for all pyramid levels.

        scale = int(2**pyramid_level)

        Help to get estimates of where lots of signal is in the tile.

        """
        level_to_use = level_to_use
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

        pyramid_levels = [0, 1, 2, 3]

        max_slices[level_to_use] = {
            "z": int(max_slice_z_index[0]),
            "y": int(max_slice_y_index[0]),
            "x": int(max_slice_x_index[0]),
        }

        # remove level_to_use from pyramid_levels
        pyramid_levels.remove(level_to_use)

        for level in pyramid_levels:
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

        # sort keys by int value
        max_slices = dict(sorted(max_slices.items(), key=lambda item: int(item[0])))

        return max_slices
