#!/usr/bin/env python3
"""
Unit tests for TileData arithmetic operations and core functionality.
"""

import unittest

import dask.array as da
import numpy as np

from aind_hcr_data_loader.tile_data import TileData


class TileDataTestUtils:
    """Utility class for creating mock TileData objects in tests."""

    @staticmethod
    def create_mock_tile_data(
        name="mock_tile",
        shape=(5, 10, 15),
        data_multiplier=1.0,
        channel="405",
        pyramid_level=0,
        seed=None,
    ):
        """
        Create a mock TileData object for testing (without actual S3 data).

        Args:
            name: Base name for the tile
            shape: Shape of the mock data (z, y, x)
            data_multiplier: Factor to multiply the mock data by
            channel: Channel identifier
            pyramid_level: Pyramid level for the tile
            seed: Random seed for reproducible data

        Returns:
            TileData object with mock data
        """
        # Set random seed for reproducible tests
        if seed is not None:
            np.random.seed(seed)

        # Create a TileData object without calling __init__
        tile = TileData.__new__(TileData)

        # Set basic attributes
        tile.tile_name = f"{name}_ch_{channel}.zarr"
        tile.bucket_name = "mock_bucket"
        tile.dataset_path = "mock/path/"
        tile.pyramid_level = pyramid_level
        tile.verbose = False

        # Create mock data
        mock_data = np.random.rand(*shape).astype(np.float32) * data_multiplier
        tile._data = da.from_array(mock_data, chunks="auto")
        tile._loaded = True

        # Set dimension attributes
        tile.shape = shape
        tile.z_dim, tile.y_dim, tile.x_dim = shape
        tile.dim_order = "ZYX"

        return tile

    @staticmethod
    def create_channel_specific_tile(
        base_name, channel, shape=(5, 10, 15), data_multiplier=1.0, pyramid_level=1
    ):
        """
        Create a mock tile for a specific channel with channel-specific factors.

        Args:
            base_name: Base tile name to modify
            channel: Channel identifier ("488", "514", etc.)
            shape: Shape of the mock data
            data_multiplier: Base data multiplier
            pyramid_level: Pyramid level

        Returns:
            TileData object with channel-specific mock data
        """
        # Different factors for each channel to simulate real channel differences
        channel_factors = {"488": 1.2, "514": 0.8, "561": 1.5, "594": 0.9, "638": 1.1, "405": 1.0}
        factor = channel_factors.get(channel, 1.0)

        tile = TileData.__new__(TileData)
        tile.tile_name = base_name.replace("_ch_405.zarr", f"_ch_{channel}.zarr")
        tile.bucket_name = "mock_bucket"
        tile.dataset_path = "mock/path/"
        tile.pyramid_level = pyramid_level
        tile.verbose = False

        # Use a seed based on channel for reproducible but different data per channel
        channel_seed = hash(channel) % 1000
        np.random.seed(channel_seed)

        mock_data = np.random.rand(*shape).astype(np.float32) * data_multiplier * factor
        tile._data = da.from_array(mock_data, chunks="auto")
        tile._loaded = True

        tile.shape = shape
        tile.z_dim, tile.y_dim, tile.x_dim = shape
        tile.dim_order = "ZYX"

        return tile

    @staticmethod
    def assert_tiles_compatible(test_case, tile1, tile2):
        """
        Assert that two tiles are compatible for arithmetic operations.

        Args:
            test_case: unittest.TestCase instance
            tile1: First TileData object
            tile2: Second TileData object
        """
        test_case.assertEqual(tile1.shape, tile2.shape)
        test_case.assertEqual(tile1.z_dim, tile2.z_dim)
        test_case.assertEqual(tile1.y_dim, tile2.y_dim)
        test_case.assertEqual(tile1.x_dim, tile2.x_dim)
        test_case.assertEqual(tile1.dim_order, tile2.dim_order)

    @staticmethod
    def assert_arithmetic_result_valid(
        test_case, result, expected_shape, expected_dim_order="ZYX"
    ):
        """
        Assert that an arithmetic operation result is valid.

        Args:
            test_case: unittest.TestCase instance
            result: Result TileData object
            expected_shape: Expected shape tuple
            expected_dim_order: Expected dimension order string
        """
        test_case.assertIsInstance(result, TileData)
        test_case.assertEqual(result.shape, expected_shape)
        test_case.assertEqual(result.dim_order, expected_dim_order)
        test_case.assertTrue(result._loaded)
        test_case.assertIsNotNone(result._data)


class TestTileDataArithmetic(unittest.TestCase):
    """Test arithmetic operations on TileData objects."""

    def setUp(self):
        """Set up test fixtures with compatible tile objects."""
        self.shape = (5, 10, 15)
        self.tile1 = TileDataTestUtils.create_mock_tile_data("tile1", self.shape, 2.0, seed=42)
        self.tile2 = TileDataTestUtils.create_mock_tile_data("tile2", self.shape, 1.0, seed=24)

    def test_addition(self):
        """Test addition operation between two tiles."""
        result = self.tile1 + self.tile2

        # Check that result is a new TileData object
        TileDataTestUtils.assert_arithmetic_result_valid(self, result, self.shape, "ZYX")
        self.assertIsNot(result, self.tile1)
        self.assertIsNot(result, self.tile2)

        # Check data integrity
        expected_data = self.tile1.data + self.tile2.data
        np.testing.assert_array_almost_equal(result.data, expected_data)

    def test_subtraction(self):
        """Test subtraction operation between two tiles."""
        result = self.tile1 - self.tile2

        TileDataTestUtils.assert_arithmetic_result_valid(self, result, self.shape, "ZYX")

        expected_data = self.tile1.data - self.tile2.data
        np.testing.assert_array_almost_equal(result.data, expected_data)

    def test_multiplication(self):
        """Test multiplication operation between two tiles."""
        result = self.tile1 * self.tile2

        TileDataTestUtils.assert_arithmetic_result_valid(self, result, self.shape, "ZYX")

        expected_data = self.tile1.data * self.tile2.data
        np.testing.assert_array_almost_equal(result.data, expected_data)

    def test_division(self):
        """Test division operation between two tiles."""
        result = self.tile1 / self.tile2

        TileDataTestUtils.assert_arithmetic_result_valid(self, result, self.shape, "ZYX")

        expected_data = self.tile1.data / self.tile2.data
        np.testing.assert_array_almost_equal(result.data, expected_data)

    def test_scalar_operations(self):
        """Test operations with scalar values."""
        # Addition
        result_add = self.tile1 + 10
        expected_add = self.tile1.data + 10
        np.testing.assert_array_almost_equal(result_add.data, expected_add)
        TileDataTestUtils.assert_arithmetic_result_valid(self, result_add, self.shape)

        # Subtraction
        result_sub = self.tile1 - 5
        expected_sub = self.tile1.data - 5
        np.testing.assert_array_almost_equal(result_sub.data, expected_sub)

        # Multiplication
        result_mul = self.tile1 * 0.5
        expected_mul = self.tile1.data * 0.5
        np.testing.assert_array_almost_equal(result_mul.data, expected_mul)

        # Division
        result_div = self.tile1 / 2
        expected_div = self.tile1.data / 2
        np.testing.assert_array_almost_equal(result_div.data, expected_div)

    def test_in_place_operations(self):
        """Test in-place arithmetic operations."""
        # Create a copy for testing
        tile_copy = TileDataTestUtils.create_mock_tile_data("copy", self.shape, 3.0, seed=99)
        original_data = tile_copy.data.copy()

        # In-place addition
        tile_copy += self.tile1
        expected_add = original_data + self.tile1.data
        np.testing.assert_array_almost_equal(tile_copy.data, expected_add)

        # In-place subtraction
        tile_copy -= self.tile2
        expected_sub = expected_add - self.tile2.data
        np.testing.assert_array_almost_equal(tile_copy.data, expected_sub)

        # In-place multiplication
        tile_copy *= 0.8
        expected_mul = expected_sub * 0.8
        np.testing.assert_array_almost_equal(tile_copy.data, expected_mul)

        # In-place division
        tile_copy /= 3
        expected_div = expected_mul / 3
        np.testing.assert_array_almost_equal(tile_copy.data, expected_div)

        # Verify tile still has correct metadata
        self.assertEqual(tile_copy.shape, self.shape)
        self.assertEqual(tile_copy.dim_order, "ZYX")

    def test_convenience_methods(self):
        """Test convenience methods for common operations."""
        # Average
        avg_result = self.tile1.average(self.tile2)
        expected_avg = (self.tile1.data + self.tile2.data) / 2
        np.testing.assert_array_almost_equal(avg_result.data, expected_avg)
        TileDataTestUtils.assert_arithmetic_result_valid(self, avg_result, self.shape)

        # Sum
        sum_result = self.tile1.sum_with(self.tile2)
        expected_sum = self.tile1.data + self.tile2.data
        np.testing.assert_array_almost_equal(sum_result.data, expected_sum)

        # Difference
        diff_result = self.tile1.difference(self.tile2)
        expected_diff = self.tile1.data - self.tile2.data
        np.testing.assert_array_almost_equal(diff_result.data, expected_diff)

        # Absolute difference
        abs_diff_result = self.tile1.abs_difference(self.tile2)
        expected_abs_diff = np.abs(self.tile1.data - self.tile2.data)
        np.testing.assert_array_almost_equal(abs_diff_result.data, expected_abs_diff)

    def test_compatibility_checking(self):
        """Test that incompatible tiles raise appropriate errors."""
        # Create tile with different shape
        bad_tile = TileDataTestUtils.create_mock_tile_data("bad", (3, 8, 12), 1.0)

        with self.assertRaises(ValueError) as context:
            _ = self.tile1 + bad_tile
        self.assertIn("Shape mismatch", str(context.exception))

        # Test with wrong type
        with self.assertRaises(TypeError) as context:
            _ = self.tile1 + "invalid"
        self.assertIn("Cannot perform operation", str(context.exception))

    def test_dimension_preservation(self):
        """Test that all dimension attributes are preserved correctly."""
        result = self.tile1 + self.tile2

        self.assertEqual(result.z_dim, self.tile1.z_dim)
        self.assertEqual(result.y_dim, self.tile1.y_dim)
        self.assertEqual(result.x_dim, self.tile1.x_dim)
        self.assertEqual(result.dim_order, self.tile1.dim_order)
        self.assertEqual(result.shape, self.tile1.shape)

    def test_chained_operations(self):
        """Test that operations can be chained together."""
        # Create a third tile
        tile3 = TileDataTestUtils.create_mock_tile_data("tile3", self.shape, 0.5, seed=12)

        # Test chained operations
        result = (self.tile1 + self.tile2) * tile3 / 2

        TileDataTestUtils.assert_arithmetic_result_valid(self, result, self.shape)

        # Verify the calculation
        expected = (self.tile1.data + self.tile2.data) * tile3.data / 2
        np.testing.assert_array_almost_equal(result.data, expected)


class TestChannelAveraging(unittest.TestCase):
    """Test channel averaging functionality similar to the notebook use case."""

    def setUp(self):
        """Set up test fixtures for channel averaging."""
        self.shape = (5, 10, 15)
        self.spots_channels = ["488", "514", "561", "594", "638"]
        self.tile1_name_base = "Tile_X_0000_Y_0000_Z_0000_ch_405.zarr"
        self.tile2_name_base = "Tile_X_0001_Y_0000_Z_0000_ch_405.zarr"

    def test_channel_averaging_workflow(self):
        """Test the complete channel averaging workflow."""
        # Simulate tile1 averaging exactly like in the notebook
        tile1 = None
        channel_means = []

        for i, ch in enumerate(self.spots_channels):
            channel_tile = TileDataTestUtils.create_channel_specific_tile(
                self.tile1_name_base, ch, self.shape, 2.0
            )
            channel_means.append(channel_tile.data.mean())

            if i == 0:
                tile1 = channel_tile
            else:
                tile1 += channel_tile

        # Store pre-averaging mean for verification
        pre_avg_mean = tile1.data.mean()
        expected_avg_mean = sum(channel_means) / len(channel_means)

        # Average the tiles
        tile1 /= len(self.spots_channels)

        # Verify that averaging worked correctly
        self.assertAlmostEqual(tile1.data.mean(), expected_avg_mean, places=5)
        self.assertLess(tile1.data.mean(), pre_avg_mean)
        self.assertEqual(tile1.shape, self.shape)
        self.assertEqual(tile1.dim_order, "ZYX")

    def test_averaged_tiles_compatibility(self):
        """Test that averaged tiles remain compatible for further operations."""
        # Create and average tile1
        tile1 = None
        for i, ch in enumerate(self.spots_channels):
            channel_tile = TileDataTestUtils.create_channel_specific_tile(
                self.tile1_name_base, ch, self.shape, 2.0
            )
            if i == 0:
                tile1 = channel_tile
            else:
                tile1 += channel_tile
        tile1 /= len(self.spots_channels)

        # Create and average tile2
        tile2 = None
        for i, ch in enumerate(self.spots_channels):
            channel_tile = TileDataTestUtils.create_channel_specific_tile(
                self.tile2_name_base, ch, self.shape, 1.0
            )
            if i == 0:
                tile2 = channel_tile
            else:
                tile2 += channel_tile
        tile2 /= len(self.spots_channels)

        # Test that averaged tiles can still be used together
        TileDataTestUtils.assert_tiles_compatible(self, tile1, tile2)

        tile_sum = tile1 + tile2
        tile_diff = tile1 - tile2

        TileDataTestUtils.assert_arithmetic_result_valid(self, tile_sum, self.shape)
        TileDataTestUtils.assert_arithmetic_result_valid(self, tile_diff, self.shape)

    def test_channel_averaging_error_handling(self):
        """Test error handling during channel averaging."""
        # Start with a valid tile
        tile1 = TileDataTestUtils.create_channel_specific_tile(
            self.tile1_name_base, "488", self.shape, 2.0
        )

        # Try to add incompatible tile
        bad_tile = TileDataTestUtils.create_mock_tile_data("bad", (3, 8, 12), 1.0)

        with self.assertRaises(ValueError):
            tile1 += bad_tile

    def test_single_channel_averaging(self):
        """Test averaging with just one channel (edge case)."""
        tile = TileDataTestUtils.create_channel_specific_tile(
            self.tile1_name_base, "488", self.shape, 2.0
        )
        original_mean = tile.data.mean()

        # Divide by 1 (should not change the data)
        tile /= 1

        self.assertAlmostEqual(tile.data.mean(), original_mean, places=6)


class TestTileDataCore(unittest.TestCase):
    """Test core TileData functionality."""

    def test_transpose_operations(self):
        """Test that transposed tiles maintain compatibility."""
        tile1 = TileDataTestUtils.create_mock_tile_data("tile1", (5, 10, 15), 1.0, seed=42)
        tile2 = TileDataTestUtils.create_mock_tile_data("tile2", (5, 10, 15), 1.0, seed=24)

        # Transpose both tiles the same way
        tile1.transpose((2, 1, 0))  # ZYX -> XYZ
        tile2.transpose((2, 1, 0))  # ZYX -> XYZ

        # They should still be compatible
        TileDataTestUtils.assert_tiles_compatible(self, tile1, tile2)

        result = tile1 + tile2
        self.assertEqual(result.dim_order, "XYZ")
        self.assertEqual(result.shape, (15, 10, 5))  # X, Y, Z

    def test_mixed_transpose_incompatibility(self):
        """Test that tiles with different transposes are incompatible."""
        tile1 = TileDataTestUtils.create_mock_tile_data("tile1", (5, 10, 15), 1.0)
        tile2 = TileDataTestUtils.create_mock_tile_data("tile2", (5, 10, 15), 1.0)

        # Transpose only one tile
        tile1.transpose((2, 1, 0))  # ZYX -> XYZ
        # tile2 remains in ZYX order

        # They should be incompatible (shape mismatch due to transpose)
        with self.assertRaises(ValueError) as context:
            _ = tile1 + tile2
        self.assertIn("Shape mismatch", str(context.exception))

    def test_create_copy_functionality(self):
        """Test the _create_copy method works correctly."""
        original = TileDataTestUtils.create_mock_tile_data("original", (5, 10, 15), 2.0)

        # Test that _create_copy preserves all attributes
        copy = original._create_copy()

        self.assertEqual(copy.tile_name, original.tile_name)
        self.assertEqual(copy.bucket_name, original.bucket_name)
        self.assertEqual(copy.dataset_path, original.dataset_path)
        self.assertEqual(copy.pyramid_level, original.pyramid_level)
        self.assertEqual(copy.shape, original.shape)
        self.assertEqual(copy.z_dim, original.z_dim)
        self.assertEqual(copy.y_dim, original.y_dim)
        self.assertEqual(copy.x_dim, original.x_dim)
        self.assertEqual(copy.dim_order, original.dim_order)
        self.assertTrue(copy._loaded)

    def test_zero_division_handling(self):
        """Test handling of division by zero."""
        tile = TileDataTestUtils.create_mock_tile_data("tile", (3, 5, 7), 1.0)

        # Division by zero should work (numpy handles it)
        result = tile / 0
        self.assertTrue(np.all(np.isinf(result.data) | np.isnan(result.data)))

    def test_metadata_consistency(self):
        """Test that metadata remains consistent after operations."""
        tile1 = TileDataTestUtils.create_mock_tile_data(
            "tile1", (4, 8, 12), 1.5, channel="488", pyramid_level=2
        )
        tile2 = TileDataTestUtils.create_mock_tile_data(
            "tile2", (4, 8, 12), 0.8, channel="561", pyramid_level=2
        )

        result = tile1 * tile2

        # Check that some metadata is preserved from tile1
        self.assertEqual(result.bucket_name, tile1.bucket_name)
        self.assertEqual(result.dataset_path, tile1.dataset_path)
        self.assertEqual(result.pyramid_level, tile1.pyramid_level)

        # Shape and dimension metadata should be consistent
        self.assertEqual(result.shape, tile1.shape)
        self.assertEqual(result.z_dim, tile1.z_dim)
        self.assertEqual(result.y_dim, tile1.y_dim)
        self.assertEqual(result.x_dim, tile1.x_dim)


if __name__ == "__main__":
    # Run tests with unittest
    unittest.main(verbosity=2)
