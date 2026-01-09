"""Tests for memory utility functions."""

import numpy as np

from aurel.utils.memory import format_size, get_size


class TestGetSize:
    """Test get_size function with various object types."""

    def test_numpy_array(self):
        """Test memory size calculation for numpy arrays."""
        arr = np.ones((100, 100), dtype=np.float64)
        expected_size = 100 * 100 * 8  # 100x100 array, 8 bytes per float64
        assert get_size(arr) == expected_size

    def test_numpy_array_different_dtypes(self):
        """Test memory size for arrays with different dtypes."""
        arr_float32 = np.ones((10, 10), dtype=np.float32)
        assert get_size(arr_float32) == 10 * 10 * 4  # 4 bytes per float32

        arr_int64 = np.ones((10, 10), dtype=np.int64)
        assert get_size(arr_int64) == 10 * 10 * 8  # 8 bytes per int64

        arr_int8 = np.ones((10, 10), dtype=np.int8)
        assert get_size(arr_int8) == 10 * 10 * 1  # 1 byte per int8

    def test_list_of_arrays(self):
        """Test memory size for lists containing numpy arrays."""
        arr1 = np.ones((10, 10), dtype=np.float64)
        arr2 = np.ones((20, 20), dtype=np.float64)
        arr_list = [arr1, arr2]

        expected_size = arr1.nbytes + arr2.nbytes
        assert get_size(arr_list) == expected_size

    def test_tuple_of_arrays(self):
        """Test memory size for tuples containing numpy arrays."""
        arr1 = np.ones((5, 5), dtype=np.float64)
        arr2 = np.ones((10, 10), dtype=np.float64)
        arr_tuple = (arr1, arr2)

        expected_size = arr1.nbytes + arr2.nbytes
        assert get_size(arr_tuple) == expected_size

    def test_nested_list_of_arrays(self):
        """Test memory size for nested lists of arrays."""
        arr1 = np.ones((10, 10), dtype=np.float64)
        arr2 = np.ones((5, 5), dtype=np.float64)
        nested_list = [[arr1], [arr2]]

        expected_size = arr1.nbytes + arr2.nbytes
        assert get_size(nested_list) == expected_size

    def test_dict_of_arrays(self):
        """Test memory size for dictionaries containing arrays."""
        data = {
            'a': np.ones((10, 10), dtype=np.float64),
            'b': np.ones((5, 5), dtype=np.float64),
        }

        expected_size = data['a'].nbytes + data['b'].nbytes
        # Dict keys also counted, but they're small
        assert get_size(data) >= expected_size

    def test_mixed_dict(self):
        """Test memory size for dict with arrays and other objects."""
        data = {
            'array': np.ones((10, 10), dtype=np.float64),
            'number': 42,
            'string': 'test',
        }

        # Should at least include the array size
        assert get_size(data) >= data['array'].nbytes

    def test_regular_python_objects(self):
        """Test memory size for regular Python objects."""
        import sys

        # For lists/tuples, we recursively sum contents
        # (this is deeper than sys.getsizeof which is shallow)
        simple_list = [1, 2, 3, 4, 5]
        assert get_size(simple_list) > sys.getsizeof(simple_list)
        # Should include the list container plus all integers
        assert get_size(simple_list) > 0

        # For non-collection objects, should use sys.getsizeof
        simple_string = "test"
        assert get_size(simple_string) == sys.getsizeof(simple_string)

        simple_number = 42
        assert get_size(simple_number) == sys.getsizeof(simple_number)

        simple_dict = {'a': 1, 'b': 2}
        # For dicts, we sum keys and values recursively
        assert get_size(simple_dict) > 0

    def test_empty_array(self):
        """Test memory size for empty arrays."""
        empty_arr = np.array([])
        assert get_size(empty_arr) == empty_arr.nbytes

    def test_3d_array(self):
        """Test memory size for 3D arrays (common in aurel)."""
        arr_3d = np.ones((10, 20, 30), dtype=np.float64)
        expected_size = 10 * 20 * 30 * 8
        assert get_size(arr_3d) == expected_size


class TestFormatSize:
    """Test format_size function."""

    def test_bytes(self):
        """Test formatting for bytes."""
        assert format_size(512) == "512.00 B"
        assert format_size(1023) == "1023.00 B"

    def test_kilobytes(self):
        """Test formatting for kilobytes."""
        assert format_size(1024) == "1.00 KB"
        assert format_size(1536) == "1.50 KB"
        assert format_size(2048) == "2.00 KB"

    def test_megabytes(self):
        """Test formatting for megabytes."""
        assert format_size(1024 * 1024) == "1.00 MB"
        assert format_size(1.5 * 1024 * 1024) == "1.50 MB"
        assert format_size(100 * 1024 * 1024) == "100.00 MB"

    def test_gigabytes(self):
        """Test formatting for gigabytes."""
        assert format_size(1024 * 1024 * 1024) == "1.00 GB"
        assert format_size(2.5 * 1024 * 1024 * 1024) == "2.50 GB"

    def test_terabytes(self):
        """Test formatting for terabytes."""
        assert format_size(1024 * 1024 * 1024 * 1024) == "1.00 TB"
        assert format_size(5.5 * 1024 * 1024 * 1024 * 1024) == "5.50 TB"

    def test_petabytes(self):
        """Test formatting for petabytes."""
        assert format_size(1024 * 1024 * 1024 * 1024 * 1024) == "1.00 PB"

    def test_zero(self):
        """Test formatting for zero bytes."""
        assert format_size(0) == "0.00 B"

    def test_real_array_sizes(self):
        """Test formatting with realistic array sizes."""
        # 100x100 float64 array
        arr_size = 100 * 100 * 8
        result = format_size(arr_size)
        assert "KB" in result or "MB" in result

        # 1000x1000 float64 array
        large_arr_size = 1000 * 1000 * 8
        result = format_size(large_arr_size)
        assert "MB" in result
        assert float(result.split()[0]) > 0
