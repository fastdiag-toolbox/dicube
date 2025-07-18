"""Unit tests for DiCube validation utilities."""

import pytest
import numpy as np
import tempfile
import os
from spacetransformer import Space

from dicube.validation import (
    validate_not_none,
    validate_file_exists,
    validate_folder_exists,
    validate_array_shape,
    validate_parameter_type,
    validate_shape_consistency,
    validate_string_not_empty,
    validate_numeric_range
)
from dicube.exceptions import (
    DicomCubeError,
    InvalidCubeFileError,
    DataConsistencyError,
    MetaDataError
)
from dicube.core.pixel_header import PixelDataHeader


class TestValidateNotNone:
    """Test the validate_not_none function."""
    
    def test_valid_value(self):
        """Test validation with valid non-None value."""
        result = validate_not_none("test_value", "test_param")
        assert result == "test_value"
    
    def test_none_value_default_exception(self):
        """Test validation with None value using default exception."""
        with pytest.raises(DicomCubeError) as exc_info:
            validate_not_none(None, "test_param", "test_context")
        
        error = exc_info.value
        assert "test_param" in str(error)
        assert "cannot be None" in str(error)
        assert "test_context" in str(error)
    
    def test_none_value_custom_exception(self):
        """Test validation with None value using custom exception class."""
        with pytest.raises(DataConsistencyError) as exc_info:
            validate_not_none(None, "test_param", "test_context", DataConsistencyError)
        
        error = exc_info.value
        assert isinstance(error, DataConsistencyError)
        assert "test_param" in str(error)


class TestValidateFileExists:
    """Test the validate_file_exists function."""
    
    def test_existing_file(self):
        """Test validation with existing file."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            result = validate_file_exists(tmp_path, "test_context")
            assert result == tmp_path
        finally:
            os.unlink(tmp_path)
    
    def test_nonexistent_file(self):
        """Test validation with non-existent file."""
        with pytest.raises(InvalidCubeFileError) as exc_info:
            validate_file_exists("/nonexistent/file.txt", "test_context")
        
        error = exc_info.value
        assert "File not found" in str(error)
        assert "test_context" in str(error)
        assert "/nonexistent/file.txt" in str(error)
    
    def test_directory_instead_of_file(self):
        """Test validation when path points to directory instead of file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with pytest.raises(InvalidCubeFileError) as exc_info:
                validate_file_exists(tmp_dir, "test_context")
            
            error = exc_info.value
            assert "not a file" in str(error)
    
    def test_none_file_path(self):
        """Test validation with None file path."""
        with pytest.raises(InvalidCubeFileError) as exc_info:
            validate_file_exists(None, "test_context")
        
        error = exc_info.value
        assert "file_path" in str(error)
        assert "cannot be None" in str(error)


class TestValidateFolderExists:
    """Test the validate_folder_exists function."""
    
    def test_existing_folder(self):
        """Test validation with existing folder."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = validate_folder_exists(tmp_dir, "test_context")
            assert result == tmp_dir
    
    def test_nonexistent_folder(self):
        """Test validation with non-existent folder."""
        with pytest.raises(InvalidCubeFileError) as exc_info:
            validate_folder_exists("/nonexistent/folder", "test_context")
        
        error = exc_info.value
        assert "Folder not found" in str(error)
        assert "test_context" in str(error)
    
    def test_file_instead_of_folder(self):
        """Test validation when path points to file instead of folder."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            with pytest.raises(InvalidCubeFileError) as exc_info:
                validate_folder_exists(tmp_path, "test_context")
            
            error = exc_info.value
            assert "not a directory" in str(error)
        finally:
            os.unlink(tmp_path)


class TestValidateArrayShape:
    """Test the validate_array_shape function."""
    
    def test_valid_array(self):
        """Test validation with valid numpy array."""
        arr = np.array([[1, 2], [3, 4]])
        result = validate_array_shape(arr, name="test_array")
        assert np.array_equal(result, arr)
    
    def test_expected_dims(self):
        """Test validation with expected dimensions."""
        arr = np.array([[1, 2], [3, 4]])  # 2D array
        result = validate_array_shape(arr, expected_dims=2, name="test_array")
        assert np.array_equal(result, arr)
    
    def test_wrong_expected_dims(self):
        """Test validation with wrong expected dimensions."""
        arr = np.array([[1, 2], [3, 4]])  # 2D array
        with pytest.raises(DataConsistencyError) as exc_info:
            validate_array_shape(arr, expected_dims=3, name="test_array")
        
        error = exc_info.value
        assert "incorrect dimensions" in str(error)
        assert "expected_dims=3" in str(error)
        assert "actual_dims=2" in str(error)
    
    def test_min_dims(self):
        """Test validation with minimum dimensions."""
        arr = np.array([[[1, 2], [3, 4]]])  # 3D array
        result = validate_array_shape(arr, min_dims=2, name="test_array")
        assert np.array_equal(result, arr)
    
    def test_insufficient_min_dims(self):
        """Test validation with insufficient minimum dimensions."""
        arr = np.array([1, 2, 3])  # 1D array
        with pytest.raises(DataConsistencyError) as exc_info:
            validate_array_shape(arr, min_dims=2, name="test_array")
        
        error = exc_info.value
        assert "too few dimensions" in str(error)
    
    def test_max_dims(self):
        """Test validation with maximum dimensions."""
        arr = np.array([[1, 2], [3, 4]])  # 2D array
        result = validate_array_shape(arr, max_dims=3, name="test_array")
        assert np.array_equal(result, arr)
    
    def test_excessive_max_dims(self):
        """Test validation with excessive maximum dimensions."""
        arr = np.array([[[1, 2], [3, 4]]])  # 3D array
        with pytest.raises(DataConsistencyError) as exc_info:
            validate_array_shape(arr, max_dims=2, name="test_array")
        
        error = exc_info.value
        assert "too many dimensions" in str(error)
    
    def test_non_array_input(self):
        """Test validation with non-array input."""
        with pytest.raises(DataConsistencyError) as exc_info:
            validate_array_shape([1, 2, 3], name="test_array")
        
        error = exc_info.value
        assert "must be a numpy array" in str(error)
    
    def test_none_array(self):
        """Test validation with None array."""
        with pytest.raises(DataConsistencyError) as exc_info:
            validate_array_shape(None, name="test_array")
        
        error = exc_info.value
        assert "cannot be None" in str(error)


class TestValidateParameterType:
    """Test the validate_parameter_type function."""
    
    def test_correct_type(self):
        """Test validation with correct type."""
        result = validate_parameter_type("test_string", str, "test_param")
        assert result == "test_string"
    
    def test_incorrect_type(self):
        """Test validation with incorrect type."""
        with pytest.raises(DataConsistencyError) as exc_info:
            validate_parameter_type(123, str, "test_param", "test_context")
        
        error = exc_info.value
        assert "incorrect type" in str(error)
        assert "expected_type='str'" in str(error)
        assert "actual_type='int'" in str(error)
    
    def test_multiple_allowed_types(self):
        """Test validation with multiple allowed types."""
        # Test with int (should pass)
        result1 = validate_parameter_type(123, (int, float), "test_param")
        assert result1 == 123
        
        # Test with float (should pass)
        result2 = validate_parameter_type(123.5, (int, float), "test_param")
        assert result2 == 123.5
        
        # Test with string (should fail)
        with pytest.raises(DataConsistencyError) as exc_info:
            validate_parameter_type("test", (int, float), "test_param")
        
        error = exc_info.value
        assert "int or float" in str(error)
    
    def test_none_value(self):
        """Test validation with None value."""
        with pytest.raises(DataConsistencyError) as exc_info:
            validate_parameter_type(None, str, "test_param")
        
        error = exc_info.value
        assert "cannot be None" in str(error)


class TestValidateShapeConsistency:
    """Test the validate_shape_consistency function."""
    
    def test_consistent_shapes(self):
        """Test validation with consistent array shapes."""
        arr1 = np.array([[1, 2], [3, 4]])
        arr2 = np.array([[5, 6], [7, 8]])
        result1, result2 = validate_shape_consistency(arr1, arr2)
        assert np.array_equal(result1, arr1)
        assert np.array_equal(result2, arr2)
    
    def test_inconsistent_shapes(self):
        """Test validation with inconsistent array shapes."""
        arr1 = np.array([[1, 2], [3, 4]])  # 2x2
        arr2 = np.array([1, 2, 3])  # 1x3
        
        with pytest.raises(DataConsistencyError) as exc_info:
            validate_shape_consistency(arr1, arr2, "array1", "array2", "test_context")
        
        error = exc_info.value
        assert "inconsistent shapes" in str(error)
        assert "array1_shape=(2, 2)" in str(error)
        assert "array2_shape=(3,)" in str(error)


class TestValidateStringNotEmpty:
    """Test the validate_string_not_empty function."""
    
    def test_valid_string(self):
        """Test validation with valid non-empty string."""
        result = validate_string_not_empty("test_string", "test_param")
        assert result == "test_string"
    
    def test_empty_string(self):
        """Test validation with empty string."""
        with pytest.raises(DicomCubeError) as exc_info:
            validate_string_not_empty("", "test_param", "test_context")
        
        error = exc_info.value
        assert "cannot be empty" in str(error)
    
    def test_whitespace_only_string(self):
        """Test validation with whitespace-only string."""
        with pytest.raises(DicomCubeError) as exc_info:
            validate_string_not_empty("   ", "test_param", "test_context")
        
        error = exc_info.value
        assert "cannot be empty" in str(error)
    
    def test_none_string(self):
        """Test validation with None string."""
        with pytest.raises(DicomCubeError) as exc_info:
            validate_string_not_empty(None, "test_param")
        
        error = exc_info.value
        assert "cannot be None" in str(error)
    
    def test_non_string_type(self):
        """Test validation with non-string type."""
        with pytest.raises(DicomCubeError) as exc_info:
            validate_string_not_empty(123, "test_param")
        
        error = exc_info.value
        assert "incorrect type" in str(error)


class TestValidateNumericRange:
    """Test the validate_numeric_range function."""
    
    def test_valid_value_in_range(self):
        """Test validation with value in valid range."""
        result = validate_numeric_range(5, "test_param", min_value=1, max_value=10)
        assert result == 5
    
    def test_value_below_minimum(self):
        """Test validation with value below minimum."""
        with pytest.raises(DataConsistencyError) as exc_info:
            validate_numeric_range(0, "test_param", min_value=1, context="test_context")
        
        error = exc_info.value
        assert "below minimum value" in str(error)
        assert "min_value=1" in str(error)
    
    def test_value_above_maximum(self):
        """Test validation with value above maximum."""
        with pytest.raises(DataConsistencyError) as exc_info:
            validate_numeric_range(15, "test_param", max_value=10, context="test_context")
        
        error = exc_info.value
        assert "exceeds maximum value" in str(error)
        assert "max_value=10" in str(error)
    
    def test_float_values(self):
        """Test validation with float values."""
        result = validate_numeric_range(3.14, "test_param", min_value=0.0, max_value=5.0)
        assert result == 3.14
    
    def test_non_numeric_type(self):
        """Test validation with non-numeric type."""
        with pytest.raises(DataConsistencyError) as exc_info:
            validate_numeric_range("not_a_number", "test_param")
        
        error = exc_info.value
        assert "incorrect type" in str(error)
    
    def test_none_value(self):
        """Test validation with None value."""
        with pytest.raises(DataConsistencyError) as exc_info:
            validate_numeric_range(None, "test_param")
        
        error = exc_info.value
        assert "cannot be None" in str(error)