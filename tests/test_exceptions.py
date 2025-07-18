"""Unit tests for DiCube exception handling system."""

import pytest
import numpy as np
from spacetransformer import Space

from dicube.exceptions import (
    DicomCubeError,
    InvalidCubeFileError,
    CodecError,
    MetaDataError,
    DataConsistencyError
)


class TestDicomCubeError:
    """Test the base DicomCubeError class."""
    
    def test_basic_error_creation(self):
        """Test basic error creation with message only."""
        error = DicomCubeError("Test error message")
        assert str(error) == "Test error message"
        assert error.context is None
        assert error.suggestion is None
        assert error.details == {}
    
    def test_error_with_context(self):
        """Test error creation with context."""
        error = DicomCubeError("Test error", context="test operation")
        assert "test operation: Test error" in str(error)
        assert error.context == "test operation"
    
    def test_error_with_suggestion(self):
        """Test error creation with suggestion."""
        error = DicomCubeError("Test error", suggestion="Try this fix")
        error_str = str(error)
        assert "Test error" in error_str
        assert "Suggestion: Try this fix" in error_str
        assert error.suggestion == "Try this fix"
    
    def test_error_with_details(self):
        """Test error creation with details."""
        details = {"param1": "value1", "param2": 42}
        error = DicomCubeError("Test error", details=details)
        error_str = str(error)
        assert "Test error" in error_str
        assert "Details:" in error_str
        assert "param1='value1'" in error_str
        assert "param2=42" in error_str
        assert error.details == details
    
    def test_error_with_all_parameters(self):
        """Test error creation with all parameters."""
        details = {"file_path": "/test/path"}
        error = DicomCubeError(
            "Test error",
            context="test operation",
            suggestion="Check the file path",
            details=details
        )
        error_str = str(error)
        assert "test operation: Test error" in error_str
        assert "Details: file_path='/test/path'" in error_str
        assert "Suggestion: Check the file path" in error_str
    
    def test_error_inheritance(self):
        """Test that DicomCubeError inherits from Exception."""
        error = DicomCubeError("Test error")
        assert isinstance(error, Exception)
        assert isinstance(error, DicomCubeError)


class TestInvalidCubeFileError:
    """Test the InvalidCubeFileError class."""
    
    def test_default_suggestion(self):
        """Test that InvalidCubeFileError has a default suggestion."""
        error = InvalidCubeFileError("File format error")
        error_str = str(error)
        assert "File format error" in error_str
        assert "Verify the file is a valid DicomCube file" in error_str
    
    def test_custom_suggestion_override(self):
        """Test that custom suggestion overrides default."""
        error = InvalidCubeFileError(
            "File format error",
            suggestion="Custom suggestion"
        )
        error_str = str(error)
        assert "Custom suggestion" in error_str
        assert "Verify the file is a valid DicomCube file" not in error_str
    
    def test_inheritance(self):
        """Test inheritance from DicomCubeError."""
        error = InvalidCubeFileError("Test error")
        assert isinstance(error, DicomCubeError)
        assert isinstance(error, InvalidCubeFileError)


class TestCodecError:
    """Test the CodecError class."""
    
    def test_default_suggestion(self):
        """Test that CodecError has a default suggestion."""
        error = CodecError("Encoding failed")
        error_str = str(error)
        assert "Encoding failed" in error_str
        assert "Check image data format and codec compatibility" in error_str
    
    def test_custom_suggestion_override(self):
        """Test that custom suggestion overrides default."""
        error = CodecError(
            "Encoding failed",
            suggestion="Check codec installation"
        )
        error_str = str(error)
        assert "Check codec installation" in error_str
        assert "Check image data format and codec compatibility" not in error_str
    
    def test_inheritance(self):
        """Test inheritance from DicomCubeError."""
        error = CodecError("Test error")
        assert isinstance(error, DicomCubeError)
        assert isinstance(error, CodecError)


class TestMetaDataError:
    """Test the MetaDataError class."""
    
    def test_default_suggestion(self):
        """Test that MetaDataError has a default suggestion."""
        error = MetaDataError("Metadata missing")
        error_str = str(error)
        assert "Metadata missing" in error_str
        assert "Verify metadata completeness and consistency" in error_str
    
    def test_custom_suggestion_override(self):
        """Test that custom suggestion overrides default."""
        error = MetaDataError(
            "Metadata missing",
            suggestion="Check DICOM tags"
        )
        error_str = str(error)
        assert "Check DICOM tags" in error_str
        assert "Verify metadata completeness and consistency" not in error_str
    
    def test_inheritance(self):
        """Test inheritance from DicomCubeError."""
        error = MetaDataError("Test error")
        assert isinstance(error, DicomCubeError)
        assert isinstance(error, MetaDataError)


class TestDataConsistencyError:
    """Test the DataConsistencyError class."""
    
    def test_default_suggestion(self):
        """Test that DataConsistencyError has a default suggestion."""
        error = DataConsistencyError("Shape mismatch")
        error_str = str(error)
        assert "Shape mismatch" in error_str
        assert "Check data array shapes, types, and dimensional consistency" in error_str
    
    def test_custom_suggestion_override(self):
        """Test that custom suggestion overrides default."""
        error = DataConsistencyError(
            "Shape mismatch",
            suggestion="Reshape your arrays"
        )
        error_str = str(error)
        assert "Reshape your arrays" in error_str
        assert "Check data array shapes, types, and dimensional consistency" not in error_str
    
    def test_inheritance(self):
        """Test inheritance from DicomCubeError."""
        error = DataConsistencyError("Test error")
        assert isinstance(error, DicomCubeError)
        assert isinstance(error, DataConsistencyError)


class TestExceptionHierarchy:
    """Test the exception hierarchy and catching behavior."""
    
    def test_catch_all_dicube_errors(self):
        """Test that all DiCube errors can be caught with base class."""
        errors = [
            InvalidCubeFileError("File error"),
            CodecError("Codec error"),
            MetaDataError("Metadata error"),
            DataConsistencyError("Data error")
        ]
        
        for error in errors:
            try:
                raise error
            except DicomCubeError as e:
                assert isinstance(e, DicomCubeError)
                # Should also be instance of specific error type
                assert type(e) in [InvalidCubeFileError, CodecError, MetaDataError, DataConsistencyError]
    
    def test_catch_specific_errors(self):
        """Test that specific errors can be caught individually."""
        # Test InvalidCubeFileError
        try:
            raise InvalidCubeFileError("File error")
        except InvalidCubeFileError as e:
            assert isinstance(e, InvalidCubeFileError)
            assert isinstance(e, DicomCubeError)
        
        # Test CodecError
        try:
            raise CodecError("Codec error")
        except CodecError as e:
            assert isinstance(e, CodecError)
            assert isinstance(e, DicomCubeError)
        
        # Test MetaDataError
        try:
            raise MetaDataError("Metadata error")
        except MetaDataError as e:
            assert isinstance(e, MetaDataError)
            assert isinstance(e, DicomCubeError)
        
        # Test DataConsistencyError
        try:
            raise DataConsistencyError("Data error")
        except DataConsistencyError as e:
            assert isinstance(e, DataConsistencyError)
            assert isinstance(e, DicomCubeError)