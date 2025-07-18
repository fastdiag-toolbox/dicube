"""Unit tests for DicomCubeImage exception handling."""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from spacetransformer import Space

from dicube.core.image import DicomCubeImage
from dicube.core.pixel_header import PixelDataHeader
from dicube.dicom import DicomMeta
from dicube.exceptions import (
    DataConsistencyError,
    MetaDataError
)


class TestDicomCubeImageConstructor:
    """Test DicomCubeImage constructor exception handling."""
    
    def test_none_raw_image(self):
        """Test constructor with None raw_image."""
        pixel_header = Mock(spec=PixelDataHeader)
        
        with pytest.raises(DataConsistencyError) as exc_info:
            DicomCubeImage(None, pixel_header)
        
        error = exc_info.value
        assert "raw_image" in str(error)
        assert "cannot be None" in str(error)
        assert "DicomCubeImage constructor" in str(error)
    
    def test_none_pixel_header(self):
        """Test constructor with None pixel_header."""
        raw_image = np.array([[1, 2], [3, 4]])
        
        with pytest.raises(DataConsistencyError) as exc_info:
            DicomCubeImage(raw_image, None)
        
        error = exc_info.value
        assert "pixel_header" in str(error)
        assert "cannot be None" in str(error)
    
    def test_invalid_raw_image_dimensions(self):
        """Test constructor with invalid raw_image dimensions."""
        raw_image = np.array([1])  # 1D array, should be at least 2D
        pixel_header = Mock(spec=PixelDataHeader)
        
        with pytest.raises(DataConsistencyError) as exc_info:
            DicomCubeImage(raw_image, pixel_header)
        
        error = exc_info.value
        assert "too few dimensions" in str(error)
        assert "raw_image" in str(error)
    
    def test_invalid_pixel_header_type(self):
        """Test constructor with invalid pixel_header type."""
        raw_image = np.array([[1, 2], [3, 4]])
        
        with pytest.raises(DataConsistencyError) as exc_info:
            DicomCubeImage(raw_image, "not_a_pixel_header")
        
        error = exc_info.value
        assert "incorrect type" in str(error)
        assert "pixel_header" in str(error)
    
    def test_invalid_dicom_meta_type(self):
        """Test constructor with invalid dicom_meta type."""
        raw_image = np.array([[1, 2], [3, 4]])
        pixel_header = Mock(spec=PixelDataHeader)
        
        with pytest.raises(MetaDataError) as exc_info:
            DicomCubeImage(raw_image, pixel_header, dicom_meta="not_dicom_meta")
        
        error = exc_info.value
        assert "incorrect type" in str(error)
        assert "dicom_meta" in str(error)
    
    def test_invalid_space_type(self):
        """Test constructor with invalid space type."""
        raw_image = np.array([[1, 2], [3, 4]])
        pixel_header = Mock(spec=PixelDataHeader)
        
        with pytest.raises(DataConsistencyError) as exc_info:
            DicomCubeImage(raw_image, pixel_header, space="not_space")
        
        error = exc_info.value
        assert "incorrect type" in str(error)
        assert "space" in str(error)
    
    def test_shape_validation_mismatch(self):
        """Test constructor with shape mismatch between image and space."""
        raw_image = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # 2x2x2
        pixel_header = Mock(spec=PixelDataHeader)
        
        # Mock space with different shape
        mock_space = Mock(spec=Space)
        mock_space.shape = [3, 3, 3]  # Different from image shape
        
        with pytest.raises(DataConsistencyError) as exc_info:
            DicomCubeImage(raw_image, pixel_header, space=mock_space)
        
        error = exc_info.value
        assert "Space shape mismatch" in str(error)
        assert "space_shape=(3, 3, 3)" in str(error)
        assert "image_shape=(2, 2, 2)" in str(error)


class TestDicomCubeImageInitMeta:
    """Test DicomCubeImage init_meta method exception handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.raw_image = np.array([[1, 2], [3, 4]])
        self.pixel_header = Mock(spec=PixelDataHeader)
        self.image = DicomCubeImage(self.raw_image, self.pixel_header)
    
    def test_empty_modality(self):
        """Test init_meta with empty modality."""
        with pytest.raises(MetaDataError) as exc_info:
            self.image.init_meta(modality="")
        
        error = exc_info.value
        assert "modality" in str(error)
        assert "cannot be empty" in str(error)
        assert "init_meta operation" in str(error)
    
    def test_empty_patient_name(self):
        """Test init_meta with empty patient_name."""
        with pytest.raises(MetaDataError) as exc_info:
            self.image.init_meta(patient_name="")
        
        error = exc_info.value
        assert "patient_name" in str(error)
        assert "cannot be empty" in str(error)
    
    def test_empty_patient_id(self):
        """Test init_meta with empty patient_id."""
        with pytest.raises(MetaDataError) as exc_info:
            self.image.init_meta(patient_id="")
        
        error = exc_info.value
        assert "patient_id" in str(error)
        assert "cannot be empty" in str(error)
    
    def test_none_modality(self):
        """Test init_meta with None modality."""
        with pytest.raises(MetaDataError) as exc_info:
            self.image.init_meta(modality=None)
        
        error = exc_info.value
        assert "modality" in str(error)
        assert "cannot be None" in str(error)
    
    @patch('dicube.core.image.DicomMeta')
    def test_dicom_meta_creation_failure(self, mock_dicom_meta):
        """Test init_meta when DicomMeta creation fails."""
        mock_dicom_meta.side_effect = Exception("DicomMeta creation failed")
        
        with pytest.raises(MetaDataError) as exc_info:
            self.image.init_meta()
        
        error = exc_info.value
        assert "Failed to initialize DicomMeta" in str(error)
        assert "DicomMeta creation failed" in str(error)
    
    @patch('dicube.core.image.DicomMeta')
    def test_dicom_meta_returns_none(self, mock_dicom_meta):
        """Test init_meta when DicomMeta constructor returns None."""
        mock_dicom_meta.return_value = None
        
        with pytest.raises(MetaDataError) as exc_info:
            self.image.init_meta()
        
        error = exc_info.value
        assert "Failed to initialize DicomMeta" in str(error)
        assert "init_meta operation" in str(error)


class TestDicomCubeImageValidation:
    """Test DicomCubeImage validation methods."""
    
    def test_valid_construction(self):
        """Test successful construction with valid parameters."""
        raw_image = np.array([[1, 2], [3, 4]])
        pixel_header = Mock(spec=PixelDataHeader)
        
        # Should not raise any exceptions
        image = DicomCubeImage(raw_image, pixel_header)
        assert image.raw_image is raw_image
        assert image.pixel_header is pixel_header
    
    def test_valid_construction_with_optional_params(self):
        """Test successful construction with valid optional parameters."""
        raw_image = np.array([[1, 2], [3, 4]])
        pixel_header = Mock(spec=PixelDataHeader)
        dicom_meta = Mock(spec=DicomMeta)
        space = Mock(spec=Space)
        space.shape = [2, 2]  # Match image shape
        
        # Should not raise any exceptions
        image = DicomCubeImage(
            raw_image, 
            pixel_header, 
            dicom_meta=dicom_meta, 
            space=space
        )
        assert image.dicom_meta is dicom_meta
        assert image.space is space
    
    def test_properties_access(self):
        """Test that properties work correctly after construction."""
        raw_image = np.array([[1, 2], [3, 4]])
        pixel_header = Mock(spec=PixelDataHeader)
        image = DicomCubeImage(raw_image, pixel_header)
        
        assert image.shape == (2, 2)
        assert image.dtype == raw_image.dtype