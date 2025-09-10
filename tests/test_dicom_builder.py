"""
Test cases for DicomBuilder

Tests the unified DICOM dataset creation functionality.
"""

import io
import tempfile
import pytest
import numpy as np
from unittest.mock import Mock, patch

from pydicom import Dataset
from pydicom.uid import ExplicitVRLittleEndian
from pydicom.encaps import encapsulate

from dicube.dicom.dicom_builder import DicomBuilder, create_file_meta, ensure_required_tags, set_pixel_data_attributes, save_dicom


@pytest.fixture
def sample_meta_dict():
    """Sample metadata dictionary for testing"""
    return {
        "StudyInstanceUID": "1.2.3.4.5",
        "SeriesInstanceUID": "1.2.3.4.6", 
        "SOPInstanceUID": "1.2.3.4.7",
        "PatientID": "TEST001",
        "PatientName": "Test^Patient",
        "Modality": "CT",
        "Rows": 512,
        "Columns": 512,
        "InstanceNumber": 1
    }

@pytest.fixture
def sample_pixel_header():
    """Sample pixel header for testing"""
    mock_header = Mock()
    mock_header.RescaleSlope = 1.0
    mock_header.RescaleIntercept = 0.0
    mock_header.PixelDtype = "int16"
    return mock_header

@pytest.fixture
def sample_pixel_array():
    """Sample pixel array for testing"""
    return np.random.randint(0, 4096, size=(512, 512), dtype=np.int16)

@pytest.fixture
def sample_compressed_data():
    """Sample compressed pixel data"""
    return b'\x00\x00\x00\x0C\xFF\x4F\xFF\x51\x00\x29\x00\x00\x00\x00\x00\x00'


class TestDicomBuilder:
    """Test cases for DicomBuilder class"""
    
    def test_build_from_raw_pixel_array(self, sample_meta_dict, sample_pixel_header, sample_pixel_array):
        """Test building DICOM from raw pixel array"""
        ds = DicomBuilder.build(
            meta_dict=sample_meta_dict,
            pixel_header=sample_pixel_header,
            pixel_data=sample_pixel_array,
            is_compressed_data=False
        )
        
        # Check basic structure
        assert isinstance(ds, Dataset)
        assert hasattr(ds, 'file_meta')
        assert hasattr(ds, 'PixelData')
        
        # Check metadata preservation
        assert ds.StudyInstanceUID == "1.2.3.4.5"
        assert ds.PatientID == "TEST001"
        assert ds.Modality == "CT"
        
        # Check pixel data attributes
        assert ds.BitsAllocated == 16
        assert ds.BitsStored == 16
        assert ds.HighBit == 15
        assert ds.PixelRepresentation == 1  # signed
        assert ds.SamplesPerPixel == 1
        assert ds.PhotometricInterpretation == "MONOCHROME2"
        
        # Check rescale parameters
        assert ds.RescaleSlope == 1.0
        assert ds.RescaleIntercept == 0.0
        
        # Check pixel data
        expected_size = sample_pixel_array.nbytes
        assert len(ds.PixelData) == expected_size
    
    def test_build_from_compressed_data_keep_compressed(self, sample_meta_dict, sample_pixel_header, sample_compressed_data):
        """Test building DICOM from compressed data, keeping compression"""
        transfer_syntax_uid = "1.2.840.10008.1.2.4.201"  # HTJ2K
        
        ds = DicomBuilder.build(
            meta_dict=sample_meta_dict,
            pixel_header=sample_pixel_header,
            pixel_data=sample_compressed_data,
            transfer_syntax_uid=transfer_syntax_uid,
            is_compressed_data=True,
            force_uncompressed=False
        )
        
        # Check transfer syntax
        assert ds.file_meta.TransferSyntaxUID == transfer_syntax_uid
        
        # Check that pixel data is encapsulated
        assert hasattr(ds, 'PixelData')
        # Encapsulated data should start with basic offset table
        assert ds.PixelData.startswith(b'\xfe\xff\x00\xe0')
    
    def test_build_from_compressed_data_force_uncompressed(self, sample_meta_dict, sample_pixel_header, sample_compressed_data):
        """Test building DICOM from compressed data, forcing decompression"""
        transfer_syntax_uid = "1.2.840.10008.1.2.4.201"  # HTJ2K
        
        # Mock the decompression to avoid actual codec requirements
        with patch.object(DicomBuilder, '_decompress_pixel_data') as mock_decompress:
            mock_array = np.random.randint(0, 4096, size=(512, 512), dtype=np.int16)
            mock_decompress.return_value = mock_array
            
            ds = DicomBuilder.build(
                meta_dict=sample_meta_dict,
                pixel_header=sample_pixel_header,
                pixel_data=sample_compressed_data,
                transfer_syntax_uid=transfer_syntax_uid,
                is_compressed_data=True,
                force_uncompressed=True
            )
            
            # Should use uncompressed transfer syntax
            assert ds.file_meta.TransferSyntaxUID == ExplicitVRLittleEndian
            
            # Should have raw pixel data
            assert len(ds.PixelData) == mock_array.nbytes
    
    def test_build_from_compressed_data_decompression_fails(self, sample_meta_dict, sample_pixel_header, sample_compressed_data):
        """Test fallback when decompression fails"""
        transfer_syntax_uid = "1.2.840.10008.1.2.4.201"  # HTJ2K
        
        # Mock decompression failure
        with patch.object(DicomBuilder, '_decompress_pixel_data') as mock_decompress:
            mock_decompress.return_value = None  # Simulate failure
            
            ds = DicomBuilder.build(
                meta_dict=sample_meta_dict,
                pixel_header=sample_pixel_header,
                pixel_data=sample_compressed_data,
                transfer_syntax_uid=transfer_syntax_uid,
                is_compressed_data=True,
                force_uncompressed=True
            )
            
            # Should fallback to compressed format
            assert ds.file_meta.TransferSyntaxUID == transfer_syntax_uid
            assert ds.PixelData.startswith(b'\xfe\xff\x00\xe0')
    
    def test_build_without_pixel_data(self, sample_meta_dict, sample_pixel_header):
        """Test building DICOM without pixel data"""
        ds = DicomBuilder.build(
            meta_dict=sample_meta_dict,
            pixel_header=sample_pixel_header,
            pixel_data=None
        )
        
        # Should still have metadata and pixel attributes
        assert ds.StudyInstanceUID == "1.2.3.4.5"
        assert ds.BitsAllocated == 16
        assert not hasattr(ds, 'PixelData')
    
    def test_build_different_pixel_dtypes(self, sample_meta_dict):
        """Test building with different pixel data types"""
        test_cases = [
            ("uint8", np.uint8, 8, 8, 7, 0),
            ("int8", np.int8, 8, 8, 7, 1),
            ("uint16", np.uint16, 16, 16, 15, 0),
            ("int16", np.int16, 16, 16, 15, 1),
            ("uint32", np.uint32, 32, 32, 31, 0),
            ("int32", np.int32, 32, 32, 31, 1),
        ]
        
        for dtype_str, numpy_dtype, bits_alloc, bits_stored, high_bit, pixel_rep in test_cases:
            # Create mock pixel header
            mock_header = Mock()
            mock_header.RescaleSlope = 1.0
            mock_header.RescaleIntercept = 0.0
            mock_header.PixelDtype = dtype_str
            
            # Create test array
            test_array = np.random.randint(0, 100, size=(64, 64)).astype(numpy_dtype)
            
            ds = DicomBuilder.build(
                meta_dict=sample_meta_dict,
                pixel_header=mock_header,
                pixel_data=test_array,
                is_compressed_data=False
            )
            
            assert ds.BitsAllocated == bits_alloc
            assert ds.BitsStored == bits_stored
            assert ds.HighBit == high_bit
            assert ds.PixelRepresentation == pixel_rep
    
    def test_to_bytes(self, sample_meta_dict, sample_pixel_header, sample_pixel_array):
        """Test serialization to bytes"""
        ds = DicomBuilder.build(
            meta_dict=sample_meta_dict,
            pixel_header=sample_pixel_header,
            pixel_data=sample_pixel_array,
            is_compressed_data=False
        )
        
        dicom_bytes = DicomBuilder.to_bytes(ds)
        
        # Check that we got bytes
        assert isinstance(dicom_bytes, bytes)
        assert len(dicom_bytes) > 0
        
        # DICOM files should start with preamble + DICM
        assert dicom_bytes[128:132] == b'DICM'


class TestHelperFunctions:
    """Test cases for helper functions"""
    
    def test_create_file_meta(self):
        """Test file meta creation"""
        # Test with CT modality
        ds = Dataset()
        ds.Modality = "CT"
        ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        ds.SOPInstanceUID = "1.2.3.4.5"
        
        file_meta = create_file_meta(ds)
        
        assert file_meta.MediaStorageSOPClassUID == "1.2.840.10008.5.1.4.1.1.2"
        assert file_meta.MediaStorageSOPInstanceUID == "1.2.3.4.5"
        assert file_meta.TransferSyntaxUID == ExplicitVRLittleEndian
        assert hasattr(file_meta, 'ImplementationClassUID')
        
        # Test with unknown modality (should default to CT)
        ds2 = Dataset()
        ds2.Modality = "XX"  # Unknown modality
        
        file_meta2 = create_file_meta(ds2)
        assert file_meta2.MediaStorageSOPClassUID == "1.2.840.10008.5.1.4.1.1.2"  # CT default
    
    def test_ensure_required_tags(self):
        """Test required tags are ensured"""
        from pydicom.dataset import FileMetaDataset
        
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        ds.file_meta.MediaStorageSOPInstanceUID = "1.2.3.4.5"
        
        ensure_required_tags(ds)
        
        assert ds.SOPClassUID == "1.2.840.10008.5.1.4.1.1.2"
        assert ds.SOPInstanceUID == "1.2.3.4.5"
    
    def test_set_pixel_data_attributes(self):
        """Test pixel data attributes setting"""
        ds = Dataset()
        
        # Mock pixel header
        mock_header = Mock()
        mock_header.RescaleSlope = 2.0
        mock_header.RescaleIntercept = -1000.0
        mock_header.PixelDtype = "int16"
        
        set_pixel_data_attributes(ds, mock_header)
        
        assert ds.RescaleSlope == 2.0
        assert ds.RescaleIntercept == -1000.0
        assert ds.BitsAllocated == 16
        assert ds.BitsStored == 16
        assert ds.HighBit == 15
        assert ds.PixelRepresentation == 1
        assert ds.SamplesPerPixel == 1
        assert ds.PhotometricInterpretation == "MONOCHROME2"
    
    def test_set_pixel_data_attributes_none_header(self):
        """Test pixel data attributes with None header"""
        ds = Dataset()
        
        set_pixel_data_attributes(ds, None)
        
        # Should not crash, but no attributes should be set
        assert not hasattr(ds, 'RescaleSlope')
    
    def test_save_dicom(self, sample_meta_dict, sample_pixel_header, sample_pixel_array):
        """Test DICOM file saving"""
        ds = DicomBuilder.build(
            meta_dict=sample_meta_dict,
            pixel_header=sample_pixel_header,
            pixel_data=sample_pixel_array,
            is_compressed_data=False
        )
        
        with tempfile.NamedTemporaryFile(suffix='.dcm', delete=False) as tmp_file:
            save_dicom(ds, tmp_file.name)
            
            # Check file was created
            import os
            assert os.path.exists(tmp_file.name)
            assert os.path.getsize(tmp_file.name) > 0
            
            # Clean up
            os.unlink(tmp_file.name)
    
    def test_save_dicom_to_buffer(self, sample_meta_dict, sample_pixel_header, sample_pixel_array):
        """Test DICOM saving to buffer"""
        ds = DicomBuilder.build(
            meta_dict=sample_meta_dict,
            pixel_header=sample_pixel_header,
            pixel_data=sample_pixel_array,
            is_compressed_data=False
        )
        
        buffer = io.BytesIO()
        save_dicom(ds, buffer)
        
        buffer.seek(0)
        data = buffer.read()
        
        assert len(data) > 0
        # Check DICOM header
        assert data[128:132] == b'DICM'


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_build_with_invalid_transfer_syntax(self, sample_meta_dict, sample_pixel_header, sample_pixel_array):
        """Test building with invalid transfer syntax"""
        import warnings
        
        # Expect the invalid UID warning from pydicom
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Invalid value for VR UI", UserWarning)
            
            ds = DicomBuilder.build(
                meta_dict=sample_meta_dict,
                pixel_header=sample_pixel_header,
                pixel_data=sample_pixel_array,
                transfer_syntax_uid="invalid.uid",
                is_compressed_data=False
            )
        
        # Should still work, just use the provided UID
        assert ds.file_meta.TransferSyntaxUID == "invalid.uid"
    
    def test_build_with_bytes_pixel_data(self, sample_meta_dict, sample_pixel_header):
        """Test building with raw bytes as pixel data"""
        raw_bytes = b'\x00\x01\x02\x03' * 1000
        
        ds = DicomBuilder.build(
            meta_dict=sample_meta_dict,
            pixel_header=sample_pixel_header,
            pixel_data=raw_bytes,
            is_compressed_data=False
        )
        
        assert ds.PixelData == raw_bytes
    
    def test_build_with_type_conversion(self, sample_meta_dict, sample_pixel_header):
        """Test automatic type conversion of pixel arrays"""
        # Create array with wrong dtype
        wrong_dtype_array = np.random.randint(0, 100, size=(64, 64), dtype=np.uint8)
        
        # Header expects int16
        ds = DicomBuilder.build(
            meta_dict=sample_meta_dict,
            pixel_header=sample_pixel_header,
            pixel_data=wrong_dtype_array,
            is_compressed_data=False
        )
        
        # Should convert to int16 and store
        expected_size = 64 * 64 * 2  # int16 = 2 bytes per pixel
        assert len(ds.PixelData) == expected_size
