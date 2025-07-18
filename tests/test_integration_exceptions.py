"""Integration tests for DiCube exception handling across modules."""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

from dicube.core.io import DicomCubeImageIO

from dicube.core.image import DicomCubeImage
from dicube.core.pixel_header import PixelDataHeader
from dicube.storage.dcb_file import DcbSFile, DcbFile
from dicube.exceptions import (
    DicomCubeError,
    InvalidCubeFileError,
    CodecError,
    MetaDataError,
    DataConsistencyError
)


class TestFileIOIntegration:
    """Test file I/O integration with exception handling."""
    
    def test_load_nonexistent_file(self):
        """Test loading a non-existent file."""
        with pytest.raises(InvalidCubeFileError) as exc_info:
            DicomCubeImageIO.load("/nonexistent/file.dcb")
        
        error = exc_info.value
        assert "File not found" in str(error)
        assert "load operation" in str(error)
        assert "/nonexistent/file.dcb" in str(error)
    
    def test_load_invalid_file_format(self):
        """Test loading a file with invalid format."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dcb") as tmp_file:
            # Write invalid header data
            tmp_file.write(b"INVALID_MAGIC_NUMBER" + b"\x00" * 100)
            tmp_path = tmp_file.name
        
        try:
            with pytest.raises(InvalidCubeFileError) as exc_info:
                DicomCubeImageIO.load(tmp_path)
            
            error = exc_info.value
            assert "Failed to load file" in str(error) or "Unsupported file format" in str(error)
            assert "load operation" in str(error)
        finally:
            os.unlink(tmp_path)
    
    def test_save_with_invalid_file_type(self):
        """Test saving with invalid file type."""
        # Create a mock image
        raw_image = np.array([[1, 2], [3, 4]])
        pixel_header = Mock(spec=PixelDataHeader)
        image = DicomCubeImage(raw_image, pixel_header)
        
        with pytest.raises(InvalidCubeFileError) as exc_info:
            DicomCubeImageIO.save(image, "test.dcb", file_type="invalid")
        
        error = exc_info.value
        assert "Unsupported file type" in str(error)
        assert "invalid" in str(error)
        assert "save operation" in str(error)
    
    def test_save_with_none_image(self):
        """Test saving with None image."""
        with pytest.raises(DataConsistencyError) as exc_info:
            DicomCubeImageIO.save(None, "test.dcb")
        
        error = exc_info.value
        assert "image" in str(error)
        assert "cannot be None" in str(error)
        assert "save operation" in str(error)
    
    
    def test_load_from_nonexistent_dicom_folder(self):
        """Test loading from non-existent DICOM folder."""
        with pytest.raises(InvalidCubeFileError) as exc_info:
            DicomCubeImageIO.load_from_dicom_folder("/nonexistent/folder")
        
        error = exc_info.value
        assert "Folder not found" in str(error)
        assert "load_from_dicom_folder operation" in str(error)
    
    def test_load_from_nifti_nonexistent_file(self):
        """Test loading from non-existent NIfTI file."""
        # Skip test if nibabel is not available
        try:
            import nibabel
        except ImportError:
            pytest.skip("nibabel not available, skipping NIfTI test")
            
        with pytest.raises(InvalidCubeFileError) as error:
            DicomCubeImageIO.load_from_nifti("/nonexistent/file.nii")
        
        assert ("file does not exist" in str(error) or "File not found" in str(error))
        assert "load_from_nifti operation" in str(error)
    
    def test_save_to_nifti_without_space(self):
        """Test saving to NIfTI without space information."""
        # Skip test if nibabel is not available
        try:
            import nibabel
        except ImportError:
            pytest.skip("nibabel not available, skipping NIfTI test")
            
        # Create a simple image without space information
        raw_data = np.random.randint(0, 1000, size=(5, 10, 15), dtype=np.uint16)
        pixel_header = PixelDataHeader(
            RescaleSlope=1.0,
            RescaleIntercept=0.0,
            OriginalPixelDtype="uint16",
            PixelDtype="uint16"
        )
        
        image = DicomCubeImage(raw_data, pixel_header, space=None)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            nifti_path = os.path.join(temp_dir, "test.nii.gz")
            
            with pytest.raises(InvalidCubeFileError) as error:
                DicomCubeImageIO.save_to_nifti(image, nifti_path)
            
            assert "Cannot save to NIfTI without space information" in str(error)
            assert "save_to_nifti operation" in str(error)

    def test_save_to_dicom_folder_with_none_image(self):
        """Test saving to DICOM folder with None image."""
        with pytest.raises(DataConsistencyError) as exc_info:
            DicomCubeImageIO.save_to_dicom_folder(None, "/tmp/output")
        
        error = exc_info.value
        assert "image" in str(error)
        assert "cannot be None" in str(error)
        assert "save_to_dicom_folder operation" in str(error)





class TestStorageIntegration:
    """Test storage operations integration with exception handling."""
    
    def test_dcb_file_invalid_mode(self):
        """Test DcbFile with invalid mode."""
        with pytest.raises(InvalidCubeFileError) as exc_info:
            DcbFile("test.dcb", mode="invalid")
        
        error = exc_info.value
        assert "Invalid file mode" in str(error)
        assert "invalid" in str(error)
        assert "DcbFile constructor" in str(error)
    
    def test_dcb_file_empty_filename(self):
        """Test DcbFile with empty filename."""
        with pytest.raises(InvalidCubeFileError) as exc_info:
            DcbFile("", mode="r")
        
        error = exc_info.value
        assert "filename" in str(error)
        assert "cannot be empty" in str(error)
    
    def test_dcb_file_none_filename(self):
        """Test DcbFile with None filename."""
        with pytest.raises(InvalidCubeFileError) as exc_info:
            DcbFile(None, mode="r")
        
        error = exc_info.value
        assert "filename" in str(error)
        assert "cannot be None" in str(error)
    
    def test_dcb_file_read_invalid_header(self):
        """Test DcbFile reading invalid header."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dcb") as tmp_file:
            # Write truncated header (less than expected size)
            tmp_file.write(b"SHORT")
            tmp_path = tmp_file.name
        
        try:
            with pytest.raises(InvalidCubeFileError) as exc_info:
                dcb_file = DcbFile(tmp_path, mode="r")
            
            error = exc_info.value
            assert "File too small" in str(error) or "Failed to read file header" in str(error)
        finally:
            os.unlink(tmp_path)
    
    @patch('dicube.storage.dcb_file.get_codec')
    def test_dcb_s_file_encode_failure(self, mock_get_codec):
        """Test DcbSFile encoding failure."""
        # Mock codec to raise an exception
        mock_codec = Mock()
        mock_codec.encode_lossless.side_effect = Exception("Codec encoding failed")
        mock_get_codec.return_value = mock_codec
        
        dcb_file = DcbSFile("test.dcb", mode="w")
        frame_data = np.array([[1, 2], [3, 4]])
        
        with pytest.raises(CodecError) as exc_info:
            dcb_file._encode_one_frame(frame_data)
        
        error = exc_info.value
        assert "Failed to encode frame" in str(error)
        assert "jph codec" in str(error)
        assert "Codec encoding failed" in str(error)
    
    @patch('dicube.storage.dcb_file.get_codec')
    def test_dcb_s_file_decode_failure(self, mock_get_codec):
        """Test DcbSFile decoding failure."""
        # Mock codec to raise an exception
        mock_codec = Mock()
        mock_codec.decode.side_effect = Exception("Codec decoding failed")
        mock_get_codec.return_value = mock_codec
        
        dcb_file = DcbSFile("test.dcb", mode="r")
        encoded_data = b"fake_encoded_data"
        
        with pytest.raises(CodecError) as exc_info:
            dcb_file._decode_one_frame(encoded_data)
        
        error = exc_info.value
        assert "Failed to decode frame" in str(error)
        assert "jph codec" in str(error)
        assert "Codec decoding failed" in str(error)


class TestEndToEndErrorScenarios:
    """Test complete end-to-end error scenarios."""
    
    def test_complete_workflow_with_invalid_input(self):
        """Test complete workflow from invalid input to proper error handling."""
        # Try to create an image with invalid parameters
        with pytest.raises(DataConsistencyError) as exc_info:
            DicomCubeImage(None, None)
        
        # Verify the error contains proper context and suggestions
        error = exc_info.value
        assert isinstance(error, DicomCubeError)
        assert "raw_image" in str(error)
        assert "cannot be None" in str(error)
    
    def test_cascading_error_handling(self):
        """Test that errors cascade properly through the system."""
        # Create a scenario where multiple validation errors could occur
        raw_image = np.array([1])  # Invalid dimensions
        
        with pytest.raises(DataConsistencyError) as exc_info:
            DicomCubeImage(raw_image, "invalid_pixel_header")
        
        # Should catch the first validation error (raw_image dimensions)
        error = exc_info.value
        assert "too few dimensions" in str(error)
    
    def test_error_context_preservation(self):
        """Test that error context is preserved through the call stack."""
        with pytest.raises(InvalidCubeFileError) as exc_info:
            DicomCubeImageIO.load("/nonexistent/file.dcb")
        
        error = exc_info.value
        assert "load operation" in str(error)
        assert "File not found" in str(error)
        assert "/nonexistent/file.dcb" in str(error)
        
        # Verify error has proper attributes
        assert hasattr(error, 'context')
        assert hasattr(error, 'suggestion')
        assert hasattr(error, 'details')
    
    def test_exception_hierarchy_catching(self):
        """Test that exception hierarchy allows proper catching."""
        # Test catching specific exception
        try:
            DicomCubeImageIO.load("/nonexistent/file.dcb")
        except InvalidCubeFileError as e:
            assert isinstance(e, InvalidCubeFileError)
            assert isinstance(e, DicomCubeError)
        except Exception:
            pytest.fail("Should have caught InvalidCubeFileError")
        
        # Test catching base DicomCubeError
        try:
            DicomCubeImageIO.load("/nonexistent/file.dcb")
        except DicomCubeError as e:
            assert isinstance(e, DicomCubeError)
        except Exception:
            pytest.fail("Should have caught DicomCubeError")
    
    def test_error_message_formatting(self):
        """Test that error messages are properly formatted."""
        with pytest.raises(DataConsistencyError) as exc_info:
            DicomCubeImageIO.save(None, "test.dcb")
        
        error = exc_info.value
        error_str = str(error)
        
        # Check that the error message contains expected components
        assert "save operation:" in error_str  # Context
        assert "cannot be None" in error_str   # Main message
        assert "Suggestion:" in error_str      # Suggestion
    
    @patch('dicube.core.io.DcbSFile')
    def test_io_error_propagation(self, mock_dcb_file):
        """Test that I/O errors are properly propagated and wrapped."""
        # Mock DcbSFile to raise an exception during write
        mock_writer = Mock()
        mock_writer.write.side_effect = Exception("Disk full")
        mock_dcb_file.return_value = mock_writer
        
        raw_image = np.array([[1, 2], [3, 4]])
        pixel_header = Mock(spec=PixelDataHeader)
        image = DicomCubeImage(raw_image, pixel_header)
        
        with pytest.raises(InvalidCubeFileError) as exc_info:
            DicomCubeImageIO.save(image, "test.dcb", file_type="s")
        
        error = exc_info.value
        assert "Failed to save file" in str(error)
        assert "Disk full" in str(error)
        assert "save operation" in str(error)