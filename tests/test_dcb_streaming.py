"""
Tests for DcbStreamingReader functionality
"""

import os
import tempfile
import pytest
import pydicom

import dicube
from dicube.dicom.dcb_streaming import DcbStreamingReader

# 检查PyDicom版本是否支持HTJ2K
HAS_PYDICOM_HTJ2K_SUPPORT = pydicom.__version__ >= "3.0.0"


@pytest.fixture(scope="module")
def dicom_folder_path():
    """Fixture providing the path to sample DICOM data"""
    folder_path = "testdata/dicom/sample_150"
    if not os.path.exists(folder_path):
        pytest.skip("Sample DICOM data not available")
    return folder_path


@pytest.fixture(scope="module")
def dicube_image(dicom_folder_path):
    """Fixture providing a shared DiCube image loaded from DICOM folder"""
    return dicube.load_from_dicom_folder(dicom_folder_path)


@pytest.fixture(scope="module")
def dcbs_file(dicube_image):
    """Fixture providing a temporary DCBS file created from the shared image"""
    with tempfile.NamedTemporaryFile(suffix='.dcbs', delete=False) as tmp_file:
        temp_filename = tmp_file.name
    
    # Save as dcbs file
    dicube.save(dicube_image, temp_filename, file_type='s')
    
    yield temp_filename
    
    # Cleanup after all tests that use this fixture
    if os.path.exists(temp_filename):
        os.unlink(temp_filename)


def test_dcb_streaming_reader_basic(dcbs_file):
    """
    Test basic functionality of DcbStreamingReader
    """
    # Test streaming reader
    reader = DcbStreamingReader(dcbs_file)
    
    # Basic functionality tests
    assert reader.get_frame_count() > 0
    metadata = reader.get_metadata()
    assert 'frame_count' in metadata
    assert 'transfer_syntax' in metadata
    
    reader.close()


def test_dcb_streaming_reader_get_dicom(dcbs_file):
    """
    Test retrieving DICOM data for a single frame
    """
    import pydicom
    from io import BytesIO
    
    # Use streaming reader
    with DcbStreamingReader(dcbs_file) as reader:
        frame_count = reader.get_frame_count()
        
        # Test retrieving the first frame
        dicom_bytes = reader.get_dicom_for_frame(0)
        assert isinstance(dicom_bytes, bytes)
        assert len(dicom_bytes) > 0
        
        # Verify DICOM can be correctly parsed
        ds = pydicom.dcmread(BytesIO(dicom_bytes))
        assert hasattr(ds, 'PatientName')
        assert hasattr(ds, 'file_meta')
        
        # Test retrieving the last frame
        if frame_count > 1:
            dicom_bytes = reader.get_dicom_for_frame(frame_count - 1)
            assert isinstance(dicom_bytes, bytes)
            assert len(dicom_bytes) > 0


def test_dcb_streaming_reader_error_handling(dcbs_file):
    """
    Test error handling
    """
    with DcbStreamingReader(dcbs_file) as reader:
        frame_count = reader.get_frame_count()
        
        # Test invalid frame indices
        with pytest.raises(IndexError):
            reader.get_dicom_for_frame(frame_count)  # Out of range
        
        with pytest.raises(IndexError):
            reader.get_dicom_for_frame(-1)  # Negative index


def test_dcb_streaming_reader_context_manager(dcbs_file):
    """
    Test context manager functionality
    """
    # Test with statement
    with DcbStreamingReader(dcbs_file) as reader:
        frame_count = reader.get_frame_count()
        assert frame_count > 0
        
        # Get a middle frame
        mid_frame = frame_count // 2
        dicom_bytes = reader.get_dicom_for_frame(mid_frame)
        assert isinstance(dicom_bytes, bytes)
    
    # After exiting the with statement, file should be closed (through __exit__ method)


@pytest.mark.skipif(
    not HAS_PYDICOM_HTJ2K_SUPPORT,
    reason="PyDicom < 3.0.0 (HTJ2K not supported)"
)
def test_dcb_streaming_pixel_data_integrity(dicom_folder_path, dcbs_file):
    """
    Test pixel data integrity of streamed DICOM data
    
    Verifies:
    1. Decompression works properly
    2. Pixel data can be correctly read
    3. Pixel values after applying rescale slope and intercept match the original images
    4. Confirm that compressed format size is at least half the original size
    
    Note: This test requires PyDicom >= 3.0.0 for HTJ2K support
    """
    import pydicom
    import numpy as np
    import glob
    from io import BytesIO
    
    # Read all original DICOM files from sample data
    dcm_files = glob.glob(os.path.join(dicom_folder_path, "*.dcm"))
    
    # Read original DICOM files and sort by InstanceNumber
    original_dcms = []
    original_file_sizes = {}
    for dcm_file in dcm_files:
        ds = pydicom.dcmread(dcm_file)
        original_dcms.append(ds)
        original_file_sizes[ds.InstanceNumber] = os.path.getsize(dcm_file)
    
    # Sort by InstanceNumber (consistent with the default SortMethod.INSTANCE_NUMBER_ASC)
    original_dcms.sort(key=lambda x: x.InstanceNumber)
    
    # Use streaming reader
    with DcbStreamingReader(dcbs_file) as reader:
        frame_count = reader.get_frame_count()
        assert frame_count == len(original_dcms)
        
        # Select several indices for testing
        test_indices = [0]  # Always test first frame
        if frame_count > 1:
            test_indices.append(frame_count // 2)  # Test middle frame
            test_indices.append(frame_count - 1)   # Test last frame
        
        for idx in test_indices:
            # Get streamed DICOM data
            dicom_bytes = reader.get_dicom_for_frame(idx)
            
            # Verify compressed data size
            original_ds = original_dcms[idx]
            instance_number = original_ds.InstanceNumber
            original_size = original_file_sizes[instance_number]
            streamed_size = len(dicom_bytes)
            
            # Output size comparison information
            print(f"Frame {idx} (InstanceNumber {instance_number}): Original size: {original_size}, Streamed size: {streamed_size}, Ratio: {streamed_size/original_size:.2f}")
            
            # Add assertion to check compression ratio is not more than 0.55 (allows 5% margin)
            assert streamed_size / original_size <= 0.55, f"Compressed size ratio too high: {streamed_size/original_size:.2f}"
            
            # Test whether it can be correctly decompressed and read
            streamed_ds = pydicom.dcmread(BytesIO(dicom_bytes))
            
            # Get original pixel data (apply rescale slope and intercept)
            original_pixels = original_ds.pixel_array.astype(np.float32)
            if hasattr(original_ds, 'RescaleSlope'):
                original_pixels = original_pixels * original_ds.RescaleSlope
            if hasattr(original_ds, 'RescaleIntercept'):
                original_pixels = original_pixels + original_ds.RescaleIntercept
            
            # Get streamed pixel data (apply rescale slope and intercept)
            streamed_pixels = streamed_ds.pixel_array.astype(np.float32)
            if hasattr(streamed_ds, 'RescaleSlope'):
                streamed_pixels = streamed_pixels * streamed_ds.RescaleSlope
            if hasattr(streamed_ds, 'RescaleIntercept'):
                streamed_pixels = streamed_pixels + streamed_ds.RescaleIntercept
            
            # Compare pixel data
            np.testing.assert_allclose(
                original_pixels, 
                streamed_pixels, 
                rtol=1e-5, 
                atol=1e-5, 
                err_msg=f"Pixel data mismatch for frame {idx}"
            )
            
            # Verify image dimensions match
            assert original_ds.Rows == streamed_ds.Rows
            assert original_ds.Columns == streamed_ds.Columns
            
            # Compare other important DICOM attributes (exclude dynamically generated attributes)
            exclude_attrs = [
                'file_meta', 'is_little_endian', 'is_implicit_VR',
                'SOPInstanceUID'  # This might be regenerated
            ]
            
            for attr in original_ds.dir():
                # Skip attributes that don't need comparison
                if attr in exclude_attrs or attr.startswith('_'):
                    continue
                
                # Skip PixelData (already compared separately)
                if attr == 'PixelData':
                    continue
                
                # Ensure other attributes exist and are equal
                if hasattr(original_ds, attr) and hasattr(streamed_ds, attr):
                    original_value = getattr(original_ds, attr)
                    streamed_value = getattr(streamed_ds, attr)
                    
                    # For sequence types, convert to list for comparison
                    if hasattr(original_value, '__iter__') and not isinstance(original_value, (str, bytes)):
                        assert list(original_value) == list(streamed_value), f"Attribute {attr} mismatch"
                    else:
                        assert original_value == streamed_value, f"Attribute {attr} mismatch" 