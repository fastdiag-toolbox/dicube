"""
Unified DICOM Dataset Builder

Provides a simple, efficient interface for creating DICOM datasets from various sources.
Uses pydicom's built-in JSON conversion for maximum simplicity and reliability.
"""

import io
import inspect
import warnings
from typing import Optional, Union, Dict, Any
import numpy as np
from pydicom import Dataset
from pydicom.dataset import FileMetaDataset
from pydicom.encaps import encapsulate
from pydicom.uid import ExplicitVRLittleEndian, generate_uid


def create_file_meta(ds):
    """Create file meta information"""
    file_meta = FileMetaDataset()

    MODALITY_SOP_CLASS_MAP = {
        "CT": "1.2.840.10008.5.1.4.1.1.2",
        "MR": "1.2.840.10008.5.1.4.1.1.4",
        "US": "1.2.840.10008.5.1.4.1.1.6.1",
        "PT": "1.2.840.10008.5.1.4.1.1.128",
        "CR": "1.2.840.10008.5.1.4.1.1.1",
        "DX": "1.2.840.10008.5.1.4.1.1.1.1",
        "NM": "1.2.840.10008.5.1.4.1.1.20",
    }

    modality = ds.Modality if hasattr(ds, "Modality") else "CT"
    default_sop_uid = MODALITY_SOP_CLASS_MAP.get(modality, MODALITY_SOP_CLASS_MAP["CT"])

    file_meta.MediaStorageSOPClassUID = (
        ds.SOPClassUID if hasattr(ds, "SOPClassUID") else default_sop_uid
    )
    file_meta.MediaStorageSOPInstanceUID = (
        ds.SOPInstanceUID if hasattr(ds, "SOPInstanceUID") else generate_uid()
    )
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    return file_meta


def ensure_required_tags(ds):
    """Ensure required DICOM tags exist"""
    if not hasattr(ds, "SOPClassUID"):
        ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
    if not hasattr(ds, "SOPInstanceUID"):
        ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID


def set_pixel_data_attributes(ds: Dataset, pixel_header):
    """Set pixel data attributes based on pixel header information.
    
    Args:
        ds (Dataset): DICOM dataset to update
        pixel_header: Pixel header containing data type information
    """
    if pixel_header:
        ds.RescaleSlope = pixel_header.RescaleSlope
        ds.RescaleIntercept = pixel_header.RescaleIntercept
        
        # Set correct pixel data attributes based on pixel_header
        pixel_dtype = pixel_header.PixelDtype
        
        # Determine bits based on data type
        dtype_to_bits = {
            "uint8": (8, 8, 7, 0),
            "int8": (8, 8, 7, 1),
            "uint16": (16, 16, 15, 0),
            "int16": (16, 16, 15, 1),
            "uint32": (32, 32, 31, 0),
            "int32": (32, 32, 31, 1),
        }
        
        bits_allocated, bits_stored, high_bit, pixel_rep = dtype_to_bits.get(pixel_dtype, (16, 16, 15, 0))
        ds.BitsAllocated = bits_allocated
        ds.BitsStored = bits_stored
        ds.HighBit = high_bit
        ds.PixelRepresentation = pixel_rep
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"


def save_dicom(ds: Dataset, output_path: str):
    """Save DICOM file.
    
    Args:
        ds: DICOM dataset
        output_path: Output file path
    """
    
    sig = inspect.signature(Dataset.save_as)
    if "enforce_file_format" in sig.parameters:  # pydicom >= 3.0
        ds.save_as(output_path, enforce_file_format=True)
    else:
        # Ensure valid transfer syntax UID
        if hasattr(ds, 'file_meta') and hasattr(ds.file_meta, 'TransferSyntaxUID'):
            # Check if valid, replace with standard ExplicitVRLittleEndian if invalid
            try:
                from pydicom.uid import UID
                uid = UID(ds.file_meta.TransferSyntaxUID)
                if not hasattr(uid, 'is_little_endian'):
                    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            except (ValueError, AttributeError):
                # If UID is invalid, use standard ExplicitVRLittleEndian
                ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.save_as(output_path, write_like_original=False)


class DicomBuilder:
    """Unified DICOM dataset builder for all scenarios.
    
    This class provides a simple interface for creating DICOM datasets from:
    - Raw pixel arrays (uncompressed)
    - Compressed pixel data (JPEG2000, HTJ2K, etc.)
    - With or without decompression
    
    Uses pydicom's built-in Dataset.from_json() for reliable metadata handling.
    """
    
    @staticmethod
    def build(
        meta_dict: Dict[str, Any],
        pixel_header: Any,
        pixel_data: Optional[Union[np.ndarray, bytes]] = None,
        transfer_syntax_uid: Optional[str] = None,
        force_uncompressed: bool = False,
        is_compressed_data: bool = False
    ) -> Dataset:
        """Build a DICOM dataset from various input types.
        
        Args:
            meta_dict: DICOM metadata dictionary (JSON format or dict)
            pixel_header: Pixel header with data type information
            pixel_data: Either raw numpy array or compressed bytes
            transfer_syntax_uid: Transfer syntax UID (defaults to ExplicitVRLittleEndian)
            force_uncompressed: If True and data is compressed, decompress it
            is_compressed_data: If True, pixel_data is compressed bytes
            
        Returns:
            Complete DICOM dataset ready for saving
            
        Examples:
            # From raw pixel array
            ds = DicomBuilder.build(meta, header, pixel_array)
            
            # From compressed data, keep compressed
            ds = DicomBuilder.build(meta, header, compressed_bytes, 
                                   transfer_syntax_uid=htj2k_uid,
                                   is_compressed_data=True)
            
            # From compressed data, force decompression
            ds = DicomBuilder.build(meta, header, compressed_bytes,
                                   transfer_syntax_uid=htj2k_uid, 
                                   is_compressed_data=True,
                                   force_uncompressed=True)
        """
        # Create base dataset
        # Handle both DICOM JSON format and simple dictionary format
        if DicomBuilder._is_dicom_json_format(meta_dict):
            # DICOM JSON format with 'vr' and 'Value' keys
            ds = Dataset.from_json(meta_dict)
        else:
            # Simple dictionary format - convert to dataset directly
            ds = Dataset()
            for key, value in meta_dict.items():
                setattr(ds, key, value)
        
        # Handle any existing file metadata with warning
        if hasattr(ds, "file_meta"):
            warnings.warn("Found original file metadata, will be overridden")
        
        # Set file metadata
        file_meta = create_file_meta(ds)
        if transfer_syntax_uid:
            file_meta.TransferSyntaxUID = transfer_syntax_uid
        else:
            file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.file_meta = file_meta
        
        # Ensure required tags
        ensure_required_tags(ds)
        
        # Set pixel data attributes
        set_pixel_data_attributes(ds, pixel_header)
        
        # Handle pixel data
        if pixel_data is not None:
            if is_compressed_data:
                # Handle compressed data
                if force_uncompressed:
                    # Decompress the data
                    decompressed = DicomBuilder._decompress_pixel_data(
                        pixel_data, ds, transfer_syntax_uid
                    )
                    if decompressed is not None:
                        ds.PixelData = decompressed.tobytes()
                        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
                    else:
                        # Fallback to compressed if decompression fails
                        ds.PixelData = encapsulate([pixel_data])
                else:
                    # Keep compressed
                    ds.PixelData = encapsulate([pixel_data])
            else:
                # Handle raw pixel array
                if isinstance(pixel_data, np.ndarray):
                    # Ensure correct data type
                    target_dtype = getattr(np, pixel_header.PixelDtype, pixel_data.dtype)
                    if pixel_data.dtype != target_dtype:
                        pixel_data = pixel_data.astype(target_dtype)
                    ds.PixelData = pixel_data.tobytes()
                else:
                    # Already raw bytes
                    ds.PixelData = pixel_data
        
        return ds
    
    @staticmethod
    def _is_dicom_json_format(meta_dict: Dict[str, Any]) -> bool:
        """Check if the metadata dictionary is in DICOM JSON format.
        
        DICOM JSON format has structure like:
        {"00080005": {"vr": "CS", "Value": ["ISO_IR 100"]}}
        
        Simple format has structure like:
        {"PatientName": "Test^Patient", "Modality": "CT"}
        """
        if not meta_dict:
            return False
            
        # Check a few entries to determine format
        sample_keys = list(meta_dict.keys())[:3]
        for key in sample_keys:
            value = meta_dict[key]
            # DICOM JSON format: values are dicts with 'vr' key
            if isinstance(value, dict) and 'vr' in value:
                return True
            # Simple format: values are direct values (str, int, etc.)
            elif not isinstance(value, dict):
                return False
        
        # If all values are dicts but no 'vr' found, assume simple format
        return False
    
    @staticmethod
    def _decompress_pixel_data(
        compressed_data: bytes, 
        ds: Dataset,
        original_transfer_syntax: str
    ) -> Optional[np.ndarray]:
        """Attempt to decompress pixel data.
        
        Returns:
            Decompressed pixel array or None if decompression fails
        """
        try:
            # Create temporary dataset for decompression
            temp_ds = Dataset()
            temp_ds.update(ds)
            temp_ds.PixelData = encapsulate([compressed_data])
            temp_ds.file_meta.TransferSyntaxUID = original_transfer_syntax
            
            # Attempt decompression
            return temp_ds.pixel_array
        except Exception as e:
            warnings.warn(f"Failed to decompress pixel data: {e}")
            return None
    
    @staticmethod
    def to_bytes(ds: Dataset) -> bytes:
        """Serialize DICOM dataset to bytes.
        
        Args:
            ds: DICOM dataset to serialize
            
        Returns:
            DICOM file as bytes
        """
        buffer = io.BytesIO()
        save_dicom(ds, buffer)
        buffer.seek(0)
        return buffer.read()