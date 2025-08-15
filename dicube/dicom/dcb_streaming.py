"""
DCB Streaming Reader for PACS Viewer

Provides efficient streaming access to DCB files for on-demand DICOM frame delivery.
Keeps files open and metadata cached for low-latency responses.
"""

import io
import struct
import warnings
from typing import Dict, Any

from pydicom import Dataset
from pydicom.dataset import FileMetaDataset
from pydicom.encaps import encapsulate
from pydicom.uid import generate_uid
import pydicom

from ..storage.dcb_file import DcbFile
from .dicom_io import save_dicom

# Required minimum PyDicom version
REQUIRED_PYDICOM_VERSION = "3.0.0"

class DcbStreamingReader:
    """DCB file streaming reader for PACS Viewer.
    
    Keeps files open and supports fast random access to frame data.
    
    Example:
        reader = DcbStreamingReader('study.dcbs')
        dicom_bytes = reader.get_dicom_for_frame(50)
        reader.close()
    """
    
    def __init__(self, dcb_file_path: str):
        """Initialize and preparse all metadata.
        
        Args:
            dcb_file_path: Path to DCB file
            
        Warnings:
            UserWarning: If PyDicom version is below 3.0.0, HTJ2K decoding may not work properly
        """
        # Check PyDicom version
        self._check_pydicom_version()
        
        self.file_path = dcb_file_path
        self.file_handle = None
        self.transfer_syntax_uid = None
        
        # Pre-parsed data
        self.header = None
        self.dicom_meta = None
        self.pixel_header = None
        self.space = None
        
        # Frame index information
        self.frame_offsets = []
        self.frame_lengths = []
        self.frame_count = 0
        
        # DcbFile instance (for reading metadata)
        self.dcb_file = None
        
        # Initialize
        self._open_and_parse()
    
    def _check_pydicom_version(self):
        """Check PyDicom version and warn if requirements not met.
        
        Warnings:
            UserWarning: If PyDicom version is below 3.0.0
        """
        current_version = pydicom.__version__
        if current_version < REQUIRED_PYDICOM_VERSION:
            warnings.warn(
                f"DcbStreamingReader requires PyDicom >= {REQUIRED_PYDICOM_VERSION} for full HTJ2K transfer syntax support. "
                f"Current PyDicom version is {current_version}, which may not be able to read pixel data. "
                f"Write functionality is not affected, but other applications may have issues reading. "
                f"Recommended upgrade: pip install pydicom>={REQUIRED_PYDICOM_VERSION}, requires python 3.10 or higher",
                UserWarning
            )
            self._has_pydicom_htj2k_support = False
        else:
            self._has_pydicom_htj2k_support = True

    def _open_and_parse(self):
        """Open file and parse all metadata."""
        try:
            # 1. Create DcbFile instance (will auto-detect file type)
            self.dcb_file = DcbFile(self.file_path, mode='r')
            
            # 2. Read and cache header information
            self.header = self.dcb_file.header
            self.frame_count = self.header['frame_count']
            
            # 3. Read and cache metadata
            self.dicom_meta = self.dcb_file.read_meta()
            self.pixel_header = self.dcb_file.read_pixel_header()
            self.space = self.dcb_file.read_space()
            
            # 4. Get transfer syntax UID (directly from file type)
            self.transfer_syntax_uid = self.dcb_file.get_transfer_syntax_uid()
            if not self.transfer_syntax_uid:
                # If file type doesn't define transfer syntax, use default uncompressed format
                self.transfer_syntax_uid = '1.2.840.10008.1.2.1'  # Explicit VR Little Endian
            
            # 5. Open file handle for reading frame data
            self.file_handle = open(self.file_path, 'rb')
            
            # 6. Read all frame offsets and lengths
            self._read_frame_indices()
            
        except Exception as e:
            self.close()
            raise RuntimeError(f"Failed to open and parse DCB file: {e}")
    
    def _read_frame_indices(self):
        """Read all frame offset and length information."""
        self.file_handle.seek(self.header['frame_offsets_offset'])
        
        # Read offsets
        for _ in range(self.frame_count):
            offset, = struct.unpack('<Q', self.file_handle.read(8))
            self.frame_offsets.append(offset)
        
        # Read lengths
        self.file_handle.seek(self.header['frame_lengths_offset'])
        for _ in range(self.frame_count):
            length, = struct.unpack('<Q', self.file_handle.read(8))
            self.frame_lengths.append(length)
    
    def get_dicom_for_frame(self, frame_index: int) -> bytes:
        """
        Get DICOM data for the specified frame.
        
        Args:
            frame_index: Frame index (0-based)
            
        Returns:
            bytes: Complete DICOM file data
            
        Raises:
            IndexError: If frame_index is out of range
            RuntimeError: If reading fails
        """
        # Validate index
        if not 0 <= frame_index < self.frame_count:
            raise IndexError(f"Frame index {frame_index} out of range [0, {self.frame_count})")
        
        try:
            # 1. Read encoded data for the frame
            encoded_pixel_data = self._read_encoded_frame(frame_index)
            
            # 2. Generate DICOM Dataset for the frame
            ds = self._create_dicom_dataset(frame_index, encoded_pixel_data)
            
            # 3. Serialize to DICOM file format
            return self._serialize_to_dicom_bytes(ds)
            
        except Exception as e:
            raise RuntimeError(f"Failed to create DICOM for frame {frame_index}: {e}")
    
    def _read_encoded_frame(self, frame_index: int) -> bytes:
        """Read encoded data for the specified frame directly."""
        offset = self.frame_offsets[frame_index]
        length = self.frame_lengths[frame_index]
        
        self.file_handle.seek(offset)
        return self.file_handle.read(length)
    
    def _create_dicom_dataset(self, frame_index: int, encoded_data: bytes) -> Dataset:
        """Quickly create DICOM Dataset."""
        # 1. Get metadata for the frame from cached DicomMeta
        if self.dicom_meta:
            frame_meta_dict = self.dicom_meta.index(frame_index)
        else:
            frame_meta_dict = {}
        
        # 2. Create Dataset
        ds = Dataset.from_json(frame_meta_dict)
        
        # 3. Create and set file metadata
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = ds.get('SOPClassUID', '1.2.840.10008.5.1.4.1.1.2')
        file_meta.MediaStorageSOPInstanceUID = ds.get('SOPInstanceUID', generate_uid())
        file_meta.TransferSyntaxUID = self.transfer_syntax_uid
        file_meta.ImplementationClassUID = generate_uid()
        
        ds.file_meta = file_meta
        
        # 4. Ensure necessary SOP information
        if not hasattr(ds, 'SOPClassUID'):
            ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        if not hasattr(ds, 'SOPInstanceUID'):
            ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        
        # 5. Set pixel-related attributes
        if self.pixel_header:
            ds.RescaleSlope = self.pixel_header.RescaleSlope
            ds.RescaleIntercept = self.pixel_header.RescaleIntercept
        
        # 6. Set pixel data (using encapsulated format for compressed data)
        ds.PixelData = encapsulate([encoded_data])
        
        return ds
    
    def _serialize_to_dicom_bytes(self, ds: Dataset) -> bytes:
        """Serialize Dataset to DICOM file byte stream."""
        # Use BytesIO to create DICOM file in memory
        buffer = io.BytesIO()
        save_dicom(ds, buffer)
        buffer.seek(0)
        return buffer.read()
    
    def get_frame_count(self) -> int:
        """Get total number of frames."""
        return self.frame_count
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get cached metadata information."""
        return {
            'frame_count': self.frame_count,
            'pixel_header': self.pixel_header.to_dict() if self.pixel_header else {},
            'has_dicom_meta': self.dicom_meta is not None,
            'has_space': self.space is not None,
            'transfer_syntax': self.transfer_syntax_uid,
            'file_type': self.dcb_file.__class__.__name__,
        }
    
    def close(self):
        """Close file handle."""
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
    
    def __enter__(self):
        """Support with statement."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Automatically close when exiting with statement."""
        self.close()
    
    def __del__(self):
        """Ensure file is closed on destruction."""
        self.close() 