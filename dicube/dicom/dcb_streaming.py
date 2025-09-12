"""
DCB Streaming Reader for PACS Viewer

Provides efficient streaming access to DCB files for on-demand DICOM frame delivery.
Keeps files open and metadata cached for low-latency responses.
"""

import io
import struct
import warnings
from typing import Dict, Any, Optional, Set
from collections import OrderedDict
import time
import threading

import pydicom

from ..storage.dcb_file import DcbFile
from .dicom_builder import DicomBuilder

# Required minimum PyDicom version
REQUIRED_PYDICOM_VERSION = "3.0.0"

class DcbStreamingReader:
    """DCB file streaming reader for PACS Viewer with intelligent caching.
    
    Features:
    - Chunk-based caching to balance memory usage and performance
    - Sliding window preloading for sequential browsing
    - LRU eviction policy for memory management
    - Thread-safe operations for concurrent access
    
    Example:
        reader = DcbStreamingReader('study.dcbs', chunk_size=10, cache_size=50)
        dicom_bytes = reader.get_dicom_for_frame(50)
        reader.close()
    """
    
    def __init__(self, dcb_file_path: str, chunk_size: int = 10, cache_size: int = 50, 
                 force_uncompressed: bool = False):
        """Initialize with intelligent caching parameters.
        
        Args:
            dcb_file_path: Path to DCB file
            chunk_size: Number of frames to load per chunk (default: 10)
            cache_size: Maximum number of frames to keep in memory (default: 50)
            force_uncompressed: If True, decode and save as uncompressed DICOM for better compatibility
            
        Warnings:
            UserWarning: If PyDicom version is below 3.0.0, HTJ2K decoding may not work properly
        """
        # Check PyDicom version
        self._check_pydicom_version()
        
        self.file_path = dcb_file_path
        self.file_handle = None
        self.transfer_syntax_uid = None
        self.force_uncompressed = force_uncompressed
        
        # Caching parameters
        self.chunk_size = max(1, chunk_size)
        self.cache_size = max(chunk_size, cache_size)
        
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
        
        # Caching infrastructure
        self._cache_lock = threading.RLock()
        self._file_lock = threading.Lock() 
        self._dicom_cache: OrderedDict[int, bytes] = OrderedDict()  # LRU cache
        self._loading_frames: Set[int] = set()  # Frames currently being loaded
        self._last_access_time = time.time()
        
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
        """Get DICOM data for the specified frame with intelligent caching.
        
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
        
        self._last_access_time = time.time()
        
        # Check cache first
        with self._cache_lock:
            if frame_index in self._dicom_cache:
                # Move to end (most recently used)
                dicom_data = self._dicom_cache.pop(frame_index)
                self._dicom_cache[frame_index] = dicom_data
                return dicom_data
        
        try:
            # Generate DICOM data
            dicom_data = self._create_dicom_for_frame(frame_index)
            
            # Cache the result
            self._cache_dicom_data(frame_index, dicom_data)
            
            # Trigger preloading for nearby frames
            self._trigger_preload(frame_index)
            
            return dicom_data
            
        except Exception as e:
            raise RuntimeError(f"Failed to create DICOM for frame {frame_index}: {e}")
            
    def _create_dicom_for_frame(self, frame_index: int) -> bytes:
        """Create DICOM data for a single frame."""
        # 1. Read encoded data for the frame
        encoded_pixel_data = self._read_encoded_frame(frame_index)
        
        # 2. Get metadata for the frame from cached DicomMeta
        if self.dicom_meta:
            frame_meta_dict = self.dicom_meta.index(frame_index)
        else:
            frame_meta_dict = {}
        
        # 3. Use unified DicomBuilder to create and serialize DICOM dataset
        ds = DicomBuilder.build(
            meta_dict=frame_meta_dict,
            pixel_header=self.pixel_header,
            pixel_data=encoded_pixel_data,
            transfer_syntax_uid=self.transfer_syntax_uid,
            force_uncompressed=self.force_uncompressed,
            is_compressed_data=True
        )
        
        # 4. Serialize to DICOM file format
        return DicomBuilder.to_bytes(ds)
    
    def _read_encoded_frame(self, frame_index: int) -> bytes:
        """Read encoded data for the specified frame directly."""
        offset = self.frame_offsets[frame_index]
        length = self.frame_lengths[frame_index]
        
        with self._file_lock:
            self.file_handle.seek(offset)
            return self.file_handle.read(length)
    
    
    def get_frame_count(self) -> int:
        """Get total number of frames."""
        return self.frame_count
    
    def _cache_dicom_data(self, frame_index: int, dicom_data: bytes):
        """Cache DICOM data with LRU eviction."""
        with self._cache_lock:
            # Remove if already exists (to update position)
            if frame_index in self._dicom_cache:
                del self._dicom_cache[frame_index]
            
            # Add to cache
            self._dicom_cache[frame_index] = dicom_data
            
            # Evict old entries if cache is full
            while len(self._dicom_cache) > self.cache_size:
                oldest_frame, _ = self._dicom_cache.popitem(last=False)
                
    def _trigger_preload(self, frame_index: int):
        """Trigger preloading of nearby frames in a separate thread."""
        def preload_worker():
            # Determine preload range (sliding window)
            start_idx = max(0, frame_index - self.chunk_size // 2)
            end_idx = min(self.frame_count, frame_index + self.chunk_size // 2 + 1)
            
            for idx in range(start_idx, end_idx):
                with self._cache_lock:
                    # Skip if already cached or being loaded
                    if idx in self._dicom_cache or idx in self._loading_frames:
                        continue
                    self._loading_frames.add(idx)
                
                try:
                    # Create DICOM data
                    dicom_data = self._create_dicom_for_frame(idx)
                    # Cache it
                    self._cache_dicom_data(idx, dicom_data)
                except Exception:
                    # Ignore preload errors
                    pass
                finally:
                    with self._cache_lock:
                        self._loading_frames.discard(idx)
        
        # Start preload in background thread
        threading.Thread(target=preload_worker, daemon=True).start()
                
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring."""
        with self._cache_lock:
            return {
                'cached_frames': len(self._dicom_cache),
                'loading_frames': len(self._loading_frames),
                'cache_size_limit': self.cache_size,
                'chunk_size': self.chunk_size,
                'last_access': self._last_access_time,
            }
            
    def clear_cache(self):
        """Clear all cached data to free memory."""
        with self._cache_lock:
            self._dicom_cache.clear()
            self._loading_frames.clear()

    def get_metadata(self) -> Dict[str, Any]:
        """Get cached metadata information."""
        return {
            'frame_count': self.frame_count,
            'pixel_header': self.pixel_header.to_dict() if self.pixel_header else {},
            'has_dicom_meta': self.dicom_meta is not None,
            'has_space': self.space is not None,
            'transfer_syntax': self.transfer_syntax_uid,
            'file_type': self.dcb_file.__class__.__name__,
            'cache_stats': self.get_cache_stats(),
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