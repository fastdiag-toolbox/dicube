import json
import os
import struct

# from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import numpy as np
import zstandard as zstd
from spacetransformer import Space

from ..codecs import get_codec
from ..core.pixel_header import PixelDataHeader
from ..dicom import DicomMeta
from ..dicom.dicom_status import DicomStatus
from ..validation import (
    validate_not_none,
    validate_parameter_type,
    validate_string_not_empty
)
from ..exceptions import (
    InvalidCubeFileError,
    CodecError,
    MetaDataError,
    DataConsistencyError
)

"""File Format Specification for DiCube (DCB) Files

-----------------------------------------------------------------
| File Header (Fixed length: 100 bytes)                           |
|   magic: 8 bytes (e.g. b"DICUBE")                              |
|   version: 4 bytes (unsigned int)                              |
|   dicom_status_offset: 8 bytes (Q)                              |
|   dicom_status_length: 8 bytes (Q)                              |
|   dicommeta_offset: 8 bytes (Q)                                |
|   dicommeta_length: 8 bytes (Q)                                |
|   space_offset: 8 bytes (Q)                                    |
|   space_length: 8 bytes (Q)                                    |
|   pixel_header_offset: 8 bytes (Q)                             |
|   pixel_header_length: 8 bytes (Q)                             |
|   encoded_frame_offsets_offset: 8 bytes (Q)                    |
|   encoded_frame_offsets_length: 8 bytes (Q)                    |
|   encoded_frame_lengths_offset: 8 bytes (Q)                    |
|   encoded_frame_lengths_length: 8 bytes (Q)                    |
|   encoded_frame_count: 8 bytes (Q)                             |
-----------------------------------------------------------------
| DicomMeta (compressed JSON, optional)                             |
-----------------------------------------------------------------
| Space (JSON)                                         |
-----------------------------------------------------------------
| PixelDataHeader (JSON, RescaleIntercept/Slope, status etc.)    |
-----------------------------------------------------------------
| encoded_frame_offsets (encoded_frame_count Q values)           |
-----------------------------------------------------------------
| encoded_frame_lengths (encoded_frame_count Q values)           |
-----------------------------------------------------------------
| encoded_frame_data[0]                                          |
-----------------------------------------------------------------
| encoded_frame_data[1] ...                                      |
-----------------------------------------------------------------
| ...                                                           |
-----------------------------------------------------------------
| encoded_frame_data[n-1]                                       |
-----------------------------------------------------------------

This format demonstrates how to store multi-frame images in a single file,
with offsets and lengths recorded in the header for random access.
"""


class DcbFile:
    """Base class implementing common file I/O logic for DiCube files.

    This class provides core functionality for:
    - Header structure management
    - write() workflow (header, metadata, space, header, offsets/lengths, images)
    - Common file operations (read/write)
    
    Subclasses should implement frame encoding via _encode_one_frame() and
    _decode_one_frame() methods, and set appropriate MAGIC and VERSION values.

    Attributes:
        HEADER_STRUCT (str): Struct format string for the header.
        MAGIC (bytes): Magic bytes for file identification.
        VERSION (int): File format version.
        TRANSFER_SYNTAX_UID (str, optional): DICOM transfer syntax UID.
        FILE_EXTENSION (str): File extension for this format.
    """

    HEADER_STRUCT = "<8sI13Q"
    MAGIC = b"DCMCUBE\x00"
    VERSION = 1
    TRANSFER_SYNTAX_UID = None  # Base class has no specific transfer syntax
    FILE_EXTENSION = ".dcb"  # Default extension

    def __init__(self, filename: str, mode: str = "r"):
        """Initialize a DCB file object.

        Args:
            filename (str): The file path.
            mode (str): "r" for reading, "w" for writing, "a" for appending.
        """
        # Validate required parameters
        validate_string_not_empty(filename, "filename", "DcbFile constructor", InvalidCubeFileError)
        validate_parameter_type(mode, str, "mode", "DcbFile constructor", InvalidCubeFileError)
        
        if mode not in ("r", "w", "a"):
            raise InvalidCubeFileError(
                f"Invalid file mode: {mode}",
                context="DcbFile constructor",
                details={"mode": mode, "supported_modes": ["r", "w", "a"]},
                suggestion="Use 'r' for reading, 'w' for writing, or 'a' for appending"
            )
        
        # For write mode, ensure filename has correct extension
        if mode == "w":
            filename = self._ensure_correct_extension(filename)
        
        self.filename = filename
        self.mode = mode
        self._header = None  # Delay reading header until needed

        if os.path.exists(filename) and mode in ("r", "a"):
            self._read_header_and_check_type()

    def _ensure_correct_extension(self, filename: str) -> str:
        """Ensure the filename has the correct extension for this file type.
        
        Args:
            filename (str): The original filename.
            
        Returns:
            str: The filename with correct extension.
        """
        if not filename.endswith(self.FILE_EXTENSION):
            return filename + self.FILE_EXTENSION
        return filename

    def _read_header_and_check_type(self):
        """Read file header and determine the correct subclass."""
        try:
            hdr = self.read_header(verify_magic=False)  # Lazy read
            magic = hdr["magic"]
            version = hdr["version"]

            if magic != self.MAGIC:
                if magic == DcbSFile.MAGIC and version == DcbSFile.VERSION:
                    self.__class__ = DcbSFile
                else:
                    raise InvalidCubeFileError(
                        f"Unsupported file format",
                        context="file header validation",
                        details={"magic_number": magic, "file_path": self.filename},
                        suggestion="Ensure the file is a valid DicomCube file"
                    )
            self.VERSION = version
        except Exception as e:
            if isinstance(e, InvalidCubeFileError):
                raise
            raise InvalidCubeFileError(
                f"Failed to read file header: {str(e)}",
                context="file header validation",
                details={"file_path": self.filename}
            ) from e

    @property
    def header(self):
        """Get the file header, reading it from disk if not already loaded.
        
        Returns:
            dict: Dictionary containing header fields.
        """
        if self._header is None:
            self._header = self.read_header()
        return self._header

    def read_header(self, verify_magic: bool = True):
        """Read and parse the file header.

        Args:
            verify_magic (bool): If True, verify the magic number. Defaults to True.

        Returns:
            dict: Dictionary containing header fields.

        Raises:
            InvalidCubeFileError: If the file is not a valid DicomCube file.
        """
        if self._header:
            return self._header

        try:
            header_size = struct.calcsize(self.HEADER_STRUCT)
            with open(self.filename, "rb") as f:
                header_data = f.read(header_size)

            if len(header_data) < header_size:
                raise InvalidCubeFileError(
                    f"File too small to contain valid header",
                    context="read_header operation",
                    details={"expected_size": header_size, "actual_size": len(header_data)},
                    suggestion="Ensure the file is a complete DicomCube file"
                )

            unpacked = struct.unpack(self.HEADER_STRUCT, header_data)
            (
                magic,
                version,
                dicom_status_offset,
                dicom_status_length,
                dicommeta_offset,
                dicommeta_length,
                space_offset,
                space_length,
                pixel_header_offset,
                pixel_header_length,
                frame_offsets_offset,
                frame_offsets_length,
                frame_lengths_offset,
                frame_lengths_length,
                frame_count,
            ) = unpacked

            if verify_magic and magic != self.MAGIC:
                raise InvalidCubeFileError(
                    f"Invalid file format magic number",
                    context="read_header operation",
                    details={"expected_magic": self.MAGIC, "actual_magic": magic},
                    suggestion="Ensure the file is a valid DicomCube file"
                )

            self._header = {
                "magic": magic,
                "version": version,
                "dicom_status_offset": dicom_status_offset,
                "dicom_status_length": dicom_status_length,
                "dicommeta_offset": dicommeta_offset,
                "dicommeta_length": dicommeta_length,
                "space_offset": space_offset,
                "space_length": space_length,
                "pixel_header_offset": pixel_header_offset,
                "pixel_header_length": pixel_header_length,
                "frame_offsets_offset": frame_offsets_offset,
                "frame_offsets_length": frame_offsets_length,
                "frame_lengths_offset": frame_lengths_offset,
                "frame_lengths_length": frame_lengths_length,
                "frame_count": frame_count,
            }
            return self._header
        except Exception as e:
            if isinstance(e, InvalidCubeFileError):
                raise
            raise InvalidCubeFileError(
                f"Failed to read file header: {str(e)}",
                context="read_header operation",
                details={"file_path": self.filename}
            ) from e

    def _prepare_metadata(
        self, 
        pixel_header: PixelDataHeader,
        dicom_meta: Optional[DicomMeta] = None,
        space: Optional[Space] = None,
        dicom_status: Optional[DicomStatus] = None,
    ):
        """Prepare metadata for writing to DCB file.
        
        Args:
            pixel_header (PixelDataHeader): Pixel data header information.
            dicom_meta (DicomMeta, optional): DICOM metadata. Defaults to None.
            space (Space, optional): Spatial information. Defaults to None.
            dicom_status (DicomStatus, optional): DICOM status. Defaults to None.
            
        Returns:
            dict: Dictionary containing prepared metadata components.
        """
        # Process dicom_status
        if dicom_status is None:
            # If not provided, try to get from pixel_header (backward compatibility)
            if hasattr(pixel_header, "DicomStatus"):
                dicom_status = pixel_header.DicomStatus
            else:
                dicom_status = DicomStatus.CONSISTENT

        # Convert DicomStatus enum to string for storage
        # If None was provided, dicom_status will be a DicomStatus enum at this point
        dicom_status_bin = dicom_status.value.encode("utf-8")

        # Process dicom_meta
        if dicom_meta:
            dicommeta_json = dicom_meta.to_json().encode("utf-8")
            dicommeta_gz = zstd.compress(dicommeta_json)
        else:
            dicommeta_gz = b""

        # Process space
        if space:
            # Convert space from internal (z,y,x) to file (x,y,z) format
            space_xyz = space.reverse_axis_order()
            space_json = space_xyz.to_json().encode("utf-8")
        else:
            space_json = b""

        # Process pixel_header
        pixel_header_bin = pixel_header.to_json().encode("utf-8")
        
        return {
            'dicom_status_bin': dicom_status_bin,
            'dicommeta_gz': dicommeta_gz,
            'space_json': space_json,
            'pixel_header_bin': pixel_header_bin
        }
    
    def _encode_frames(self, images: List):
        """Encode frames using parallel or serial processing.
        
        Args:
            images (List): List of frames to encode.
            
        Returns:
            List[bytes]: List of encoded frame data.
        """
        # Import get_num_threads function to avoid circular import
        from .. import get_num_threads
        num_threads = get_num_threads()
        
        if num_threads > 1:
            # Parallel encoding
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                encoded_blobs = list(
                    executor.map(lambda x: self._encode_one_frame(x), images)
                )
            return encoded_blobs
        else:
            # Serial encoding
            encoded_blobs = []
            for one_frame in images:
                encoded_bytes = self._encode_one_frame(one_frame)
                encoded_blobs.append(encoded_bytes)
            return encoded_blobs
    
    def _write_file_structure(self, f, metadata_components, encoded_frames, frame_count):
        """Write the complete file structure including metadata and frames.
        
        Args:
            f: File handle for writing.
            metadata_components (dict): Prepared metadata components.
            encoded_frames (List[bytes]): List of encoded frame data.
            frame_count (int): Number of frames.
            
        Returns:
            dict: Dictionary containing offset and length information for header.
        """
        # Write dicom_status
        dicom_status_offset = f.tell()
        f.write(metadata_components['dicom_status_bin'])
        dicom_status_length = f.tell() - dicom_status_offset

        # Write dicommeta_gz
        dicommeta_offset = f.tell()
        f.write(metadata_components['dicommeta_gz'])
        dicommeta_length = f.tell() - dicommeta_offset

        # Write space_json
        space_offset = f.tell()
        f.write(metadata_components['space_json'])
        space_length = f.tell() - space_offset

        # Write pixel_header_bin
        pixel_header_offset = f.tell()
        f.write(metadata_components['pixel_header_bin'])
        pixel_header_length = f.tell() - pixel_header_offset

        # Reserve offsets / lengths space
        frame_offsets_offset = f.tell()
        f.write(b"\x00" * (8 * frame_count))
        frame_offsets_length = 8 * frame_count

        frame_lengths_offset = f.tell()
        f.write(b"\x00" * (8 * frame_count))
        frame_lengths_length = 8 * frame_count

        # Write frames and collect offset/length info
        offsets = []
        lengths = []
        
        for encoded_bytes in encoded_frames:
            offset_here = f.tell()
            f.write(encoded_bytes)
            length_here = f.tell() - offset_here

            offsets.append(offset_here)
            lengths.append(length_here)

        # Backfill offsets & lengths
        current_pos = f.tell()
        f.seek(frame_offsets_offset)
        for off in offsets:
            f.write(struct.pack("<Q", off))

        f.seek(frame_lengths_offset)
        for lng in lengths:
            f.write(struct.pack("<Q", lng))

        # Go back to the end of the file
        f.seek(current_pos)
        
        return {
            'dicom_status_offset': dicom_status_offset,
            'dicom_status_length': dicom_status_length,
            'dicommeta_offset': dicommeta_offset,
            'dicommeta_length': dicommeta_length,
            'space_offset': space_offset,
            'space_length': space_length,
            'pixel_header_offset': pixel_header_offset,
            'pixel_header_length': pixel_header_length,
            'frame_offsets_offset': frame_offsets_offset,
            'frame_offsets_length': frame_offsets_length,
            'frame_lengths_offset': frame_lengths_offset,
            'frame_lengths_length': frame_lengths_length,
        }

    def write(
        self,
        images: List,  # Can be List[np.ndarray] or List[Tuple] for ROI data
        pixel_header: PixelDataHeader,
        dicom_meta: Optional[DicomMeta] = None,
        dicom_status: DicomStatus = DicomStatus.CONSISTENT,
        space: Optional[Space] = None,
    ):
        """Write image data and metadata to a DCB file.

        This is a generic write method that subclasses can reuse, customizing 
        single-frame encoding via _encode_one_frame().

        Args:
            images (List): List of frames to write. Can be List[np.ndarray] for standard files,
                or List[Tuple[np.ndarray, np.ndarray, np.ndarray]] for ROI files.
            pixel_header (PixelDataHeader): PixelDataHeader instance containing pixel metadata.
            dicom_meta (DicomMeta, optional): DICOM metadata. Defaults to None.
            dicom_status (DicomStatus): DICOM status enumeration. Defaults to DicomStatus.CONSISTENT.
            space (Space, optional): Spatial information. Defaults to None.
        """
        if images is None:
            images = []
        frame_count = len(images)

        # Prepare metadata components
        metadata_components = self._prepare_metadata(
            pixel_header, dicom_meta, space, dicom_status
        )
        
        # Encode frames
        encoded_frames = self._encode_frames(images)

        # Write file structure
        header_size = struct.calcsize(self.HEADER_STRUCT)
        
        with open(self.filename, "wb") as f:
            # Write placeholder header
            f.write(b"\x00" * header_size)
            
            # Write file structure and get offset information
            offset_info = self._write_file_structure(
                f, metadata_components, encoded_frames, frame_count
            )

            # Backfill header
            f.seek(0)
            header_data = struct.pack(
                self.HEADER_STRUCT,
                self.MAGIC,
                self.VERSION,
                offset_info['dicom_status_offset'],
                offset_info['dicom_status_length'],
                offset_info['dicommeta_offset'],
                offset_info['dicommeta_length'],
                offset_info['space_offset'],
                offset_info['space_length'],
                offset_info['pixel_header_offset'],
                offset_info['pixel_header_length'],
                offset_info['frame_offsets_offset'],
                offset_info['frame_offsets_length'],
                offset_info['frame_lengths_offset'],
                offset_info['frame_lengths_length'],
                frame_count,
            )
            f.write(header_data)

    def read_meta(self, DicomMetaClass=DicomMeta):
        """Read DICOM metadata from the file.

        Args:
            DicomMetaClass (class): The class to use for creating the DicomMeta object.
                Defaults to DicomMeta.

        Returns:
            DicomMeta: The DICOM metadata, or None if not present in the file.
        """
        hdr = self.header
        dicommeta_offset = hdr["dicommeta_offset"]
        dicommeta_length = hdr["dicommeta_length"]

        if dicommeta_length == 0:
            return None

        with open(self.filename, "rb") as f:
            f.seek(dicommeta_offset)
            dicommeta_gz = f.read(dicommeta_length)

        dicommeta_json = zstd.decompress(dicommeta_gz)
        dicommeta_dict = json.loads(dicommeta_json.decode("utf-8"))

        try:
            return DicomMetaClass.from_json(json.dumps(dicommeta_dict))
        except Exception as e:
            # Backwards compatibility for older file format
            return DicomMetaClass(
                dicommeta_dict, ["slice_{i:04d}.dcm" for i in range(hdr["frame_count"])]
            )

    def read_space(self, SpaceClass=Space):
        """Read spatial information from the file.

        Args:
            SpaceClass (class): The class to use for creating the Space object.
                Defaults to Space.

        Returns:
            Space: The spatial information, or None if not present in the file.
        """
        hdr = self.header
        space_offset = hdr["space_offset"]
        space_length = hdr["space_length"]

        if space_length == 0:
            return None

        with open(self.filename, "rb") as f:
            f.seek(space_offset)
            space_json = f.read(space_length)

        try:
            space = SpaceClass.from_json(space_json.decode("utf-8"))
            # Convert from file (x,y,z) format to internal (z,y,x) format
            return space.reverse_axis_order()
        except Exception as e:
            # If space reading fails, return None
            return None

    def read_pixel_header(self, HeaderClass=PixelDataHeader):
        """Read pixel header information from the file.

        Args:
            HeaderClass (class): The class to use for creating the PixelDataHeader object.
                Defaults to PixelDataHeader.

        Returns:
            PixelDataHeader: The pixel header information.

        Raises:
            ValueError: If the pixel header is not found in the file.
        """
        hdr = self.header
        pixel_header_offset = hdr["pixel_header_offset"]
        pixel_header_length = hdr["pixel_header_length"]

        if pixel_header_length == 0:
            raise ValueError("Pixel header not found in file.")

        with open(self.filename, "rb") as f:
            f.seek(pixel_header_offset)
            pixel_header_bin = f.read(pixel_header_length)

        pixel_header_json = pixel_header_bin.decode("utf-8")
        return HeaderClass.from_json(pixel_header_json)

    def read_images(self):
        """Read all image frames from the file.
            
        Returns:
            List[np.ndarray] or np.ndarray: The decoded image frames. If the number of frames is 1,
                returns a single numpy array, otherwise returns a list of arrays.
        """
        # Import get_num_threads function to avoid circular import
        from .. import get_num_threads
        
        hdr = self.header
        frame_count = hdr["frame_count"]
        
        if frame_count == 0:
            # No frames to read
            pixel_header = self.read_pixel_header()
            return np.array([], dtype=pixel_header.OriginalPixelDtype)

        # Read frame offsets and lengths
        frame_offsets_offset = hdr["frame_offsets_offset"]
        frame_offsets_length = hdr["frame_offsets_length"]
        frame_lengths_offset = hdr["frame_lengths_offset"]
        frame_lengths_length = hdr["frame_lengths_length"]
        
        with open(self.filename, "rb") as f:
            # Read frame offsets
            f.seek(frame_offsets_offset)
            frame_offsets_bin = f.read(frame_offsets_length)
            frame_offsets = struct.unpack(f"<{frame_count}Q", frame_offsets_bin)
            
            # Read frame lengths
            f.seek(frame_lengths_offset)
            frame_lengths_bin = f.read(frame_lengths_length)
            frame_lengths = struct.unpack(f"<{frame_count}Q", frame_lengths_bin)
            
            # Read each frame data
            frame_data_list = []
            for offset, length in zip(frame_offsets, frame_lengths):
                if offset == 0 or length == 0:
                    # Skip empty frames
                    frame_data_list.append(None)
                    continue

                f.seek(offset)
                encoded_bytes = f.read(length)
                frame_data_list.append(encoded_bytes)

        # Decode frames (with parallelization if needed)
        frames = []
        
        num_threads = get_num_threads()
        if num_threads > 1 and frame_count > 1:
            # Parallel decoding
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                frames = list(
                    executor.map(
                        lambda x: None if x is None else self._decode_one_frame(x),
                        frame_data_list,
                    )
                )
        else:
            # Serial decoding
            frames = [
                None if data is None else self._decode_one_frame(data)
                for data in frame_data_list
            ]

        # Filter out None frames
        frames = [f for f in frames if f is not None]

        if len(frames) == 0:
            # Return empty array if no frames were decoded
            pixel_header = self.read_pixel_header()
            return np.array([], dtype=pixel_header.OriginalPixelDtype)
        elif len(frames) == 1:
            # Return single frame directly
            return frames[0]
        else:
            # Return list of frames
            return frames

    def _encode_one_frame(self, frame_data: np.ndarray) -> bytes:
        """Encode a single frame to bytes.
        
        Default implementation returns empty bytes.
        Subclasses should override this method to implement specific encoding.

        Args:
            frame_data (np.ndarray): The image frame to encode.
            
        Returns:
            bytes: The encoded frame data.
        """
        return np.array([], dtype=self.pixel_header.OriginalPixelDtype)

    def _decode_one_frame(self, bytes) -> np.ndarray:
        """Decode a single frame from bytes.
        
        Default implementation returns an empty array with the correct data type.
        Subclasses should override this method to implement specific decoding.

        Args:
            bytes (bytes): The encoded frame data.

        Returns:
            np.ndarray: The decoded image frame.
        """
        return np.array([], dtype=self.header_data['pixel_header'].OriginalPixelDtype)

    def read_dicom_status(self) -> DicomStatus:
        """Read DICOM status information from the file.

        Returns:
            DicomStatus: The DICOM status enum value, or DicomStatus.CONSISTENT if not present.
        """
        hdr = self.header
        dicom_status_offset = hdr["dicom_status_offset"]
        dicom_status_length = hdr["dicom_status_length"]

        if dicom_status_length == 0:
            return DicomStatus.CONSISTENT

        with open(self.filename, "rb") as f:
            f.seek(dicom_status_offset)
            dicom_status_bin = f.read(dicom_status_length)

        return DicomStatus(dicom_status_bin.decode("utf-8"))
    
    def get_transfer_syntax_uid(self) -> Optional[str]:
        """Get the DICOM transfer syntax UID for this file.

        Returns:
            str or None: The transfer syntax UID, or None if not defined.
        """
        return self.TRANSFER_SYNTAX_UID





class DcbSFile(DcbFile):
    """DICOM cube file implementation optimized for speed.

    This format prioritizes quick read/write operations while maintaining
    lossless compression with average compression ratio.

    Attributes:
        MAGIC (bytes): Magic bytes for file identification ("DCMCUBES").
        VERSION (int): File format version.
        TRANSFER_SYNTAX_UID (str): DICOM transfer syntax UID for HTJ2K Lossless.
        CODEC_NAME (str): Codec name used for compression.
        FILE_EXTENSION (str): File extension for speed-optimized format.
    """

    MAGIC = b"DCMCUBES"
    VERSION = 1
    TRANSFER_SYNTAX_UID = "1.2.840.10008.1.2.4.201"  # HTJ2K Lossless
    CODEC_NAME = "jph"
    FILE_EXTENSION = ".dcbs"

    def _encode_one_frame(self, frame_data: np.ndarray) -> bytes:
        """Encode a single frame using the HTJ2K codec.

        Args:
            frame_data (np.ndarray): The frame data to encode.

        Returns:
            bytes: The encoded frame data.
        
        Raises:
            CodecError: If encoding fails.
        """
        try:
            codec = get_codec(self.CODEC_NAME)
            return codec.encode_lossless(frame_data)
        except Exception as e:
            raise CodecError(
                f"Failed to encode frame using {self.CODEC_NAME} codec: {str(e)}",
                context="frame encoding operation",
                details={"codec_name": self.CODEC_NAME, "frame_shape": frame_data.shape if hasattr(frame_data, 'shape') else None}
            ) from e

    def _decode_one_frame(self, bytes) -> np.ndarray:
        """Decode a single frame using the HTJ2K codec.

        Args:
            bytes (bytes): The encoded frame data.

        Returns:
            np.ndarray: The decoded frame data.
        
        Raises:
            CodecError: If decoding fails.
        """
        try:
            codec = get_codec(self.CODEC_NAME)
            return codec.decode(bytes)
        except Exception as e:
            raise CodecError(
                f"Failed to decode frame using {self.CODEC_NAME} codec: {str(e)}",
                context="frame decoding operation",
                details={"codec_name": self.CODEC_NAME, "data_size": len(bytes) if bytes else 0}
            ) from e


class DcbAFile(DcbFile):
    """DICOM cube file implementation optimized for archiving.

    This format prioritizes high compression ratio for long-term storage
    while maintaining lossless compression, at the expense of speed.

    Attributes:
        MAGIC (bytes): Magic bytes for file identification ("DCMCUBEA").
        VERSION (int): File format version.
        TRANSFER_SYNTAX_UID (str, optional): DICOM transfer syntax UID, set when codec is selected.
        CODEC_NAME (str, optional): Codec name, set when codec is selected.
        FILE_EXTENSION (str): File extension for archive-optimized format.
    """

    MAGIC = b"DCMCUBEA"
    VERSION = 1
    TRANSFER_SYNTAX_UID = None  # To be defined when codec is selected
    CODEC_NAME = None  # To be defined when codec is selected
    FILE_EXTENSION = ".dcba"


class DcbLFile(DcbFile):
    """DICOM cube file implementation with lossy compression.

    This format prioritizes very high compression ratio by using lossy compression,
    sacrificing some image quality and perfect reconstruction.

    Attributes:
        MAGIC (bytes): Magic bytes for file identification ("DCMCUBEL").
        VERSION (int): File format version.
        TRANSFER_SYNTAX_UID (str, optional): DICOM transfer syntax UID, set when codec is selected.
        CODEC_NAME (str, optional): Codec name, set when codec is selected.
        FILE_EXTENSION (str): File extension for lossy compression format.
    """

    MAGIC = b"DCMCUBEL"
    VERSION = 1
    TRANSFER_SYNTAX_UID = None  # To be defined when codec is selected
    CODEC_NAME = None  # To be defined when codec is selected
    FILE_EXTENSION = ".dcbl"