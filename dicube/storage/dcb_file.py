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

"""
File Format Specification
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
    """
    Base class implementing common file I/O logic including:
    - Header structure
    - write() workflow (header, metadata, space, header, offsets/lengths, images)
    - ENCODER/DECODER/MAGIC/VERSION that can be overridden by subclasses
    - Subclasses only need to implement frame encoding via _encode_one_frame()
    """

    HEADER_STRUCT = "<8sI13Q"
    MAGIC = b"DCMCUBE\x00"
    VERSION = 1

    def __init__(self, filename: str, mode: str = "r"):
        """
        Args:
            filename: The file path.
            mode: "r" for reading, "w" for writing, "a" for appending.
        """
        self.filename = filename
        self.mode = mode
        self._header = None  # Delay reading header until needed

        if os.path.exists(filename) and mode in ("r", "a"):
            self._read_header_and_check_type()

    def _read_header_and_check_type(self):
        """Read file header and determine the correct subclass."""
        hdr = self.read_header(verify_magic=False)  # Lazy read
        magic = hdr["magic"]
        version = hdr["version"]

        if magic != self.MAGIC:
            if magic == DcbSFile.MAGIC and version == DcbSFile.VERSION:
                self.__class__ = DcbSFile
            else:
                raise ValueError(f"不支持的文件格式, 魔数: {magic}")
        self.VERSION = version

    @property
    def header(self):
        if self._header is None:
            self._header = self.read_header()
        return self._header

    def read_header(self, verify_magic: bool = True):
        """Read and parse the file header, but only when needed."""
        if self._header:
            return self._header

        header_size = struct.calcsize(self.HEADER_STRUCT)
        with open(self.filename, "rb") as f:
            header_data = f.read(header_size)

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
            raise ValueError("Not a valid DicomCube file.")

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

    def write(
        self,
        images: List,  # Can be List[np.ndarray] or List[Tuple] for ROI data
        pixel_header: PixelDataHeader,
        dicom_meta: Optional[DicomMeta] = None,
        space: Optional[Space] = None,
        num_threads: int = 4,
        dicom_status: Optional[str] = None,
    ):
        """
        Generic write method that subclasses can reuse, customizing single-frame encoding via _encode_one_frame().

        Args:
            images: List of frames to write (List[np.ndarray] for standard files,
                   List[Tuple[np.ndarray, np.ndarray, np.ndarray]] for ROI files)
            pixel_header: PixelDataHeader instance
            dicom_meta: Optional DicomMeta instance
            space: Optional Space instance
            num_threads: Number of worker threads for parallel encoding (default: 4)
            dicom_status: Optional DicomStatus string value (default: None)
        """
        if images is None:
            images = []
        frame_count = len(images)

        # (1) 处理 dicom_status
        if dicom_status is None:
            # 如果没有提供，尝试从pixel_header中获取（向后兼容）
            if hasattr(pixel_header, "DicomStatus"):
                dicom_status = pixel_header.DicomStatus
            else:
                dicom_status = DicomStatus.CONSISTENT.value

        dicom_status_bin = dicom_status.encode("utf-8")

        # (2) 处理 dicom_meta
        if dicom_meta:
            dicommeta_json = dicom_meta.to_json().encode("utf-8")
            dicommeta_gz = zstd.compress(dicommeta_json)
        else:
            dicommeta_gz = b""

        # (3) 处理 space
        if space:
            space_json = space.to_json().encode("utf-8")
        else:
            space_json = b""

        # (4) 处理 pixel_header
        pixel_header_bin = pixel_header.to_json().encode("utf-8")

        # (5) 先写一个空的头占位
        header_size = struct.calcsize(self.HEADER_STRUCT)

        with open(self.filename, "wb") as f:
            f.write(b"\x00" * header_size)  # 先占位

            # (6) 写 dicom_status
            dicom_status_offset = f.tell()
            f.write(dicom_status_bin)
            dicom_status_length = f.tell() - dicom_status_offset

            # (7) 写 dicommeta_gz
            dicommeta_offset = f.tell()
            f.write(dicommeta_gz)
            dicommeta_length = f.tell() - dicommeta_offset

            # (8) 写 space_json
            space_offset = f.tell()
            f.write(space_json)
            space_length = f.tell() - space_offset

            # (9) 写 pixel_header_bin
            pixel_header_offset = f.tell()
            f.write(pixel_header_bin)
            pixel_header_length = f.tell() - pixel_header_offset

            # (10) 预留 offsets / lengths 空间
            frame_offsets_offset = f.tell()
            f.write(b"\x00" * (8 * frame_count))
            frame_offsets_length = 8 * frame_count

            frame_lengths_offset = f.tell()
            f.write(b"\x00" * (8 * frame_count))
            frame_lengths_length = 8 * frame_count

            # (11) 逐帧编码
            offsets = []
            lengths = []

            if num_threads is not None and num_threads > 1:
                # 并行编码
                with ThreadPoolExecutor(max_workers=num_threads) as executor:
                    # 先把所有帧编码
                    encoded_blobs = list(
                        executor.map(lambda x: self._encode_one_frame(x), images)
                    )

                    # 写入文件并记录offset/length
                    for encoded_bytes in encoded_blobs:
                        offset_here = f.tell()
                        f.write(encoded_bytes)
                        length_here = f.tell() - offset_here

                        offsets.append(offset_here)
                        lengths.append(length_here)
            else:
                # 串行编码
                for one_frame in images:
                    offset_here = f.tell()
                    encoded_bytes = self._encode_one_frame(one_frame)
                    f.write(encoded_bytes)
                    length_here = f.tell() - offset_here

                    offsets.append(offset_here)
                    lengths.append(length_here)

            # (12) 回填 offsets & lengths
            current_pos = f.tell()
            f.seek(frame_offsets_offset)
            for off in offsets:
                f.write(struct.pack("<Q", off))

            f.seek(frame_lengths_offset)
            for lng in lengths:
                f.write(struct.pack("<Q", lng))

            # 回到文件末尾
            f.seek(current_pos)

            # (13) 回填头部
            f.seek(0)
            header_data = struct.pack(
                self.HEADER_STRUCT,
                self.MAGIC,
                self.VERSION,
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
            )
            f.write(header_data)

    def read_meta(self, DicomMetaClass=DicomMeta):
        """
        Read and parse the DicomMeta section.

        Args:
            DicomMetaClass: Class to use for instantiating DicomMeta (default: DicomMeta)

        Returns:
            DicomMeta: Parsed metadata object
        """
        hdr = self.header
        if hdr["dicommeta_length"] == 0:
            return None

        with open(self.filename, "rb") as f:
            f.seek(hdr["dicommeta_offset"])
            dicommeta_gz = f.read(hdr["dicommeta_length"])
        dicommeta_json = zstd.decompress(dicommeta_gz).decode("utf-8")

        if DicomMetaClass is not None:
            return DicomMetaClass.from_json(dicommeta_json)
        else:
            # 若没有提供类，就返回 JSON 字符串或 dict
            return json.loads(dicommeta_json)

    def read_space(self, SpaceClass=Space):
        """
        Read and parse the Space section.

        Args:
            SpaceClass: Class to use for instantiating Space (default: Space)

        Returns:
            Space: Parsed space object
        """
        hdr = self.header
        if hdr["space_length"] == 0:
            return None

        with open(self.filename, "rb") as f:
            f.seek(hdr["space_offset"])
            space_json_bin = f.read(hdr["space_length"])
        space_json = space_json_bin.decode("utf-8")

        if SpaceClass is not None:
            # 假设 SpaceClass(**dict) 可以构造
            return SpaceClass.from_json(space_json)
        else:
            return json.loads(space_json)

    def read_pixel_header(self, HeaderClass=PixelDataHeader):
        """
        Read and parse the PixelDataHeader section.

        Args:
            HeaderClass: Class to use for instantiating header (default: PixelDataHeader)

        Returns:
            PixelDataHeader: Parsed pixel header object
        """
        hdr = self.header
        if hdr["pixel_header_length"] == 0:
            return {}

        with open(self.filename, "rb") as f:
            f.seek(hdr["pixel_header_offset"])
            raw_bin = f.read(hdr["pixel_header_length"])
        header_json = raw_bin.decode("utf-8")
        if HeaderClass is not None:
            # 假设 HeaderClass(**dict) 可以构造
            return HeaderClass.from_json(header_json)
        else:
            return header_json

    def read_images(self, num_threads: int = 4):
        """
        Read and decode all image frames.

        Args:
            num_threads: Number of worker threads for parallel decoding (default: 4)

        Returns:
            List[np.ndarray]: List of decoded image frames
        """
        hdr = self.header
        n = hdr["frame_count"]
        if n == 0:
            return []

        # 1) 先把 offsets & lengths 读出来
        offsets = []
        lengths = []
        with open(self.filename, "rb") as f:
            # frame_offsets
            f.seek(hdr["frame_offsets_offset"])
            for _ in range(n):
                (off,) = struct.unpack("<Q", f.read(8))
                offsets.append(off)

            # frame_lengths
            f.seek(hdr["frame_lengths_offset"])
            for _ in range(n):
                (lng,) = struct.unpack("<Q", f.read(8))
                lengths.append(lng)

        # 2) 一次性把各帧数据读到内存（减少多线程随机 I/O 开销）
        encoded_blobs = []
        with open(self.filename, "rb") as f:
            for i in range(n):
                f.seek(offsets[i])
                encoded_blobs.append(f.read(lengths[i]))

        # 3) 并行/串行地 decode
        if num_threads is not None and num_threads > 1:
            # 多进程解码
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                arr_list = list(executor.map(self._decode_one_frame, encoded_blobs))
        else:
            # 串行解码
            arr_list = [self._decode_one_frame(b) for b in encoded_blobs]

        # 4) 看 pixel_header 是否说明可以拼接
        info = self.read_pixel_header()
        dicom_status = getattr(info, "DicomStatus", None)
        if dicom_status == "Consistent":
            shapes = [a.shape for a in arr_list]
            if all(s == shapes[0] for s in shapes):
                # 可以堆叠
                return np.stack(arr_list, axis=0)
            else:
                return arr_list
        else:
            return arr_list

    def _encode_one_frame(self, frame_data: np.ndarray) -> bytes:
        """
        编码单帧图像。子类需要实现具体的编码方法。

        Args:
            frame_data: 输入图像数组

        Returns:
            bytes: 编码后的数据
        """
        return frame_data.tobytes()

    def _decode_one_frame(self, bytes) -> np.ndarray:
        """
        解码单帧图像。子类需要实现具体的解码方法。

        Args:
            bytes: 编码的数据
        """
        return np.frombuffer(bytes, dtype=self.pixel_header.PIXEL_DTYPE)

    def read_dicom_status(self) -> str:
        """
        Read and return the DicomStatus string.

        Returns:
            str: DicomStatus string value
        """
        hdr = self.header
        if hdr["dicom_status_length"] == 0:
            # 如果没有独立的DicomStatus，尝试从pixel_header中获取（向后兼容）
            pixel_header = self.read_pixel_header()
            if hasattr(pixel_header, "DicomStatus"):
                return pixel_header.DicomStatus
            return DicomStatus.CONSISTENT.value

        with open(self.filename, "rb") as f:
            f.seek(hdr["dicom_status_offset"])
            raw_bin = f.read(hdr["dicom_status_length"])
        return raw_bin.decode("utf-8")





class DcbSFile(DcbFile):
    """
    DICOM cube file implementation for Speed need.
    (losless, average compression ratio, speed sensitive)
    """

    MAGIC = b"DCMCUBES"
    VERSION = 1

    def _encode_one_frame(self, frame_data: np.ndarray) -> bytes:
        """
        Encode a single frame using OJPH compression.

        Args:
            frame_data: Input image array

        Returns:
            bytes: OJPH encoded image data
        """
        jph_codec = get_codec('jph')
        return jph_codec.encode(frame_data, reversible=True, num_decompositions=6)

    def _decode_one_frame(self, bytes) -> np.ndarray:
        """
        Decode a single OJPH compressed frame.

        Args:
            bytes: OJPH encoded image data

        Returns:
            np.ndarray: Decoded image array
        """
        jph_codec = get_codec('jph')
        return jph_codec.decode(bytes) 
    

class DcbAFile(DcbFile):
    """
    DICOM cube file implementation for Archiving need.
    (loseless, high compression ratio, speed insensitive)
    """

    MAGIC = b"DCMCUBEA"
    VERSION = 1

    NotImplementedError("DcbAFile is not implemented yet, no suitable codec")



class DcbLFile(DcbFile):
    """
    DICOM cube file implementation for Lossy need.
    (lossy, very high compression ratio, speed insensitive)
    """

    MAGIC = b"DCMCUBEL"
    VERSION = 1

    NotImplementedError("DcbLFile is not implemented yet, no suitable codec")