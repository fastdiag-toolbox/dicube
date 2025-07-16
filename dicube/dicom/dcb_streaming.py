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

# 定义所需的最低PyDicom版本
REQUIRED_PYDICOM_VERSION = "3.0.0"

class DcbStreamingReader:
    """
    PACS Viewer 的 dcb 文件流式读取器
    保持文件打开状态，支持快速随机访问帧数据
    
    Example:
        reader = DcbStreamingReader('study.dcbs')
        dicom_bytes = reader.get_dicom_for_frame(50)
        reader.close()
    """
    
    def __init__(self, dcb_file_path: str):
        """
        初始化并预解析所有元数据
        
        Args:
            dcb_file_path: dcb 文件路径
            
        Warnings:
            UserWarning: 如果 PyDicom 版本低于 3.0.0，HTJ2K 解码可能无法正常工作
        """
        # 检查 PyDicom 版本
        self._check_pydicom_version()
        
        self.file_path = dcb_file_path
        self.file_handle = None
        self.transfer_syntax_uid = None
        
        # 预解析的数据
        self.header = None
        self.dicom_meta = None
        self.pixel_header = None
        self.space = None
        
        # 帧索引信息
        self.frame_offsets = []
        self.frame_lengths = []
        self.frame_count = 0
        
        # DcbFile 实例（用于读取元数据）
        self.dcb_file = None
        
        # 初始化
        self._open_and_parse()
    
    def _check_pydicom_version(self):
        """
        检查 PyDicom 版本，如果不满足要求则发出警告
        
        Warnings:
            UserWarning: 如果 PyDicom 版本低于 3.0.0
        """
        current_version = pydicom.__version__
        if current_version < REQUIRED_PYDICOM_VERSION:
            warnings.warn(
                f"DcbStreamingReader 需要 PyDicom >= {REQUIRED_PYDICOM_VERSION} 以完全支持 HTJ2K 传输语法。"
                f"当前 PyDicom 版本为 {current_version}，可能无法读取像素数据。"
                f"写入功能不受影响，但其他应用读取时可能会出现问题。建议升级: pip install pydicom>={REQUIRED_PYDICOM_VERSION}，需要 python 3.10 或更高版本",
                UserWarning
            )
            self._has_pydicom_htj2k_support = False
        else:
            self._has_pydicom_htj2k_support = True

    def _open_and_parse(self):
        """打开文件并解析所有元数据"""
        try:
            # 1. 创建 DcbFile 实例（会自动检测文件类型）
            self.dcb_file = DcbFile(self.file_path, mode='r')
            
            # 2. 读取并缓存头部信息
            self.header = self.dcb_file.header
            self.frame_count = self.header['frame_count']
            
            # 3. 读取并缓存元数据
            self.dicom_meta = self.dcb_file.read_meta()
            self.pixel_header = self.dcb_file.read_pixel_header()
            self.space = self.dcb_file.read_space()
            
            # 4. 获取 transfer syntax UID（从文件类型直接获取）
            self.transfer_syntax_uid = self.dcb_file.get_transfer_syntax_uid()
            if not self.transfer_syntax_uid:
                # 如果文件类型没有定义 transfer syntax，使用默认的未压缩格式
                self.transfer_syntax_uid = '1.2.840.10008.1.2.1'  # Explicit VR Little Endian
            
            # 5. 打开文件句柄用于读取帧数据
            self.file_handle = open(self.file_path, 'rb')
            
            # 6. 读取所有帧的 offset 和 length
            self._read_frame_indices()
            
        except Exception as e:
            self.close()
            raise RuntimeError(f"Failed to open and parse DCB file: {e}")
    
    def _read_frame_indices(self):
        """读取所有帧的偏移量和长度信息"""
        self.file_handle.seek(self.header['frame_offsets_offset'])
        
        # 读取 offsets
        for _ in range(self.frame_count):
            offset, = struct.unpack('<Q', self.file_handle.read(8))
            self.frame_offsets.append(offset)
        
        # 读取 lengths
        self.file_handle.seek(self.header['frame_lengths_offset'])
        for _ in range(self.frame_count):
            length, = struct.unpack('<Q', self.file_handle.read(8))
            self.frame_lengths.append(length)
    
    def get_dicom_for_frame(self, frame_index: int) -> bytes:
        """
        获取指定帧的 DICOM 数据
        
        Args:
            frame_index: 帧索引（0-based）
            
        Returns:
            bytes: 完整的 DICOM 文件数据
            
        Raises:
            IndexError: 如果 frame_index 超出范围
            RuntimeError: 如果读取失败
        """
        # 验证索引
        if not 0 <= frame_index < self.frame_count:
            raise IndexError(f"Frame index {frame_index} out of range [0, {self.frame_count})")
        
        try:
            # 1. 读取该帧的编码数据
            encoded_pixel_data = self._read_encoded_frame(frame_index)
            
            # 2. 生成该帧的 DICOM Dataset
            ds = self._create_dicom_dataset(frame_index, encoded_pixel_data)
            
            # 3. 序列化为 DICOM 文件格式
            return self._serialize_to_dicom_bytes(ds)
            
        except Exception as e:
            raise RuntimeError(f"Failed to create DICOM for frame {frame_index}: {e}")
    
    def _read_encoded_frame(self, frame_index: int) -> bytes:
        """直接读取指定帧的编码数据"""
        offset = self.frame_offsets[frame_index]
        length = self.frame_lengths[frame_index]
        
        self.file_handle.seek(offset)
        return self.file_handle.read(length)
    
    def _create_dicom_dataset(self, frame_index: int, encoded_data: bytes) -> Dataset:
        """快速创建 DICOM Dataset"""
        # 1. 从缓存的 DicomMeta 获取该帧的元数据
        if self.dicom_meta:
            frame_meta_dict = self.dicom_meta.index(frame_index)
        else:
            frame_meta_dict = {}
        
        # 2. 创建 Dataset
        ds = Dataset.from_json(frame_meta_dict)
        
        # 3. 创建并设置文件元信息
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = ds.get('SOPClassUID', '1.2.840.10008.5.1.4.1.1.2')
        file_meta.MediaStorageSOPInstanceUID = ds.get('SOPInstanceUID', generate_uid())
        file_meta.TransferSyntaxUID = self.transfer_syntax_uid
        file_meta.ImplementationClassUID = generate_uid()
        
        ds.file_meta = file_meta
        
        # 4. 确保必要的 SOP 信息
        if not hasattr(ds, 'SOPClassUID'):
            ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        if not hasattr(ds, 'SOPInstanceUID'):
            ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        
        # 5. 设置像素相关属性
        if self.pixel_header:
            ds.RescaleSlope = self.pixel_header.RESCALE_SLOPE
            ds.RescaleIntercept = self.pixel_header.RESCALE_INTERCEPT
        
        # 6. 设置像素数据（使用 encapsulated format for compressed data）
        # 压缩数据需要封装
        ds.PixelData = encapsulate([encoded_data])
        
        return ds
    
    def _serialize_to_dicom_bytes(self, ds: Dataset) -> bytes:
        """将 Dataset 序列化为 DICOM 文件字节流"""
        # 使用 BytesIO 在内存中创建 DICOM 文件
        buffer = io.BytesIO()
        save_dicom(ds, buffer)
        buffer.seek(0)
        return buffer.read()
    
    def get_frame_count(self) -> int:
        """获取总帧数"""
        return self.frame_count
    
    def get_metadata(self) -> Dict[str, Any]:
        """获取缓存的元数据信息"""
        return {
            'frame_count': self.frame_count,
            'pixel_header': self.pixel_header.to_dict() if self.pixel_header else {},
            'has_dicom_meta': self.dicom_meta is not None,
            'has_space': self.space is not None,
            'transfer_syntax': self.transfer_syntax_uid,
            'file_type': self.dcb_file.__class__.__name__,
        }
    
    def close(self):
        """关闭文件句柄"""
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
    
    def __enter__(self):
        """支持 with 语句"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出 with 语句时自动关闭"""
        self.close()
    
    def __del__(self):
        """析构时确保文件关闭"""
        self.close() 