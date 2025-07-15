import struct
import warnings
from typing import Optional, Union

import numpy as np
from spacetransformer import Space, get_space_from_nifti

from ..dicom import (
    CommonTags,
    DicomMeta,
    DicomStatus,
    SortMethod,
    get_dicom_status,
    read_dicom_dir,
    get_space_from_DicomMeta,
)
from ..dicom.dicom_io import save_to_dicom_folder
from ..storage.dcb_file import DcbSFile, DcbFile, DcbAFile, DcbLFile
from ..storage.pixel_utils import derive_pixel_header_from_array
from .pixel_header import PixelDataHeader


class DicomCubeImageIO:
    """
    静态I/O工具类，负责DicomCubeImage的文件读写操作。
    
    职责：
    - 提供统一的文件I/O接口
    - 自动检测文件格式
    - 处理各种文件格式的转换
    """
    
    @staticmethod
    def save(
        image: 'DicomCubeImage',
        filename: str,
        file_type: str = "s",
        num_threads: int = 4,
        **kwargs
    ) -> None:
        """
        保存DicomCubeImage到文件。
        
        Args:
            image: 要保存的DicomCubeImage对象
            filename: 输出文件路径
            file_type: 文件类型，"s"(速度优先), "a"(压缩优先), "l"(有损压缩)
            num_threads: 并行编码线程数
            **kwargs: 其他参数传递给底层writer
        """
        # 根据文件类型选择合适的writer
        if file_type == "s":
            writer = DcbSFile(filename, mode="w")
        elif file_type == "a":
            writer = DcbAFile(filename, mode="w")
        elif file_type == "l":
            writer = DcbLFile(filename, mode="w")
        else:
            raise ValueError(f"不支持的文件类型: {file_type}，必须是 's', 'a', 'l' 之一")
        
        # 写入文件
        writer.write(
            images=image.raw_image,
            pixel_header=image.pixel_header,
            dicom_meta=image.dicom_meta,
            space=image.space,
            num_threads=num_threads,
            **kwargs
        )
    
    @staticmethod
    def load(filename: str, num_threads: int = 4, **kwargs) -> 'DicomCubeImage':
        """
        从文件加载DicomCubeImage。
        
        Args:
            filename: 输入文件路径
            num_threads: 并行解码线程数
            **kwargs: 其他参数传递给底层reader
            
        Returns:
            DicomCubeImage: 从文件加载的对象
            
        Raises:
            ValueError: 当文件格式不支持时
        """
        # 延迟导入避免循环依赖
        from .image import DicomCubeImage
        
        # 读取文件头部判断格式
        header_size = struct.calcsize(DcbFile.HEADER_STRUCT)
        with open(filename, "rb") as f:
            header_data = f.read(header_size)
        magic = struct.unpack(DcbFile.HEADER_STRUCT, header_data)[0]
        
        # 根据魔数选择合适的reader
        if magic == DcbAFile.MAGIC:
            reader = DcbAFile(filename, mode="r")
        elif magic == DcbSFile.MAGIC:
            reader = DcbSFile(filename, mode="r")
        else:
            raise ValueError(f"不支持的文件格式，魔数: {magic}")
        
        # 读取文件内容
        dicom_meta = reader.read_meta()
        space = reader.read_space()
        pixel_header = reader.read_pixel_header()
        
        images = reader.read_images(num_threads=num_threads)
        if isinstance(images, list):
            if len(images) == 0:
                raw_image = np.array([], dtype=pixel_header.ORIGINAL_PIXEL_DTYPE)
            else:
                raw_image = np.stack(images, axis=0)
        else:
            raw_image = images
        
        return DicomCubeImage(
            raw_image=raw_image,
            pixel_header=pixel_header,
            dicom_meta=dicom_meta,
            space=space,
        )
    
    @staticmethod
    def load_from_dicom_folder(
        folder_path: str,
        sort_method: SortMethod = SortMethod.INSTANCE_NUMBER_ASC,
        **kwargs
    ) -> 'DicomCubeImage':
        """
        从DICOM文件夹加载DicomCubeImage。
        
        Args:
            folder_path: DICOM文件夹路径
            sort_method: DICOM文件排序方法
            **kwargs: 其他参数
            
        Returns:
            DicomCubeImage: 从DICOM文件夹创建的对象
            
        Raises:
            ValueError: 当DICOM状态不支持时
        """
        # 延迟导入避免循环依赖
        from .image import DicomCubeImage
        
        # 读取DICOM文件夹
        meta, datasets = read_dicom_dir(folder_path, sort_method=sort_method)
        images = [d.pixel_array for d in datasets]
        status = get_dicom_status(meta)
        
        if status in (
            DicomStatus.NON_UNIFORM_RESCALE_FACTOR,
            DicomStatus.MISSING_DTYPE,
            DicomStatus.NON_UNIFORM_DTYPE,
            DicomStatus.MISSING_SHAPE,
            DicomStatus.INCONSISTENT,
        ):
            raise ValueError(f"不支持的DICOM状态: {status}")
        
        if status in (
            DicomStatus.MISSING_SPACING,
            DicomStatus.NON_UNIFORM_SPACING,
            DicomStatus.MISSING_ORIENTATION,
            DicomStatus.NON_UNIFORM_ORIENTATION,
            DicomStatus.MISSING_LOCATION,
            DicomStatus.REVERSED_LOCATION,
            DicomStatus.DWELLING_LOCATION,
            DicomStatus.GAP_LOCATION,
        ):
            warnings.warn(f"DICOM状态: {status}，无法计算space信息")
            space = None
        else:
            if get_space_from_DicomMeta is not None:
                space = get_space_from_DicomMeta(meta, axis_order="zyx")
            else:
                space = None
        
        # 获取rescale参数
        slope = meta.get(CommonTags.RESCALE_SLOPE, force_shared=True)[0]
        intercept = meta.get(CommonTags.RESCALE_INTERCEPT, force_shared=True)[0]
        wind_center = meta.get(CommonTags.WINDOW_CENTER, force_shared=True)
        wind_width = meta.get(CommonTags.WINDOW_WIDTH, force_shared=True)
        
        # 创建pixel_header
        pixel_header = PixelDataHeader(
            RESCALE_SLOPE=float(slope) if slope is not None else 1.0,
            RESCALE_INTERCEPT=float(intercept) if intercept is not None else 0.0,
            ORIGINAL_PIXEL_DTYPE=str(images[0].dtype),
            PIXEL_DTYPE=str(images[0].dtype),
            WINDOW_CENTER=float(wind_center[0]) if wind_center is not None else None,
            WINDOW_WIDTH=float(wind_width[0]) if wind_width is not None else None,
        )
        
        return DicomCubeImage(np.array(images), pixel_header, meta, space)
    
    @staticmethod
    def load_from_nifti(nii_path: str, **kwargs) -> 'DicomCubeImage':
        """
        从NIfTI文件加载DicomCubeImage。
        
        Args:
            nii_path: NIfTI文件路径
            **kwargs: 其他参数
            
        Returns:
            DicomCubeImage: 从NIfTI文件创建的对象
            
        Raises:
            ImportError: 当nibabel未安装时
        """
        # 延迟导入避免循环依赖
        from .image import DicomCubeImage
        
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError("需要安装nibabel才能读取NIfTI文件")
        
        nii = nib.load(nii_path)
        space = get_space_from_nifti(nii)
        
        # 修复numpy数组警告
        raw_image, header = derive_pixel_header_from_array(
            np.asarray(nii.dataobj, dtype=nii.dataobj.dtype)
        )
        
        return DicomCubeImage(raw_image, header, space=space)
    
    @staticmethod
    def save_to_dicom_folder(
        image: 'DicomCubeImage',
        output_dir: str,
        filenames: Optional[list] = None,
        use_j2k: bool = False,
        lossless: bool = True,
        **compress_kwargs
    ) -> None:
        """
        保存DicomCubeImage为DICOM文件夹。
        
        Args:
            image: 要保存的DicomCubeImage对象
            output_dir: 输出目录路径
            filenames: 可选的文件名列表
            use_j2k: 是否使用JPEG 2000压缩
            lossless: 是否使用无损压缩
            **compress_kwargs: 压缩相关参数
            
        Raises:
            ValueError: 当dicom_meta不存在时
        """
        if image.dicom_meta is None:
            warnings.warn("dicom_meta为None，使用默认值初始化")
            image.init_meta()
        
        save_to_dicom_folder(
            raw_images=image.raw_image,
            dicom_meta=image.dicom_meta,
            pixel_header=image.pixel_header,
            output_dir=output_dir,
            filenames=filenames,
            use_j2k=use_j2k,
            lossless=lossless,
            **compress_kwargs,
        ) 