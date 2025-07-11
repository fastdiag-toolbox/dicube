from .core.image import DicomCubeImage
from .core.io import DicomCubeImageIO
from .dicom import (
    CommonTags,
    DicomMeta,
    DicomStatus,
    SortMethod,
    get_dicom_status,
    read_dicom_dir,
)

__version__ = "1.0.0"

# 顶层便利方法
def load(filename: str, num_threads: int = 4, **kwargs) -> DicomCubeImage:
    """
    从文件加载DicomCubeImage。
    
    Args:
        filename: 输入文件路径
        num_threads: 并行解码线程数
        **kwargs: 其他参数传递给底层reader
        
    Returns:
        DicomCubeImage: 从文件加载的对象
    """
    return DicomCubeImageIO.load(filename, num_threads, **kwargs)


def save(
    image: DicomCubeImage,
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
    return DicomCubeImageIO.save(image, filename, file_type, num_threads, **kwargs)


def load_from_dicom_folder(
    folder_path: str,
    sort_method: SortMethod = SortMethod.INSTANCE_NUMBER_ASC,
    **kwargs
) -> DicomCubeImage:
    """
    从DICOM文件夹加载DicomCubeImage。
    
    Args:
        folder_path: DICOM文件夹路径
        sort_method: DICOM文件排序方法
        **kwargs: 其他参数
        
    Returns:
        DicomCubeImage: 从DICOM文件夹创建的对象
    """
    return DicomCubeImageIO.load_from_dicom_folder(folder_path, sort_method, **kwargs)


def load_from_nifti(nii_path: str, **kwargs) -> DicomCubeImage:
    """
    从NIfTI文件加载DicomCubeImage。
    
    Args:
        nii_path: NIfTI文件路径
        **kwargs: 其他参数
        
    Returns:
        DicomCubeImage: 从NIfTI文件创建的对象
    """
    return DicomCubeImageIO.load_from_nifti(nii_path, **kwargs)


def save_to_dicom_folder(
    image: DicomCubeImage,
    output_dir: str,
    filenames: list = None,
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
    """
    return DicomCubeImageIO.save_to_dicom_folder(
        image, output_dir, filenames, use_j2k, lossless, **compress_kwargs
    )


__all__ = [
    "DicomCubeImage",
    "DicomMeta",
    "read_dicom_dir",
    "DicomStatus",
    "get_dicom_status",
    "CommonTags",
    "SortMethod",
    # 顶层便利方法
    "load",
    "save",
    "load_from_dicom_folder",
    "load_from_nifti",
    "save_to_dicom_folder",
    # IO类（如果需要直接使用）
    "DicomCubeImageIO",
] 