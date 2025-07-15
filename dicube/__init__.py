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

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.1.0"

# 顶层便利方法
def load(filename: str, num_threads: int = 4, **kwargs) -> DicomCubeImage:
    """从文件加载DicomCubeImage"""
    return DicomCubeImageIO.load(filename, num_threads, **kwargs)


def save(
    image: DicomCubeImage,
    filename: str,
    file_type: str = "s",
    num_threads: int = 4,
    **kwargs
) -> None:
    """保存DicomCubeImage到文件"""
    return DicomCubeImageIO.save(image, filename, file_type, num_threads, **kwargs)


def load_from_dicom_folder(
    folder_path: str,
    sort_method: SortMethod = SortMethod.INSTANCE_NUMBER_ASC,
    **kwargs
) -> DicomCubeImage:
    """从DICOM文件夹加载DicomCubeImage"""
    return DicomCubeImageIO.load_from_dicom_folder(folder_path, sort_method, **kwargs)


def load_from_nifti(nii_path: str, **kwargs) -> DicomCubeImage:
    """从NIfTI文件加载DicomCubeImage"""
    return DicomCubeImageIO.load_from_nifti(nii_path, **kwargs)


def save_to_dicom_folder(
    image: DicomCubeImage,
    output_dir: str,
    **kwargs
) -> None:
    """保存DicomCubeImage为DICOM文件夹"""
    return DicomCubeImageIO.save_to_dicom_folder(image, output_dir, **kwargs)


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