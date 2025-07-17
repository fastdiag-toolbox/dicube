"""DiCube: Python library for efficient storage and processing of 3D medical images.

DiCube provides functionality for working with DICOM image data while preserving
complete metadata. It offers efficient storage formats, image processing capabilities,
and interoperability with various medical image formats.

Main functionality:
- Load/save 3D medical images with complete DICOM metadata
- Efficient binary storage format with multiple compression options
- Spatial transformation and orientation handling
- Conversion between different medical image formats

Example:
    >>> import dicube
    >>> # Load from DICOM folder
    >>> image = dicube.load_from_dicom_folder("path/to/dicom_folder")
    >>> # Save to DCB file
    >>> dicube.save(image, "output.dcb")
    >>> # Access the data
    >>> pixel_data = image.get_fdata()
"""

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

# Top-level convenience methods
def load(file_path: str, num_threads: int = 4, **kwargs) -> DicomCubeImage:
    """Load a DicomCubeImage from a file.
    
    Args:
        file_path (str): Path to the input file.
        num_threads (int): Number of parallel decoding threads. Defaults to 4.
        **kwargs: Additional parameters passed to the underlying reader.
    
    Returns:
        DicomCubeImage: The loaded image object.
    """
    return DicomCubeImageIO.load(file_path, num_threads, **kwargs)


def save(
    image: DicomCubeImage,
    file_path: str,
    file_type: str = "s",
    num_threads: int = 4,
    **kwargs
) -> None:
    """Save a DicomCubeImage to a file.
    
    Args:
        image (DicomCubeImage): The image object to save.
        file_path (str): Output file path.
        file_type (str): File type, "s" (speed priority), "a" (compression priority), 
                        or "l" (lossy compression). Defaults to "s".
        num_threads (int): Number of parallel encoding threads. Defaults to 4.
        **kwargs: Additional parameters passed to the underlying writer.
    """
    return DicomCubeImageIO.save(image, file_path, file_type, num_threads, **kwargs)


def load_from_dicom_folder(
    folder_path: str,
    sort_method: SortMethod = SortMethod.INSTANCE_NUMBER_ASC,
    **kwargs
) -> DicomCubeImage:
    """Load a DicomCubeImage from a DICOM folder.
    
    Args:
        folder_path (str): Path to the DICOM folder.
        sort_method (SortMethod): Method to sort DICOM files. 
                                 Defaults to SortMethod.INSTANCE_NUMBER_ASC.
        **kwargs: Additional parameters.
    
    Returns:
        DicomCubeImage: The loaded image object.
    """
    return DicomCubeImageIO.load_from_dicom_folder(folder_path, sort_method, **kwargs)


def load_from_nifti(file_path: str, **kwargs) -> DicomCubeImage:
    """Load a DicomCubeImage from a NIfTI file.
    
    Args:
        file_path (str): Path to the NIfTI file.
        **kwargs: Additional parameters.
    
    Returns:
        DicomCubeImage: The loaded image object.
    """
    return DicomCubeImageIO.load_from_nifti(file_path, **kwargs)


def save_to_dicom_folder(
    image: DicomCubeImage,
    folder_path: str,
    **kwargs
) -> None:
    """Save a DicomCubeImage as a DICOM folder.
    
    Args:
        image (DicomCubeImage): The image object to save.
        folder_path (str): Output directory path.
        **kwargs: Additional parameters.
    """
    return DicomCubeImageIO.save_to_dicom_folder(image, folder_path)


__all__ = [
    "DicomCubeImage",
    "DicomMeta",
    "read_dicom_dir",
    "DicomStatus",
    "get_dicom_status",
    "CommonTags",
    "SortMethod",
    # Top-level convenience methods
    "load",
    "save",
    "load_from_dicom_folder",
    "load_from_nifti",
    "save_to_dicom_folder",
    # IO class (for direct use if needed)
    "DicomCubeImageIO",
] 