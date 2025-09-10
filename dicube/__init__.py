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

from importlib.metadata import version as _pkg_version, PackageNotFoundError

try:
    __version__ = _pkg_version("dicube")
except PackageNotFoundError:
    # editable install / source tree
    __version__ = "0.1.0+unknown"

# Default to the number of CPU cores, but cap at 8 threads to avoid excessive resource usage
# Fall back to 4 if cpu_count() returns None (which can happen in some environments)
_num_threads = 4

def get_num_threads() -> int:
    """Get the global number of threads for parallel processing.
    
    Returns:
        int: Current number of threads setting.
    """
    global _num_threads
    return _num_threads

def set_num_threads(num_threads: int) -> None:
    """Set the global number of threads for parallel processing.
    
    Args:
        num_threads (int): Number of threads for parallel processing tasks.
    """
    global _num_threads
    if num_threads < 1:
        raise ValueError("Number of threads must be at least 1")
    _num_threads = num_threads

# Top-level convenience methods
def load(file_path: str, skip_meta: bool = False) -> DicomCubeImage:
    """Load a DicomCubeImage from a file.
    
    Args:
        file_path (str): Path to the input file.
    
    Returns:
        DicomCubeImage: The loaded image object.
    """
    return DicomCubeImageIO.load(file_path, skip_meta)

def load_meta(file_path: str) -> DicomMeta:
    """Load the metadata from a file.
    
    Args:
        file_path (str): Path to the input file.
    """
    return DicomCubeImageIO.load_meta(file_path)

def save(
    image: DicomCubeImage,
    file_path: str,
    file_type: str = "s",
) -> None:
    """Save a DicomCubeImage to a file.
    
    Args:
        image (DicomCubeImage): The image object to save.
        file_path (str): Output file path.
        file_type (str): File type, "s" (speed priority), "a" (compression priority), 
                        or "l" (lossy compression). Defaults to "s".
    """
    return DicomCubeImageIO.save(image, file_path, file_type)


def load_from_dicom_folder(
    folder_path: str,
    sort_method: SortMethod = SortMethod.INSTANCE_NUMBER_ASC,
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
    return DicomCubeImageIO.load_from_dicom_folder(folder_path, sort_method)


def load_from_nifti(file_path: str) -> DicomCubeImage:
    """Load a DicomCubeImage from a NIfTI file.
    
    Args:
        file_path (str): Path to the NIfTI file.
        **kwargs: Additional parameters.
    
    Returns:
        DicomCubeImage: The loaded image object.
    """
    return DicomCubeImageIO.load_from_nifti(file_path)


def save_to_dicom_folder(
    image: DicomCubeImage,
    folder_path: str,
) -> None:
    """Save a DicomCubeImage as a DICOM folder.
    
    Args:
        image (DicomCubeImage): The image object to save.
        folder_path (str): Output directory path.
    """
    return DicomCubeImageIO.save_to_dicom_folder(image, folder_path)


def save_to_nifti(
    image: DicomCubeImage,
    file_path: str,
) -> None:
    """Save a DicomCubeImage as a NIfTI file.
    
    Args:
        image (DicomCubeImage): The image object to save.
        file_path (str): Output file path.
    
    Raises:
        ImportError: When nibabel is not installed.
    """
    return DicomCubeImageIO.save_to_nifti(image, file_path)


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
    "save_to_nifti",
    "set_num_threads",
    "get_num_threads",
    # IO class (for direct use if needed)
    "DicomCubeImageIO",
] 