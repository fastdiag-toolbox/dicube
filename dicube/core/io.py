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
    """Static I/O utility class responsible for DicomCubeImage file operations.
    
    Responsibilities:
    - Provides unified file I/O interface
    - Automatically detects file formats
    - Handles conversion between various file formats
    """
    
    @staticmethod
    def save(
        image: 'DicomCubeImage',
        filename: str,
        file_type: str = "s",
        num_threads: int = 4,
        **kwargs
    ) -> None:
        """Save DicomCubeImage to a file.
        
        Args:
            image (DicomCubeImage): The DicomCubeImage object to save.
            filename (str): Output file path.
            file_type (str): File type, "s" (speed priority), "a" (compression priority), 
                             or "l" (lossy compression). Defaults to "s".
            num_threads (int): Number of parallel encoding threads. Defaults to 4.
            **kwargs: Additional parameters passed to the underlying writer.
            
        Raises:
            ValueError: If the file_type is not supported.
        """
        # Choose appropriate writer based on file type
        if file_type == "s":
            writer = DcbSFile(filename, mode="w")
        elif file_type == "a":
            writer = DcbAFile(filename, mode="w")
        elif file_type == "l":
            writer = DcbLFile(filename, mode="w")
        else:
            raise ValueError(f"Unsupported file type: {file_type}, must be one of 's', 'a', 'l'")
        
        # Write to file
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
        """Load DicomCubeImage from a file.
        
        Args:
            filename (str): Input file path.
            num_threads (int): Number of parallel decoding threads. Defaults to 4.
            **kwargs: Additional parameters passed to the underlying reader.
            
        Returns:
            DicomCubeImage: The loaded object from the file.
            
        Raises:
            ValueError: When the file format is not supported.
        """
        # Delayed import to avoid circular dependency
        from .image import DicomCubeImage
        
        # Read file header to determine format
        header_size = struct.calcsize(DcbFile.HEADER_STRUCT)
        with open(filename, "rb") as f:
            header_data = f.read(header_size)
        magic = struct.unpack(DcbFile.HEADER_STRUCT, header_data)[0]
        
        # Choose appropriate reader based on magic number
        if magic == DcbAFile.MAGIC:
            reader = DcbAFile(filename, mode="r")
        elif magic == DcbSFile.MAGIC:
            reader = DcbSFile(filename, mode="r")
        else:
            raise ValueError(f"Unsupported file format, magic number: {magic}")
        
        # Read file contents
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
        """Load DicomCubeImage from a DICOM folder.
        
        Args:
            folder_path (str): Path to the DICOM folder.
            sort_method (SortMethod): Method to sort DICOM files. 
                                      Defaults to SortMethod.INSTANCE_NUMBER_ASC.
            **kwargs: Additional parameters.
            
        Returns:
            DicomCubeImage: The object created from the DICOM folder.
            
        Raises:
            ValueError: When the DICOM status is not supported.
        """
        # Delayed import to avoid circular dependency
        from .image import DicomCubeImage
        
        # Read DICOM folder
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
            raise ValueError(f"Unsupported DICOM status: {status}")
        
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
            warnings.warn(f"DICOM status: {status}, cannot calculate space information")
            space = None
        else:
            if get_space_from_DicomMeta is not None:
                space = get_space_from_DicomMeta(meta, axis_order="zyx")
            else:
                space = None
        
        # Get rescale parameters
        slope = meta.get(CommonTags.RESCALE_SLOPE, force_shared=True)[0]
        intercept = meta.get(CommonTags.RESCALE_INTERCEPT, force_shared=True)[0]
        wind_center = meta.get(CommonTags.WINDOW_CENTER, force_shared=True)
        wind_width = meta.get(CommonTags.WINDOW_WIDTH, force_shared=True)
        
        # Create pixel_header
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
        """Load DicomCubeImage from a NIfTI file.
        
        Args:
            nii_path (str): Path to the NIfTI file.
            **kwargs: Additional parameters.
            
        Returns:
            DicomCubeImage: The object created from the NIfTI file.
            
        Raises:
            ImportError: When nibabel is not installed.
        """
        # Delayed import to avoid circular dependency
        from .image import DicomCubeImage
        
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError("nibabel is required to read NIfTI files")
        
        nii = nib.load(nii_path)
        space = get_space_from_nifti(nii)
        
        # Fix numpy array warning
        raw_image, header = derive_pixel_header_from_array(
            np.asarray(nii.dataobj, dtype=nii.dataobj.dtype)
        )
        
        return DicomCubeImage(raw_image, header, space=space)
    
    @staticmethod
    def save_to_dicom_folder(
        image: 'DicomCubeImage',
        output_dir: str,
    ) -> None:
        """Save DicomCubeImage as a DICOM folder.
        
        Args:
            image (DicomCubeImage): The DicomCubeImage object to save.
            output_dir (str): Output directory path.
        """
        if image.dicom_meta is None:
            warnings.warn("dicom_meta is None, initializing with default values")
            image.init_meta()
        
        save_to_dicom_folder(
            raw_images=image.raw_image,
            dicom_meta=image.dicom_meta,
            pixel_header=image.pixel_header,
            output_dir=output_dir,
        ) 