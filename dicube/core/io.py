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
from .factory import ImageFactory, get_default_factory
import os


class DicomCubeImageIO:
    """Static I/O utility class responsible for DicomCubeImage file operations.
    
    Responsibilities:
    - Provides unified file I/O interface
    - Automatically detects file formats
    - Handles conversion between various file formats
    """
    
    @staticmethod
    def _validate_not_none(value, name: str, context: str = ""):
        """Validate that a value is not None.
        
        Args:
            value: The value to validate.
            name (str): Name of the parameter.
            context (str): Context for the error message.
            
        Raises:
            ValueError: If the value is None.
        """
        if value is None:
            raise ValueError(f"{name} cannot be None{f' in {context}' if context else ''}")
        return value
    
    @staticmethod
    def _validate_file_exists(file_path: str, context: str = ""):
        """Validate that a file exists.
        
        Args:
            file_path (str): Path to the file.
            context (str): Context for the error message.
            
        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}{f' in {context}' if context else ''}")
        return file_path
    
    @staticmethod
    def _validate_folder_exists(folder_path: str, context: str = ""):
        """Validate that a folder exists.
        
        Args:
            folder_path (str): Path to the folder.
            context (str): Context for the error message.
            
        Raises:
            FileNotFoundError: If the folder does not exist.
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}{f' in {context}' if context else ''}")
        if not os.path.isdir(folder_path):
            raise ValueError(f"Path is not a directory: {folder_path}{f' in {context}' if context else ''}")
        return folder_path
    
    @staticmethod
    def save(
        image: "DicomCubeImage",
        file_path: str,
        file_type: str = "s",
        num_threads: int = 4,
    ) -> None:
        """Save DicomCubeImage to a file.
        
        Args:
            image (DicomCubeImage): The DicomCubeImage object to save.
            file_path (str): Output file path.
            file_type (str): File type, "s" (speed priority), "a" (compression priority), 
                             or "l" (lossy compression). Defaults to "s".
            num_threads (int): Number of parallel encoding threads. Defaults to 4.
            
        Raises:
            ValueError: If the file_type is not supported.
        """
        # Validate required parameters
        DicomCubeImageIO._validate_not_none(image, "image", "save operation")
        DicomCubeImageIO._validate_not_none(file_path, "file_path", "save operation")
        
        # Choose appropriate writer based on file type
        if file_type == "s":
            writer = DcbSFile(file_path, mode="w")
        elif file_type == "a":
            writer = DcbAFile(file_path, mode="w")
        elif file_type == "l":
            writer = DcbLFile(file_path, mode="w")
        else:
            raise ValueError(f"Unsupported file type: {file_type}, must be one of 's', 'a', 'l'")
        
        # Write to file
        writer.write(
            images=image.raw_image,
            pixel_header=image.pixel_header,
            dicom_meta=image.dicom_meta,
            space=image.space,
            num_threads=num_threads,
            dicom_status=image.dicom_status
        )
    
    @staticmethod
    def load(file_path: str, num_threads: int = 4, **kwargs) -> 'DicomCubeImage':
        """Load DicomCubeImage from a file.
        
        Args:
            file_path (str): Input file path.
            num_threads (int): Number of parallel decoding threads. Defaults to 4.
            **kwargs: Additional parameters passed to the underlying reader.
            
        Returns:
            DicomCubeImage: The loaded object from the file.
            
        Raises:
            ValueError: When the file format is not supported.
        """
        # Validate required parameters
        DicomCubeImageIO._validate_not_none(file_path, "file_path", "load operation")
        DicomCubeImageIO._validate_file_exists(file_path, "load operation")
        
        # Use factory pattern to avoid circular dependency
        factory = get_default_factory()
        
        # Read file header to determine format
        header_size = struct.calcsize(DcbFile.HEADER_STRUCT)
        with open(file_path, "rb") as f:
            header_data = f.read(header_size)
        magic = struct.unpack(DcbFile.HEADER_STRUCT, header_data)[0]
        
        # Choose appropriate reader based on magic number
        if magic == DcbAFile.MAGIC:
            reader = DcbAFile(file_path, mode="r")
        elif magic == DcbSFile.MAGIC:
            reader = DcbSFile(file_path, mode="r")
        else:
            raise ValueError(f"Unsupported file format, magic number: {magic}")
        
        # Read file contents
        dicom_meta = reader.read_meta()
        space = reader.read_space()
        pixel_header = reader.read_pixel_header()
        dicom_status = reader.read_dicom_status()
        
        images = reader.read_images(num_threads=num_threads)
        if isinstance(images, list):
            # Convert list to ndarray if needed
            images = np.stack(images)
        
        
        return factory.create_image(
            raw_image=images,
            pixel_header=pixel_header,
            dicom_meta=dicom_meta,
            space=space,
            dicom_status=dicom_status,
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
        # Validate required parameters
        DicomCubeImageIO._validate_not_none(folder_path, "folder_path", "load_from_dicom_folder operation")
        DicomCubeImageIO._validate_folder_exists(folder_path, "load_from_dicom_folder operation")
        
        # Use factory pattern to avoid circular dependency
        factory = get_default_factory()
        
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
        slope = meta.get_shared_value(CommonTags.RescaleSlope)
        intercept = meta.get_shared_value(CommonTags.RescaleIntercept)
        wind_center = meta.get_shared_value(CommonTags.WindowCenter)
        wind_width = meta.get_shared_value(CommonTags.WindowWidth)
        
        # Create pixel_header
        pixel_header = PixelDataHeader(
            RESCALE_SLOPE=float(slope) if slope is not None else 1.0,
            RESCALE_INTERCEPT=float(intercept) if intercept is not None else 0.0,
            ORIGINAL_PIXEL_DTYPE=str(images[0].dtype),
            PIXEL_DTYPE=str(images[0].dtype),
            WINDOW_CENTER=float(wind_center) if wind_center is not None else None,
            WINDOW_WIDTH=float(wind_width) if wind_width is not None else None,
        )
        
        # Validate PixelDataHeader initialization success
        if pixel_header is None:
            raise RuntimeError("PixelDataHeader initialization failed in load_from_dicom_folder operation")
        
        return factory.create_image(
            raw_image=np.array(images),
            pixel_header=pixel_header,
            dicom_meta=meta,
            space=space,
            dicom_status=status
        )
    
    @staticmethod
    def load_from_nifti(file_path: str, **kwargs) -> 'DicomCubeImage':
        """Load DicomCubeImage from a NIfTI file.
        
        Args:
            file_path (str): Path to the NIfTI file.
            **kwargs: Additional parameters.
            
        Returns:
            DicomCubeImage: The object created from the NIfTI file.
            
        Raises:
            ImportError: When nibabel is not installed.
        """
        # Validate required parameters
        DicomCubeImageIO._validate_not_none(file_path, "file_path", "load_from_nifti operation")
        DicomCubeImageIO._validate_file_exists(file_path, "load_from_nifti operation")
        
        # Use factory pattern to avoid circular dependency
        factory = get_default_factory()
        
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError("nibabel is required to read NIfTI files")
        
        nii = nib.load(file_path)
        space = get_space_from_nifti(nii)
        
        # Fix numpy array warning
        raw_image, header = derive_pixel_header_from_array(
            np.asarray(nii.dataobj, dtype=nii.dataobj.dtype)
        )
        
        return factory.create_image(raw_image, header, space=space)
    
    @staticmethod
    def save_to_dicom_folder(
        image: 'DicomCubeImage',
        folder_path: str,
    ) -> None:
        """Save DicomCubeImage as a DICOM folder.
        
        Args:
            image (DicomCubeImage): The DicomCubeImage object to save.
            folder_path (str): Output directory path.
        """
        # Validate required parameters
        DicomCubeImageIO._validate_not_none(image, "image", "save_to_dicom_folder operation")
        DicomCubeImageIO._validate_not_none(folder_path, "folder_path", "save_to_dicom_folder operation")
        
        if image.dicom_meta is None:
            warnings.warn("dicom_meta is None, initializing with default values")
            image.init_meta()
        
        save_to_dicom_folder(
            raw_images=image.raw_image,
            dicom_meta=image.dicom_meta,
            pixel_header=image.pixel_header,
            output_dir=folder_path,
        ) 