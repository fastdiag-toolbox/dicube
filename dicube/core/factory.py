"""Factory pattern implementation for DicomCubeImage creation.

This module provides abstract interfaces and concrete implementations for creating
DicomCubeImage instances without circular dependencies.
"""

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

import numpy as np
from spacetransformer import Space

from .pixel_header import PixelDataHeader
from ..validation import (
    validate_not_none,
    validate_parameter_type,
    validate_array_shape
)
from ..exceptions import (
    DataConsistencyError,
    MetaDataError
)

if TYPE_CHECKING:
    from .image import DicomCubeImage
    from ..dicom import DicomMeta


class ImageFactory(ABC):
    """Abstract factory interface for creating DicomCubeImage instances.
    
    This factory pattern eliminates circular dependencies by providing
    a clean interface for image creation without direct imports.
    """
    
    @abstractmethod
    def create_image(
        self,
        raw_image: np.ndarray,
        pixel_header: PixelDataHeader,
        dicom_meta: Optional["DicomMeta"] = None,
        space: Optional[Space] = None,
        dicom_status: str = "consistent",
    ) -> "DicomCubeImage":
        """Create a DicomCubeImage instance.
        
        Args:
            raw_image (np.ndarray): Raw image data array.
            pixel_header (PixelDataHeader): Pixel data header information.
            dicom_meta (DicomMeta, optional): DICOM metadata. Defaults to None.
            space (Space, optional): Spatial information. Defaults to None.
            dicom_status (str): DICOM status string. Defaults to "consistent".
            
        Returns:
            DicomCubeImage: The created image instance.
        """
        pass


class DicomCubeImageFactory(ImageFactory):
    """Concrete factory implementation for creating DicomCubeImage instances.
    
    This factory handles the actual instantiation of DicomCubeImage objects
    while avoiding circular import issues through delayed imports.
    """
    
    def create_image(
        self,
        raw_image: np.ndarray,
        pixel_header: PixelDataHeader,
        dicom_meta: Optional["DicomMeta"] = None,
        space: Optional[Space] = None,
        dicom_status: str = "consistent",
    ) -> "DicomCubeImage":
        """Create a DicomCubeImage instance.
        
        Args:
            raw_image (np.ndarray): Raw image data array.
            pixel_header (PixelDataHeader): Pixel data header information.
            dicom_meta (DicomMeta, optional): DICOM metadata. Defaults to None.
            space (Space, optional): Spatial information. Defaults to None.
            dicom_status (str): DICOM status string. Defaults to "consistent".
            
        Returns:
            DicomCubeImage: The created image instance.
        """
        # Validate required parameters before creating the image
        validate_not_none(raw_image, "raw_image", "DicomCubeImageFactory.create_image", DataConsistencyError)
        validate_not_none(pixel_header, "pixel_header", "DicomCubeImageFactory.create_image", DataConsistencyError)
        validate_array_shape(raw_image, min_dims=2, name="raw_image", context="DicomCubeImageFactory.create_image")
        validate_parameter_type(pixel_header, PixelDataHeader, "pixel_header", "DicomCubeImageFactory.create_image", DataConsistencyError)
        
        # Validate optional parameters if provided
        if dicom_meta is not None:
            # Import here to avoid circular dependency for validation
            from ..dicom import DicomMeta
            validate_parameter_type(dicom_meta, DicomMeta, "dicom_meta", "DicomCubeImageFactory.create_image", MetaDataError)
        if space is not None:
            validate_parameter_type(space, Space, "space", "DicomCubeImageFactory.create_image", DataConsistencyError)
        
        try:
            # Import here to avoid circular dependency
            from .image import DicomCubeImage
            
            return DicomCubeImage(
                raw_image=raw_image,
                pixel_header=pixel_header,
                dicom_meta=dicom_meta,
                space=space,
                dicom_status=dicom_status,
            )
        except Exception as e:
            if isinstance(e, (DataConsistencyError, MetaDataError)):
                raise
            raise DataConsistencyError(
                f"Failed to create DicomCubeImage: {str(e)}",
                context="DicomCubeImageFactory.create_image",
                details={
                    "raw_image_shape": raw_image.shape if hasattr(raw_image, 'shape') else None,
                    "pixel_header_type": type(pixel_header).__name__,
                    "dicom_meta_provided": dicom_meta is not None,
                    "space_provided": space is not None
                },
                suggestion="Verify all parameters are valid and compatible"
            ) from e


# Default factory instance
_default_factory: Optional[ImageFactory] = None


def get_default_factory() -> ImageFactory:
    """Get the default image factory instance.
    
    Returns:
        ImageFactory: The default factory instance.
    """
    global _default_factory
    if _default_factory is None:
        _default_factory = DicomCubeImageFactory()
    return _default_factory


def set_default_factory(factory: ImageFactory) -> None:
    """Set the default image factory instance.
    
    Args:
        factory (ImageFactory): The factory instance to set as default.
    """
    global _default_factory
    _default_factory = factory