"""Core module for DiCube library.

This module provides the main interfaces and core functionality for working with
medical images in the DiCube format, including image representation, metadata handling,
and file I/O operations.

Classes:
    DicomCubeImage: Main class for representing DICOM image data with metadata.
    PixelDataHeader: Header class for storing pixel data information.
    DicomCubeImageIO: Static I/O utility class for file operations.
"""

from .image import DicomCubeImage
from .pixel_header import PixelDataHeader
from .io import DicomCubeImageIO

__all__ = [
    "DicomCubeImage",
    "PixelDataHeader",
    "DicomCubeImageIO",
] 