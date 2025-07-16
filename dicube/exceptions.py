"""Exceptions module for DiCube.

This module defines the exception hierarchy used throughout the DiCube library.
All exceptions inherit from the base DicomCubeError class to allow for
easy catching of all DiCube-related exceptions.
"""


class DicomCubeError(Exception):
    """Base exception class for all DicomCube-related errors.
    
    All other exceptions in the DiCube library inherit from this class,
    allowing applications to catch all DiCube-related exceptions with:
    
    ```python
    try:
        # DiCube operations
    except DicomCubeError:
        # Handle any DiCube error
    ```
    """
    pass


class InvalidCubeFileError(DicomCubeError):
    """Raised when a file is not a valid DicomCube file.
    
    This exception is raised when attempting to load a file that is not
    in the expected DicomCube format. This could be due to incorrect 
    magic number, version mismatch, or a corrupted file structure.
    """
    pass


class CodecError(DicomCubeError):
    """Raised when an error occurs in the encoding/decoding process.
    
    This exception is raised when there are problems with image compression
    or decompression, such as JPEG 2000 processing failures.
    """
    pass


class MetaDataError(DicomCubeError):
    """Raised when metadata is missing or inconsistent.
    
    This exception is raised when critical metadata (DicomMeta, Space, etc.)
    is missing, corrupted, or inconsistent in a DicomCube file or operation.
    """
    pass


class DataConsistencyError(DicomCubeError):
    """Raised when data arrays have consistency issues.
    
    This exception is raised when image data arrays have mismatched
    shapes, incompatible types, or other consistency-related issues.
    """
    pass 