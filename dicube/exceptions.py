# exceptions.py


class DicomCubeError(Exception):
    """
    Base exception class for all DicomCube-related errors.
    """

    pass


class InvalidCubeFileError(DicomCubeError):
    """
    Raised when a file is not a valid DicomCube file.
    This could be due to incorrect magic number, version mismatch,
    or a corrupted file structure.
    """

    pass


class CodecError(DicomCubeError):
    """
    Raised when an error occurs in the encoding/decoding process
    (e.g., JPEG 2000 compression or decompression failures).
    """

    pass


class MetaDataError(DicomCubeError):
    """
    Raised when critical meta information (DicomMeta, Frame, etc.)
    is missing or inconsistent in the DicomCube file.
    """

    pass


class DataConsistencyError(DicomCubeError):
    """
    Raised when data arrays (foreground/background/mask) have mismatched
    shapes, lengths, or other consistency-related issues.
    """

    pass 