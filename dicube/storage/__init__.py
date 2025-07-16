"""Storage module for DiCube library.

This module provides implementations of DiCube binary file formats (.dcb)
and utilities for pixel data processing. It handles the storage and retrieval
of 3D medical images along with their metadata.

Classes:
    DcbFile: Base class for DiCube file implementations.
    DcbSFile: Speed-optimized implementation of DiCube file format.
"""

from .dcb_file import DcbFile, DcbSFile

__all__ = [
    "DcbFile",
    "DcbSFile",
] 