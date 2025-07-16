"""JPEG 2000 (HTJ2K) codec module for DiCube.

This module provides JPEG 2000 High Throughput (HTJ2K) compression functionality
through the OpenJPH library. It implements the ImageCodec interface for seamless
integration with DiCube's storage system.

Classes:
    JphCodec: Implementation of the JPEG 2000 codec using OpenJPH.
"""

from .codec import JphCodec

__all__ = [
    "JphCodec",
]
