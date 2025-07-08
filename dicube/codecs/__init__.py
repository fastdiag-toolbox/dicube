from __future__ import annotations

"""Image codec sub-package for dicube.

This package defines the codec registry and base interface for image codecs.
Currently supports JXL and JPH formats with extensible design for future formats.
"""

from typing import Dict, Protocol, runtime_checkable, Optional, Union, Any
import numpy as np
from pathlib import Path

__all__ = [
    "ImageCodec",
    "get_codec",
    "list_codecs",
    "register_codec",
    "is_codec_available",
]


@runtime_checkable
class ImageCodec(Protocol):
    """Base interface that all image codecs must implement."""

    id: int  # unique numeric ID
    name: str  # codec name (e.g. "jxl", "jph")
    extensions: tuple[str, ...]  # supported file extensions (e.g. (".jxl",))
    
    def encode(
        self, 
        image: np.ndarray, 
        /, 
        **kwargs: Any
    ) -> bytes:
        """Encode numpy array to compressed bytes.
        
        Args:
            image: Input image array
            **kwargs: Codec-specific parameters
            
        Returns:
            Compressed image data as bytes
        """
        ...
    
    def decode(
        self, 
        data: bytes, 
        /,
        **kwargs: Any
    ) -> np.ndarray:
        """Decode compressed bytes to numpy array.
        
        Args:
            data: Compressed image data
            **kwargs: Codec-specific parameters
            
        Returns:
            Decoded image as numpy array
        """
        ...
    

    
    def is_available(self) -> bool:
        """Check if codec is available and functional."""
        ...
    
    def get_version(self) -> str:
        """Get codec version information."""
        ...


# ---------------------------------------------------------------------------
# Registry implementation ---------------------------------------------------
# ---------------------------------------------------------------------------

_codec_registry: Dict[str, ImageCodec] = {}


def register_codec(codec: ImageCodec) -> None:
    """Register a codec in the global registry."""
    _codec_registry[codec.name.lower()] = codec


def get_codec(name: str) -> ImageCodec:
    """Get codec by name (case-insensitive).
    
    Args:
        name: Codec name (e.g. "jxl", "jph")
        
    Returns:
        Codec instance
        
    Raises:
        ValueError: If codec not found
    """
    try:
        return _codec_registry[name.lower()]
    except KeyError as err:
        available = list(_codec_registry.keys())
        raise ValueError(f"Unknown codec '{name}'. Available: {available}") from err


def list_codecs() -> list[str]:
    """List all registered codec names."""
    return list(_codec_registry.keys())


def is_codec_available(name: str) -> bool:
    """Check if a codec is available.
    
    Args:
        name: Codec name (e.g. "jxl", "jph")
        
    Returns:
        True if codec is available and functional, False otherwise
    """
    try:
        codec = get_codec(name)
        return codec.is_available()
    except ValueError:
        return False





# Import and register concrete implementations
try:
    from .jph.codec import JphCodec  
    register_codec(JphCodec())
except ImportError:
    pass  # JPH codec not available 