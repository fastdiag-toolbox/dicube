"""JPH codec adapter implementing ImageCodec interface."""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Union, Any, Tuple

from .ojph_complete import encode_image
from .ojph_decode_complete import decode_image


class JphCodec:
    """JPEG 2000 codec (OpenJPH) implementing ImageCodec interface.
    
    Attributes:
        id (int): Unique numeric ID for the codec.
        name (str): Codec name ("jph").
        extensions (Tuple[str, ...]): Supported file extensions (.j2k, .j2c, .jp2).
    """
    
    id: int = 2
    name: str = "jph"
    extensions: Tuple[str, ...] = (".j2k", ".j2c", ".jp2")
    
    def encode(
        self, 
        image: np.ndarray, 
        /, 
        reversible: bool = True,
        num_decompositions: int = 5,
        block_size: tuple = (64, 64),
        precinct_size: tuple = None,
        progression_order: str = "RPCL",
        color_transform: bool = False,
        profile: str = None,
        **kwargs: Any
    ) -> bytes:
        """Encode numpy array to JPEG 2000 bytes.
        
        Args:
            image (np.ndarray): Input image array.
            reversible (bool): Whether to use reversible transform. Defaults to True.
            num_decompositions (int): Number of wavelet decompositions. Defaults to 5.
            block_size (tuple): Code block size as (width, height). Defaults to (64, 64).
            precinct_size (tuple, optional): Precinct size for each level as (width, height).
                Defaults to None.
            progression_order (str): Progression order, one of LRCP, RLCP, RPCL, PCRL, CPRL.
                Defaults to "RPCL".
            color_transform (bool): Whether to use color transform. Defaults to False.
            profile (str, optional): Profile to use, one of None, IMF, BROADCAST.
                Defaults to None.
            **kwargs: Additional parameters (ignored for compatibility).
            
        Returns:
            bytes: Compressed JPEG 2000 data.
            
        Raises:
            ValueError: If the image dimensions or block size are invalid.
        """
        # Parameter validation
        if len(image.shape) not in (2, 3):
            raise ValueError("Image must be 2D or 3D array")

        # Validate code block size
        if not all(
            size > 0 and size <= 64 and (size & (size - 1)) == 0 for size in block_size
        ):
            raise ValueError(
                "Code block dimensions must be powers of 2 and not larger than 64"
            )

        # Ensure data is contiguous
        if not image.flags["C_CONTIGUOUS"]:
            image = np.ascontiguousarray(image)

        # Call C++ implementation
        return encode_image(
            image,
            reversible=reversible,
            num_decompositions=num_decompositions,
            block_size=block_size,
            precinct_size=precinct_size if precinct_size is not None else (0, 0),
            progression_order=progression_order,
            color_transform=color_transform,
            profile="" if profile is None else profile,
        )
    
    def encode_lossless(
        self,
        image: np.ndarray,
        /,
        **kwargs: Any
    ) -> bytes:
        """Encode numpy array to lossless JPEG 2000 bytes.
        
        This is a convenience method that calls encode() with reversible=True.
        
        Args:
            image (np.ndarray): Input image array.
            **kwargs: Additional parameters passed to encode().
            
        Returns:
            bytes: Compressed JPEG 2000 data.
        """
        return self.encode(image, reversible=True, **kwargs)
    
    def decode(
        self, 
        data: bytes, 
        /,
        level: int = 0,
        resilient: bool = False,
        **kwargs: Any
    ) -> np.ndarray:
        """Decode JPEG 2000 bytes to numpy array.
        
        Args:
            data (bytes): Compressed JPEG 2000 data.
            level (int): Resolution level to decode at (0 = full resolution).
                Defaults to 0.
            resilient (bool): Whether to enable resilient decoding. Defaults to False.
            **kwargs: Additional parameters (ignored for compatibility).
            
        Returns:
            np.ndarray: Decoded image as numpy array.
        """
        # Use C++ implementation
        return decode_image(data, level=level, resilient=resilient)

    
    def is_available(self) -> bool:
        """Check if JPEG 2000 codec is available and functional.
        
        Returns:
            bool: True if the codec is available and operational.
        """
        try:
            # Test with a small image
            test_image = np.ones((10, 10), dtype=np.uint8)
            encoded = self.encode(test_image)
            decoded = self.decode(encoded)
            return decoded.shape == test_image.shape
        except Exception:
            return False
    
    def get_version(self) -> str:
        """Get JPEG 2000 codec version.
        
        Returns:
            str: Version information string.
        """
        return "OpenJPH"  # TODO: Get actual version from OpenJPH
    
    def __repr__(self) -> str:
        """Get string representation of the codec.
        
        Returns:
            str: String representation.
        """
        return f"<{self.__class__.__name__} id={self.id} name='{self.name}' version='{self.get_version()}'>" 