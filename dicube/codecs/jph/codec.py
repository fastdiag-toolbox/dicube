"""JPH codec adapter implementing ImageCodec interface."""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Union, Any

from .ojph_complete import encode_image
from .ojph_decode_complete import decode_image


class JphCodec:
    """JPEG 2000 codec (OpenJPH) implementing ImageCodec interface."""
    
    id: int = 2
    name: str = "jph"
    extensions: tuple[str, ...] = (".j2k", ".j2c", ".jp2")
    
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
            image: Input image array
            reversible: Whether to use reversible transform (default: True)
            num_decompositions: Number of wavelet decompositions (default: 5)
            block_size: Code block size as (width, height) (default: (64, 64))
            precinct_size: Precinct size for each level as (width, height) (default: None)
            progression_order: Progression order, one of LRCP, RLCP, RPCL, PCRL, CPRL (default: RPCL)
            color_transform: Whether to use color transform (default: False)
            profile: Profile to use, one of None, IMF, BROADCAST (default: None)
            **kwargs: Additional parameters (ignored for compatibility)
            
        Returns:
            Compressed JPEG 2000 data as bytes
        """
        # 参数验证
        if len(image.shape) not in (2, 3):
            raise ValueError("Image must be 2D or 3D array")

        # 验证代码块大小
        if not all(
            size > 0 and size <= 64 and (size & (size - 1)) == 0 for size in block_size
        ):
            raise ValueError(
                "Code block dimensions must be powers of 2 and not larger than 64"
            )

        # 确保数据是连续的
        if not image.flags["C_CONTIGUOUS"]:
            image = np.ascontiguousarray(image)

        # 调用C++实现
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
            data: Compressed JPEG 2000 data
            level: Resolution level to decode at (0 = full resolution) (default: 0)
            resilient: Whether to enable resilient decoding (default: False)
            **kwargs: Additional parameters (ignored for compatibility)
            
        Returns:
            Decoded image as numpy array
        """
        # 使用C++实现
        return decode_image(data, level=level, resilient=resilient)

    
    def is_available(self) -> bool:
        """Check if JPEG 2000 codec is available and functional."""
        try:
            # Test with a small image
            test_image = np.ones((10, 10), dtype=np.uint8)
            encoded = self.encode(test_image)
            decoded = self.decode(encoded)
            return decoded.shape == test_image.shape
        except Exception:
            return False
    
    def get_version(self) -> str:
        """Get JPEG 2000 codec version."""
        return "OpenJPH"  # TODO: Get actual version from OpenJPH
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.id} name='{self.name}' version='{self.get_version()}'>" 