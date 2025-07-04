"""JXL codec adapter implementing ImageCodec interface."""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Union, Any

from .jxl_encode import imencode_jxl as _imencode_jxl_raw, jpegxl_version
from .jxl_decode import imdecode_jxl as _imdecode_jxl_raw


class JxlCodec:
    """JPEG XL codec implementing ImageCodec interface."""
    
    id: int = 1
    name: str = "jxl"
    extensions: tuple[str, ...] = (".jxl",)
    
    def encode(
        self, 
        image: np.ndarray, 
        /, 
        quality: int = 100,
        effort: int = 6,
        colorspace: str = None,
        bit_depth: int = None,
        **kwargs: Any
    ) -> bytes:
        """Encode numpy array to JPEG XL bytes.
        
        Args:
            image: Input image array
            quality: Compression quality [0, 100], default 100 (lossless)
            effort: Compression effort [1, 9], default 6
            colorspace: Color space specification (e.g. 'RGB', 'RGBA', 'L')
            bit_depth: Bit depth (8, 10, 12, 16), auto-detected if None
            **kwargs: Additional parameters (ignored for compatibility)
            
        Returns:
            Compressed JPEG XL data as bytes
        """
        # 参数验证
        if not isinstance(image, np.ndarray):
            raise ValueError("Input must be a numpy array")

        if image.ndim not in (2, 3):
            raise ValueError("Input must be a 2D or 3D array")
            
        # 处理None参数，转换为C++绑定期望的默认值
        if colorspace is None:
            colorspace = ""
        if bit_depth is None:
            bit_depth = -1
            
        # 使用基本的编码函数
        return _imencode_jxl_raw(
            image,
            quality=quality,
            effort=effort,
            colorspace=colorspace,
            bit_depth=bit_depth,
        )
    
    def decode(
        self, 
        data: bytes, 
        /,
        keep_orientation: bool = False,
        **kwargs: Any
    ) -> np.ndarray:
        """Decode JPEG XL bytes to numpy array.
        
        Args:
            data: Compressed JPEG XL data
            keep_orientation: Whether to keep original orientation
            **kwargs: Additional parameters (ignored for compatibility)
            
        Returns:
            Decoded image as numpy array
        """
        # 参数验证
        if not isinstance(data, bytes):
            raise ValueError("Input must be bytes")

        if len(data) == 0:
            raise ValueError("Input cannot be empty")

        try:
            return _imdecode_jxl_raw(data, keep_orientation)
        except Exception as e:
            raise ValueError(f"Failed to decode JPEG XL data: {e}") from e

    
    def is_available(self) -> bool:
        """Check if JPEG XL codec is available and functional."""
        try:
            # Test with a small image
            test_image = np.ones((10, 10), dtype=np.uint8)
            encoded = self.encode(test_image)
            decoded = self.decode(encoded)
            return decoded.shape == test_image.shape
        except Exception:
            return False
    
    def get_version(self) -> str:
        """Get JPEG XL codec version."""
        return jpegxl_version()
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.id} name='{self.name}' version='{self.get_version()}'>" 