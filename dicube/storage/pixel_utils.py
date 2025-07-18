from typing import Tuple

import numpy as np

from ..core.pixel_header import PixelDataHeader


def derive_pixel_header_from_array(
    image: np.ndarray, preferred_dtype=np.uint16
) -> Tuple[np.ndarray, PixelDataHeader]:
    """Derive pixel data header information from input numpy array.

    Process different data types in different ways:
    - For unsigned integers (uint8/16/32): use raw data directly
    - For signed integers (int8/16/32): convert to unsigned and record offset
    - For floating point (float16/32/64): normalize to specified unsigned integer range

    Args:
        image (np.ndarray): Input image array.
        preferred_dtype (np.dtype): Preferred output data type. Defaults to np.uint16.

    Returns:
        Tuple[np.ndarray, PixelDataHeader]: A tuple containing:
            - The converted image array
            - A PixelDataHeader object with appropriate metadata

    Raises:
        ValueError: When preferred_dtype is not supported.
        NotImplementedError: When input array dtype is not supported.
    """
    dtype = str(image.dtype)
    if image.dtype in (np.uint16, np.uint8, np.uint32):
        return image, PixelDataHeader(
            RescaleSlope=1,
            RescaleIntercept=0,
            PixelDtype=dtype,
            OriginalPixelDtype=dtype,
        )
    elif image.dtype == np.int16:
        min_val = int(np.min(image))
        image = (image - min_val).astype("uint16")
        return image, PixelDataHeader(
            RescaleSlope=1,
            RescaleIntercept=min_val,
            PixelDtype="uint16",
            OriginalPixelDtype=dtype,
        )
    elif image.dtype == np.int8:
        min_val = int(np.min(image))
        image = (image - min_val).astype("uint8")
        return image, PixelDataHeader(
            RescaleSlope=1,
            RescaleIntercept=min_val,
            PixelDtype="uint8",
            OriginalPixelDtype=dtype,
        )
    elif image.dtype == np.int32:
        min_val = int(np.min(image))
        image = (image - min_val).astype("uint32")
        return image, PixelDataHeader(
            RescaleSlope=1,
            RescaleIntercept=min_val,
            PixelDtype="uint32",
            OriginalPixelDtype=dtype,
        )
    elif image.dtype in (np.float16, np.float32, np.float64):
        if preferred_dtype == "uint8":
            dtype_max = 255
        elif preferred_dtype == "uint16":
            dtype_max = 65535
        else:
            raise ValueError(f"Unsupported preferred_dtype: {preferred_dtype}")

        min_val = image.min()
        max_val = image.max()
        if np.isclose(min_val, max_val):
            # For constant value arrays:
            # Set all pixels to 0, slope=0, intercept=min_val
            # When reading back: i*slope+intercept = min_val
            header = PixelDataHeader(
                RescaleSlope=1.0,
                RescaleIntercept=float(min_val),
                PixelDtype=preferred_dtype,
                OriginalPixelDtype=dtype,
            )
            raw_image = np.zeros_like(image, dtype=preferred_dtype)
            return raw_image, header
        else:
            slope = float(max_val - min_val)
            intercept = float(min_val)
            raw_image = ((image - intercept) / slope * dtype_max).astype(
                preferred_dtype
            )
            header = PixelDataHeader(
                RescaleSlope=slope,
                RescaleIntercept=intercept,
                PixelDtype=preferred_dtype,
                OriginalPixelDtype=dtype,
                MaxVal=max_val,
                MinVal=min_val,
            )
            return raw_image, header
    else:
        raise NotImplementedError("Unsupported dtype")


def get_float_data(
    raw_image: np.ndarray, pixel_header: PixelDataHeader, dtype="float32"
) -> np.ndarray:
    """Get image data as floating point array with slope/intercept applied.

    Inspired by NIfTI's get_fdata method, this converts the raw image data
    to floating point format and applies the rescale slope and intercept.

    Args:
        raw_image (np.ndarray): Raw image data array.
        pixel_header (PixelDataHeader): Pixel data header containing rescale information.
        dtype (str): Output data type, must be one of: float16, float32, float64. 
            Defaults to "float32".

    Returns:
        np.ndarray: Floating point image data with rescale factors applied.

    Raises:
        AssertionError: If dtype is not one of the allowed float types.
    """
    assert dtype in (
        "float16",
        "float32",
        "float64",
    ), "only accept float16, float32, float64"

    # Note: Output may be positive or negative depending on original dtype and slope/intercept
    output_img = raw_image.astype(dtype)
    if pixel_header.RescaleSlope is not None:
        slope = np.array(pixel_header.RescaleSlope).astype(dtype)
        if slope != 1.0:
            output_img *= slope
    if pixel_header.RescaleIntercept is not None:
        intercept = np.array(pixel_header.RescaleIntercept).astype(dtype)
        if intercept != 0.0:
            output_img += intercept
    return output_img


def determine_optimal_nifti_dtype(
    image: np.ndarray, pixel_header: PixelDataHeader
) -> Tuple[np.ndarray, str]:
    """Determine the optimal data type for saving to NIfTI and return the converted data.

    This function selects the most appropriate data type for NIfTI export based on the value range
    of the raw image and the rescale slope/intercept. It minimizes unnecessary data conversion and
    only applies scaling or offset if needed.

    Args:
        image (np.ndarray): The raw image data (integer type guaranteed).
        pixel_header (PixelDataHeader): Pixel header containing rescale information.

    Returns:
        Tuple[np.ndarray, str]:
            - The image data converted to the optimal type for NIfTI export.
            - The name of the chosen data type as a string.

    Raises:
        ValueError: If the data cannot be represented in any supported NIfTI type.

    Example:
        >>> arr = np.array([0, 100, 200], dtype=np.uint16)
        >>> header = PixelDataHeader(RescaleSlope=1.0, RescaleIntercept=0.0, OriginalPixelDtype="uint16", PixelDtype="uint16")
        >>> data, dtype_name = determine_optimal_nifti_dtype(arr, header)
        >>> print(data.dtype, dtype_name)
        uint8 uint8
    """
    # 获取slope和intercept
    slope = pixel_header.RescaleSlope if pixel_header.RescaleSlope is not None else 1.0
    intercept = pixel_header.RescaleIntercept if pixel_header.RescaleIntercept is not None else 0.0
    
    # 直接从原始数据计算值域
    raw_min = float(image.min())
    raw_max = float(image.max())
    
    # 计算应用slope和intercept后的值域
    min_val = raw_min * slope + intercept
    max_val = raw_max * slope + intercept
    
    # 如果斜率为负，需要交换min和max
    if slope < 0:
        min_val, max_val = max_val, min_val
    
    # 检查原始数据类型
    original_dtype = pixel_header.OriginalPixelDtype
    is_signed_original = original_dtype in ("int8", "int16", "int32", "int64")
    
    # 检查slope和intercept是否为整数值
    has_integer_transform = (
        np.isclose(slope % 1, 0) and 
        np.isclose(intercept % 1, 0)
    )
    
    # 准备最终数据 - 仅在确定dtype后执行一次转换
    result_dtype = None
    result_dtype_name = None
    
    # 如果slope和intercept都是整数，则可以使用整数类型
    if has_integer_transform:
        # 尊重原始数据类型的符号属性
        if is_signed_original or min_val < 0:
            # 有符号整数
            if min_val >= -128 and max_val <= 127:
                result_dtype = np.int8
                result_dtype_name = "int8"
            elif min_val >= -32768 and max_val <= 32767:
                result_dtype = np.int16
                result_dtype_name = "int16"
            elif min_val >= -2147483648 and max_val <= 2147483647:
                result_dtype = np.int32
                result_dtype_name = "int32"
            elif max_val <= 2147483647:  # 值域在int32范围内，但原始类型是int32
                result_dtype = np.int32
                result_dtype_name = "int32"
        else:
            # 无符号整数
            if max_val <= 255:
                result_dtype = np.uint8
                result_dtype_name = "uint8"
            elif max_val <= 65535:
                result_dtype = np.uint16
                result_dtype_name = "uint16"
            elif max_val <= 4294967295:
                result_dtype = np.uint32
                result_dtype_name = "uint32"
    
    # 如果没有找到合适的整数类型，使用浮点类型
    if result_dtype is None:
        if np.issubdtype(image.dtype, np.float64) or min_val < -3.4e38 or max_val > 3.4e38:
            result_dtype = np.float64
            result_dtype_name = "float64"
        else:
            result_dtype = np.float32
            result_dtype_name = "float32"
            
    if has_integer_transform:
        intercept = int(intercept)
    else:
        intercept = np.array(intercept,dtype=result_dtype)

    if slope == 1.0:
        # 只要加法
        return image.astype(result_dtype) + intercept, result_dtype_name
    else:
        # 需要乘法，生成最终数据
        if result_dtype in (np.float32, np.float64):
            # 浮点类型，直接使用浮点运算
            result = image.astype(result_dtype) * slope + intercept
        else:
            # 整数类型，先做浮点运算再转换
            result = (image.astype(np.float32) * slope + intercept).astype(result_dtype)
        
        return result, result_dtype_name 