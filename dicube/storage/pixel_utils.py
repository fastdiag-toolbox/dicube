from typing import Tuple

import numpy as np

from ..core.pixel_header import PixelDataHeader


def derive_pixel_header_from_array(
    image: np.ndarray, preferred_dtype=np.uint16
) -> Tuple[np.ndarray, PixelDataHeader]:
    """
    Derive pixel data header information from input numpy array.

    Process different data types in different ways:
    - For unsigned integers (uint8/16/32): use raw data directly
    - For signed integers (int8/16/32): convert to unsigned and record offset
    - For floating point (float16/32/64): normalize to specified unsigned integer range

    Args:
        image: Input image array
        preferred_dtype: Preferred output data type, defaults to np.uint16

    Returns:
        tuple: (Converted image array, PixelDataHeader object)

    Raises:
        ValueError: When preferred_dtype is not supported
        NotImplementedError: When input array dtype is not supported
    """
    dtype = str(image.dtype)
    if image.dtype in (np.uint16, np.uint8, np.uint32):
        return image, PixelDataHeader(
            RESCALE_SLOPE=1,
            RESCALE_INTERCEPT=0,
            PIXEL_DTYPE=dtype,
            ORIGINAL_PIXEL_DTYPE=dtype,
        )
    elif image.dtype == np.int16:
        min_val = int(np.min(image))
        image = (image - min_val).astype("uint16")
        return image, PixelDataHeader(
            RESCALE_SLOPE=1,
            RESCALE_INTERCEPT=min_val,
            PIXEL_DTYPE="uint16",
            ORIGINAL_PIXEL_DTYPE=dtype,
        )
    elif image.dtype == np.int8:
        min_val = int(np.min(image))
        image = (image - min_val).astype("uint8")
        return image, PixelDataHeader(
            RESCALE_SLOPE=1,
            RESCALE_INTERCEPT=min_val,
            PIXEL_DTYPE="uint8",
            ORIGINAL_PIXEL_DTYPE=dtype,
        )
    elif image.dtype == np.int32:
        min_val = int(np.min(image))
        image = (image - min_val).astype("uint32")
        return image, PixelDataHeader(
            RESCALE_SLOPE=1,
            RESCALE_INTERCEPT=min_val,
            PIXEL_DTYPE="uint32",
            ORIGINAL_PIXEL_DTYPE=dtype,
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
                RESCALE_SLOPE=1.0,
                RESCALE_INTERCEPT=float(min_val),
                PIXEL_DTYPE=preferred_dtype,
                ORIGINAL_PIXEL_DTYPE=dtype,
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
                RESCALE_SLOPE=slope,
                RESCALE_INTERCEPT=intercept,
                PIXEL_DTYPE=preferred_dtype,
                ORIGINAL_PIXEL_DTYPE=dtype,
                MAX_VAL=max_val,
                MIN_VAL=min_val,
            )
            return raw_image, header
    else:
        raise NotImplementedError("Unsupported dtype")


def get_float_data(
    raw_image: np.ndarray, pixel_header: PixelDataHeader, dtype="float32"
) -> np.ndarray:
    """
    Get image data as floating point array with slope/intercept applied.

    Inspired by NIfTI's get_fdata method, this converts the raw image data
    to floating point format and applies the rescale slope and intercept.

    Args:
        raw_image: Raw image data array
        pixel_header: Pixel data header containing rescale information
        dtype: Output data type, must be one of: float16, float32, float64

    Returns:
        np.ndarray: Floating point image data with rescale factors applied

    Raises:
        AssertionError: If dtype is not one of the allowed float types
    """
    assert dtype in (
        "float16",
        "float32",
        "float64",
    ), "only accept float16, float32, float64"

    # Note: Output may be positive or negative depending on original dtype and slope/intercept
    output_img = raw_image.astype(dtype)
    if pixel_header.RESCALE_SLOPE is not None:
        slope = np.array(pixel_header.RESCALE_SLOPE).astype(dtype)
        output_img *= slope
    if pixel_header.RESCALE_INTERCEPT is not None:
        intercept = np.array(pixel_header.RESCALE_INTERCEPT).astype(dtype)
        output_img += intercept
    return output_img 