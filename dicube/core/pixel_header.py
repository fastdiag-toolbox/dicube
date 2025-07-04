import json
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional


@dataclass
class PixelDataHeader:
    """
    Header class for storing pixel data information in medical images.

    Stores metadata including:
    - Rescale factors (slope/intercept)
    - Original pixel data type
    - Window settings (center/width)
    - Value range (min/max)
    - Additional metadata in EXTRAS

    Attributes:
        RESCALE_SLOPE: Slope for linear transformation
        RESCALE_INTERCEPT: Intercept for linear transformation
        PIXEL_DTYPE: Pixel data type string (after convert to dcb file)
        ORIGINAL_PIXEL_DTYPE: Original pixel data type string (before convert to dcb file)
        WINDOW_CENTER: Optional window center value
        WINDOW_WIDTH: Optional window width value
        MAX_VAL: Optional maximum pixel value
        MIN_VAL: Optional minimum pixel value
        EXTRAS: Dictionary for additional metadata
    """

    RESCALE_SLOPE: float = 1.0
    RESCALE_INTERCEPT: float = 0.0
    ORIGINAL_PIXEL_DTYPE: str = "uint16"
    PIXEL_DTYPE: str = "uint16"
    WINDOW_CENTER: Optional[float] = None
    WINDOW_WIDTH: Optional[float] = None
    MAX_VAL: Optional[float] = None
    MIN_VAL: Optional[float] = None
    EXTRAS: Dict[str, any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """
        Convert the header to a dictionary for serialization.

        Merges EXTRAS field into the main dictionary and removes
        the redundant EXTRAS key.

        Returns:
            dict: Dictionary representation of the header
        """
        data = asdict(self)
        data.update(self.EXTRAS)  # Merge EXTRAS into dictionary
        data.pop("EXTRAS", None)  # Remove redundant EXTRAS field
        return data

    @classmethod
    def from_dict(cls, d: dict):
        """
        Create a PixelDataHeader from a dictionary.

        Args:
            d: Dictionary containing header data

        Returns:
            PixelDataHeader: A new instance with values from the dictionary
        """
        rescale_slope = d.get("RESCALE_SLOPE", 1.0)
        rescale_intercept = d.get("RESCALE_INTERCEPT", 0.0)
        original_pixel_dtype = d.get("ORIGINAL_PIXEL_DTYPE", "uint16")
        window_center = d.get("WINDOW_CENTER")  # Defaults to None
        window_width = d.get("WINDOW_WIDTH")  # Defaults to None
        max_val = d.get("MAX_VAL")  # Defaults to None
        min_val = d.get("MIN_VAL")  # Defaults to None

        # All other keys go into EXTRAS
        extras = {
            k: v
            for k, v in d.items()
            if k
            not in {
                "RESCALE_SLOPE",
                "RESCALE_INTERCEPT",
                "ORIGINAL_PIXEL_DTYPE",
                "WINDOW_CENTER",
                "WINDOW_WIDTH",
                "MAX_VAL",
                "MIN_VAL",
            }
        }

        return cls(
            RESCALE_SLOPE=rescale_slope,
            RESCALE_INTERCEPT=rescale_intercept,
            ORIGINAL_PIXEL_DTYPE=original_pixel_dtype,
            WINDOW_CENTER=window_center,
            WINDOW_WIDTH=window_width,
            MAX_VAL=max_val,
            MIN_VAL=min_val,
            EXTRAS=extras,
        )

    def to_json(self) -> str:
        """
        Serialize the header to a JSON string.

        Returns:
            str: JSON string representation of the header
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str):
        """
        Create a PixelDataHeader from a JSON string.

        Args:
            json_str: JSON string containing header data

        Returns:
            PixelDataHeader: A new instance created from the JSON data
        """
        obj_dict = json.loads(json_str)
        return cls.from_dict(obj_dict) 