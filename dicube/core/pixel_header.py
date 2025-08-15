import json
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional


@dataclass
class PixelDataHeader:
    """Header class for storing pixel data information in medical images.

    Stores metadata including:
    - Rescale factors (slope/intercept)
    - Original pixel data type
    - Window settings (center/width)
    - Value range (min/max)
    - Additional metadata in extras

    Attributes:
        RescaleSlope (float): Slope for linear transformation.
        RescaleIntercept (float): Intercept for linear transformation.
        PixelDtype (str): Pixel data type string (after convert to dcb file).
        OriginalPixelDtype (str): Original pixel data type string (before convert to dcb file).
        WindowCenter (float, optional): Window center value for display.
        WindowWidth (float, optional): Window width value for display.
        MaxVal (float, optional): Maximum pixel value.
        MinVal (float, optional): Minimum pixel value.
        Extras (Dict[str, any]): Dictionary for additional metadata.
    """

    RescaleSlope: float = 1.0
    RescaleIntercept: float = 0.0
    OriginalPixelDtype: str = "uint16"
    PixelDtype: str = "uint16"
    WindowCenter: Optional[float] = None
    WindowWidth: Optional[float] = None
    MaxVal: Optional[float] = None
    MinVal: Optional[float] = None
    Extras: Dict[str, any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert the header to a dictionary for serialization.

        Merges extras field into the main dictionary and removes
        the redundant extras key.

        Returns:
            dict: Dictionary representation of the header.
        """
        data = asdict(self)
        data.update(self.Extras)  # Merge Extras into dictionary
        data.pop("Extras", None)  # Remove redundant Extras field
        return data

    @classmethod
    def from_dict(cls, d: dict):
        """Create a PixelDataHeader from a dictionary.

        Args:
            d (dict): Dictionary containing header data.

        Returns:
            PixelDataHeader: A new instance with values from the dictionary.
        """
        rescale_slope = d.get("RescaleSlope", 1.0)
        rescale_intercept = d.get("RescaleIntercept", 0.0)
        original_pixel_dtype = d.get("OriginalPixelDtype", "uint16")
        pixel_dtype = d.get("PixelDtype", "uint16")
        window_center = d.get("WindowCenter")  # Defaults to None
        window_width = d.get("WindowWidth")  # Defaults to None
        max_val = d.get("MaxVal")  # Defaults to None
        min_val = d.get("MinVal")  # Defaults to None

        # All other keys go into Extras
        extras = {
            k: v
            for k, v in d.items()
            if k
            not in {
                "RescaleSlope",
                "RescaleIntercept",
                "OriginalPixelDtype",
                "PixelDtype",
                "WindowCenter",
                "WindowWidth",
                "MaxVal",
                "MinVal",
            }
        }

        return cls(
            RescaleSlope=rescale_slope,
            RescaleIntercept=rescale_intercept,
            OriginalPixelDtype=original_pixel_dtype,
            PixelDtype=pixel_dtype,
            WindowCenter=window_center,
            WindowWidth=window_width,
            MaxVal=max_val,
            MinVal=min_val,
            Extras=extras,
        )

    def to_json(self) -> str:
        """Serialize the header to a JSON string.

        Returns:
            str: JSON string representation of the header.
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str):
        """Create a PixelDataHeader from a JSON string.

        Args:
            json_str (str): JSON string containing header data.

        Returns:
            PixelDataHeader: A new instance created from the JSON data.
        """
        obj_dict = json.loads(json_str)
        return cls.from_dict(obj_dict) 