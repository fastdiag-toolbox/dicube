from typing import Any, Tuple, Union

# Utility Functions


def parse_tag(
    input: Union[str, Tuple[Union[int, str], Union[int, str]], "Tag"]
) -> "Tag":
    """
    Parse a DICOM tag key from various input formats.

    Args:
        input: The input can be a string, a tuple of integers or strings, or a Tag object.

    Returns:
        A tuple of two integers representing the (group, element) of the DICOM tag.

    Raises:
        ValueError: If the input format is invalid or cannot be parsed.
    """
    if isinstance(input, Tag):
        # 如果已经是 Tag 对象，尝试从 CommonTags 获取更多信息
        try:
            return CommonTags.get_tag_by_tuple((input.group, input.element))
        except ValueError:
            return input

    if isinstance(input, tuple):
        if len(input) != 2:
            raise ValueError("Tuple input must have exactly two elements.")
        group, element = input
        if isinstance(group, str):
            group = int(group, 16)
        if isinstance(element, str):
            element = int(element, 16)
        # 尝试从 CommonTags 获取
        try:
            return CommonTags.get_tag_by_tuple((group, element))
        except ValueError:
            return Tag(group, element)

    if isinstance(input, str):
        input = input.strip()
        if len(input) == 8:  # e.g., "00100010"
            group = int(input[:4], 16)
            element = int(input[4:], 16)
        elif (
            len(input) == 10 and input.startswith("(") and input.endswith(")")
        ):  # e.g., "(0010,0010)"
            parts = input[1:-1].split(",")
            if len(parts) == 2:
                group = int(parts[0], 16)
                element = int(parts[1], 16)
        else:
            # Try to extract group and element from other formats
            parts = input.replace("(", "").replace(")", "").split(",")
            if len(parts) == 2:
                group = int(parts[0], 16)
                element = int(parts[1], 16)
            else:
                raise ValueError(f"Invalid input format: {input}")

        # 尝试从 CommonTags 获取
        try:
            return CommonTags.get_tag_by_tuple((group, element))
        except ValueError:
            return Tag(group, element)

    raise ValueError(f"Invalid input format: {input}")


def format_tag(tag: "Tag") -> str:
    """Format a Tag object into a string representation."""
    return f"({tag.group:04X},{tag.element:04X})"


def format_key(input: Any) -> str:
    if isinstance(input, Tag):
        return f"{input.group:04X}{input.element:04X}"
    if isinstance(input, tuple):
        return f"{input[0]:04X}{input[1]:04X}"


class Tag:
    """
    A class representing a DICOM tag with group, element, name and VR.

    Attributes:
        group: Group number of the DICOM tag (e.g., 0x0010)
        element: Element number of the DICOM tag (e.g., 0x0020)
        name: Human-readable name of the tag (e.g., "Patient ID")
        vr: Value Representation of the tag (e.g., "PN", "DA")
    """

    def __init__(self, group: int, element: int, name: str = "", vr: str = ""):
        """
        Initialize a Tag instance.

        Args:
            group: Group number of the DICOM tag
            element: Element number of the DICOM tag
            name: Optional human-readable name of the tag
            vr: Optional Value Representation of the tag
        """
        self.group = group
        self.element = element
        self.name = name
        self.vr = vr

    def __repr__(self):
        """
        Get the string representation of the tag for debugging.

        Returns:
            str: Format: Tag((XXXX,XXXX), 'name', 'VR')
        """
        return f"Tag({format_tag(self)}, '{self.name}', '{self.vr}')"

    def __str__(self):
        """
        Get the human-readable string representation of the tag.

        Returns:
            str: Format: name (XXXX,XXXX) [VR]
        """
        return f"{self.name} {format_tag(self)} [{self.vr}]"

    def __eq__(self, other):
        """
        Compare this tag with another tag or tuple.

        Args:
            other: Another Tag instance or (group, element) tuple

        Returns:
            bool: True if the tags have the same group and element
        """
        if isinstance(other, Tag):
            return (self.group, self.element) == (other.group, other.element)
        elif isinstance(other, tuple):
            return (self.group, self.element) == other
        return False

    def __hash__(self):
        """
        Get the hash value of the tag.

        Returns:
            int: Hash value based on group and element
        """
        return hash((self.group, self.element))

    @property
    def key(self):
        """
        Get the tag key in 8-digit hex format.

        Returns:
            str: Format: XXXXXXXX (e.g., '00100020')
        """
        return f"{self.group:04X}{self.element:04X}"

    @property
    def tag(self):
        """
        Get the tag as a tuple of group and element.

        Returns:
            tuple: Format: (group, element)
        """
        return (self.group, self.element)

    def format_tag(self):
        """
        Format the tag in DICOM standard format.

        Returns:
            str: Format: (XXXX,XXXX)
        """
        return format_tag(self)


# CommonTags Class


class CommonTags:
    """
    Collection of commonly used DICOM tags as Tag instances.

    Organized into categories:
    - Patient Information
    - Study Information
    - Series Information
    - Image Information
    - Modality Information
    - Pixel Data
    - Manufacturer Information
    - Other Common Tags
    """

    # Patient Information
    PATIENT_NAME = Tag(0x0010, 0x0010, "Patient Name", "PN")
    PATIENT_ID = Tag(0x0010, 0x0020, "Patient ID", "LO")
    PATIENT_BIRTH_DATE = Tag(0x0010, 0x0030, "Patient Birth Date", "DA")
    PATIENT_SEX = Tag(0x0010, 0x0040, "Patient Sex", "CS")
    PATIENT_AGE = Tag(0x0010, 0x1010, "Patient Age", "AS")
    PATIENT_WEIGHT = Tag(0x0010, 0x1030, "Patient Weight", "DS")

    # Study Information
    STUDY_INSTANCE_UID = Tag(0x0020, 0x000D, "Study Instance UID", "UI")
    STUDY_DATE = Tag(0x0008, 0x0020, "Study Date", "DA")
    STUDY_TIME = Tag(0x0008, 0x0030, "Study Time", "TM")
    STUDY_ID = Tag(0x0020, 0x0010, "Study ID", "SH")
    STUDY_DESCRIPTION = Tag(0x0008, 0x1030, "Study Description", "LO")
    ACCESSION_NUMBER = Tag(0x0008, 0x0050, "Accession Number", "SH")

    # Series Information
    SERIES_INSTANCE_UID = Tag(0x0020, 0x000E, "Series Instance UID", "UI")
    SERIES_NUMBER = Tag(0x0020, 0x0011, "Series Number", "IS")
    SERIES_DESCRIPTION = Tag(0x0008, 0x103E, "Series Description", "LO")

    # Image Information
    SOP_INSTANCE_UID = Tag(0x0008, 0x0018, "SOP Instance UID", "UI")
    INSTANCE_NUMBER = Tag(0x0020, 0x0013, "Instance Number", "IS")
    IMAGE_POSITION_PATIENT = Tag(0x0020, 0x0032, "Image Position (Patient)", "DS")
    IMAGE_ORIENTATION_PATIENT = Tag(0x0020, 0x0037, "Image Orientation (Patient)", "DS")
    SLICE_LOCATION = Tag(0x0020, 0x1041, "Slice Location", "DS")
    PIXEL_SPACING = Tag(0x0028, 0x0030, "Pixel Spacing", "DS")
    SLICE_THICKNESS = Tag(0x0018, 0x0050, "Slice Thickness", "DS")
    ROWS = Tag(0x0028, 0x0010, "Rows", "US")
    COLUMNS = Tag(0x0028, 0x0011, "Columns", "US")
    BITS_ALLOCATED = Tag(0x0028, 0x0100, "Bits Allocated", "US")
    BITS_STORED = Tag(0x0028, 0x0101, "Bits Stored", "US")
    HIGH_BIT = Tag(0x0028, 0x0102, "High Bit", "US")
    PIXEL_REPRESENTATION = Tag(0x0028, 0x0103, "Pixel Representation", "US")
    PHOTOMETRIC_INTERPRETATION = Tag(0x0028, 0x0004, "Photometric Interpretation", "CS")
    SAMPLES_PER_PIXEL = Tag(0x0028, 0x0002, "Samples Per Pixel", "US")
    RESCALE_INTERCEPT = Tag(0x0028, 0x1052, "Rescale Intercept", "DS")
    RESCALE_SLOPE = Tag(0x0028, 0x1053, "Rescale Slope", "DS")
    WINDOW_CENTER = Tag(0x0028, 0x1050, "Window Center", "DS")
    WINDOW_WIDTH = Tag(0x0028, 0x1051, "Window Width", "DS")
    PATIENT_POSITION = Tag(0x0018, 0x5100, "Patient Position", "CS")
    BODY_PART_EXAMINED = Tag(0x0018, 0x0015, "Body Part Examined", "CS")

    # Modality Information
    MODALITY = Tag(0x0008, 0x0060, "Modality", "CS")

    # Pixel Data
    PIXEL_DATA = Tag(0x7FE0, 0x0010, "Pixel Data", "OW")

    # Manufacturer Information
    MANUFACTURER = Tag(0x0008, 0x0070, "Manufacturer", "LO")
    MANUFACTURER_MODEL_NAME = Tag(0x0008, 0x1090, "Manufacturer's Model Name", "LO")
    SOFTWARE_VERSION = Tag(0x0018, 0x1020, "Software Version(s)", "LO")

    # Other Common Tags
    FRAME_OF_REFERENCE_UID = Tag(0x0020, 0x0052, "Frame of Reference UID", "UI")
    REFERENCED_IMAGE_SEQUENCE = Tag(0x0008, 0x1140, "Referenced Image Sequence", "SQ")
    REFERENCED_SOP_INSTANCE_UID = Tag(
        0x0008, 0x1155, "Referenced SOP Instance UID", "UI"
    )
    ACQUISITION_NUMBER = Tag(0x0020, 0x0012, "Acquisition Number", "IS")
    CONTRAST_AGENT = Tag(0x0018, 0x0010, "Contrast Agent", "LO")
    FLUORO_RATE = Tag(0x0018, 0x1151, "Fluoroscopy Rate", "DS")
    FLUORO_TIME = Tag(0x0018, 0x1150, "Fluoroscopy Total Time", "DS")

    @classmethod
    def get_tag_by_name(cls, name: str) -> Tag:
        """
        Get a Tag instance by its name.

        Args:
            name: Name of the tag to find

        Returns:
            Tag: The matching Tag instance

        Raises:
            ValueError: If no tag with the given name is found
        """
        for attr_name in dir(cls):
            if attr_name.isupper():  # Only check uppercase attributes
                attr = getattr(cls, attr_name)
                if isinstance(attr, Tag) and attr.name == name:
                    return attr
        raise ValueError(f"No tag found with name: {name}")

    @classmethod
    def get_tag_by_tuple(cls, tag_tuple: Tuple[int, int]) -> Tag:
        """
        Get a Tag instance by its (group, element) tuple.

        Args:
            tag_tuple: Tuple of (group, element) values

        Returns:
            Tag: The matching Tag instance

        Raises:
            ValueError: If no tag with the given tuple is found
        """
        for attr_name in dir(cls):
            if attr_name.isupper():  # Only check uppercase attributes
                attr = getattr(cls, attr_name)
                if isinstance(attr, Tag) and (attr.group, attr.element) == tag_tuple:
                    return attr
        raise ValueError(f"No tag found with tuple: {tag_tuple}") 