import json
import os
import warnings
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pydicom
from pydicom.tag import Tag
from pydicom import datadict
from pydicom.uid import generate_uid

from .dicom_tags import CommonTags, get_tag_key


###############################################################################
# Enum: Specify sorting methods
###############################################################################
class SortMethod(Enum):
    """Enumeration of available sorting methods for DICOM datasets.

    Attributes:
        INSTANCE_NUMBER_ASC (int): Sort by instance number in ascending order.
        INSTANCE_NUMBER_DESC (int): Sort by instance number in descending order.
        POSITION_RIGHT_HAND (int): Sort by position using right-hand coordinate system.
        POSITION_LEFT_HAND (int): Sort by position using left-hand coordinate system.
    """

    INSTANCE_NUMBER_ASC = 1
    INSTANCE_NUMBER_DESC = 2
    POSITION_RIGHT_HAND = 3
    POSITION_LEFT_HAND = 4


def _get_projection_location(meta: "DicomMeta") -> List[float]:
    """Calculate projection locations for each dataset along the slice direction.
    
    Uses the normal vector from ImageOrientationPatient and positions from
    ImagePositionPatient to compute the projection distance for each slice.
    
    Args:
        meta: DicomMeta instance containing the required tags
        
    Returns:
        List[float]: List of projection locations, one for each dataset
        
    Raises:
        ValueError: If ImageOrientationPatient is not found or invalid
    """
    # Get ImageOrientationPatient - should be shared
    if not meta.is_shared(CommonTags.ImageOrientationPatient):
        raise ValueError("ImageOrientationPatient is not shared across datasets.")
        
    orientation = meta.get_shared_value(CommonTags.ImageOrientationPatient)
    if not orientation:
        raise ValueError("ImageOrientationPatient not found or invalid.")
        
    # Convert orientation values to float
    orientation = [float(v) for v in orientation]
    row_orientation = np.array(orientation[:3])
    col_orientation = np.array(orientation[3:])
    normal_vector = np.cross(row_orientation, col_orientation)
    
    # Get positions for each dataset
    positions = meta.get_values(CommonTags.ImagePositionPatient)
    
    projection_locations = []
    for pos in positions:
        if pos:
            position_array = np.array([float(v) for v in pos])
            # Project position onto normal vector
            projection = np.dot(position_array, normal_vector)
            projection_locations.append(projection)
        else:
            projection_locations.append(None)
    
    return projection_locations


###############################################################################
# Helper functions: Create metadata tables for display
###############################################################################


def _display(meta, show_shared=True, show_non_shared=True):
    """Display the shared and non-shared metadata in tabular format.

    Creates two separate tables:
    1. Shared metadata table with columns: Tag, Name, Value
    2. Non-shared metadata table where:
       - First row: Tag
       - Second row: Name
       - Following rows: Values for each dataset
       - Row labels: Filenames (without paths)

    Args:
        meta (DicomMeta): DicomMeta object to display.
        show_shared (bool): If True, display shared metadata. Defaults to True.
        show_non_shared (bool): If True, display non-shared metadata. Defaults to True.

    Returns:
        pandas.DataFrame: Formatted metadata tables.
    """

    import pandas as pd
    from .dicom_tags import CommonTags
    # Prepare shared and non-shared data
    shared_data = []
    non_shared_data = {}
    non_shared_tags = []

    # Ensure filenames are available and extract base filenames without paths
    if meta.filenames:
        filenames = meta.filenames
    else:
        # If filenames are not stored, generate default filenames
        filenames = [f"Dataset_{i}" for i in range(meta.slice_count)]

    # Define priority tags for ordering
    priority_shared_tags = [
        CommonTags.PatientName,
        CommonTags.PatientID,
        CommonTags.StudyDate,
        CommonTags.StudyDescription,
        CommonTags.SeriesDescription,
        CommonTags.Modality,
        # Add other common shared tags as needed
    ]

    priority_non_shared_tags = [
        CommonTags.InstanceNumber,
        CommonTags.SliceLocation,
        CommonTags.ImagePositionPatient,
        # Add other common non-shared tags as needed
    ]

    # Process each tag
    for tag_key in meta.keys():
        tag = Tag(int(tag_key[:4], 16), int(tag_key[4:], 16))
        tag_name = datadict.keyword_for_tag(tag)
        vr = meta.get_vr(tag)
        
        if meta.is_shared(tag):
            if show_shared and vr != "SQ":  # Skip sequences for simplicity
                value = meta.get_shared_value(tag)
                shared_data.append({
                    "Tag": f"({tag.group:04X},{tag.element:04X})",
                    "Name": tag_name,
                    "Value": value,
                })
        else:
            if show_non_shared and vr != "SQ":
                values = meta.get_values(tag)
                non_shared_tags.append(tag)
                non_shared_data[tag_key] = {
                    "Name": tag_name,
                    "Values": values,
                }

    # Sort shared tags, prioritizing common tags
    def tag_sort_key(tag_info):
        tag_str = tag_info["Tag"]
        tag_obj = Tag(int(tag_str[1:5], 16), int(tag_str[6:10], 16))
        if any(tag_obj == priority_tag for priority_tag in priority_shared_tags):
            for i, priority_tag in enumerate(priority_shared_tags):
                if tag_obj == priority_tag:
                    return (0, i)
        return (1, tag_str)

    shared_data.sort(key=tag_sort_key)

    # Sort non-shared tags, prioritizing common tags
    def non_shared_sort_key(tag_obj):
        if any(tag_obj == priority_tag for priority_tag in priority_non_shared_tags):
            for i, priority_tag in enumerate(priority_non_shared_tags):
                if tag_obj == priority_tag:
                    return (0, i)
        return (1, f"({tag_obj.group:04X},{tag_obj.element:04X})")

    non_shared_tags.sort(key=non_shared_sort_key)

    # Display shared metadata
    if show_shared:
        print("Shared Metadata:")
        if shared_data:
            shared_df = pd.DataFrame(shared_data)
            print(shared_df.to_string(index=False))
        else:
            print("No shared metadata.")

    # Display non-shared metadata
    if show_non_shared:
        print("\nNon-Shared Metadata:")
        if non_shared_tags:
            # Create the tag and name rows
            tag_row = {
                f"({tag.group:04X},{tag.element:04X})": f"({tag.group:04X},{tag.element:04X})"
                for tag in non_shared_tags
            }
            name_row = {
                f"({tag.group:04X},{tag.element:04X})": non_shared_data[get_tag_key(tag)]["Name"]
                for tag in non_shared_tags
            }

            # Collect values for each dataset
            values_rows = []
            for idx in range(meta.slice_count):
                row = {
                    f"({tag.group:04X},{tag.element:04X})": non_shared_data[get_tag_key(tag)]["Values"][idx]
                    for tag in non_shared_tags
                }
                values_rows.append(row)

            # Create DataFrame with tag, name, and values
            non_shared_df = pd.DataFrame([tag_row, name_row] + values_rows)
            # Set index with filenames starting from the third row
            non_shared_df.index = ["Tag", "Name"] + filenames

            print(non_shared_df.to_string())
        else:
            print("No non-shared metadata.")


###############################################################################
# DicomMeta Class
###############################################################################
class DicomMeta:
    """A class for managing metadata from multiple DICOM datasets.

    Uses pydicom's to_json_dict() to extract information from all levels (including sequences)
    of multiple DICOM datasets. Recursively determines which fields are:
    - Shared (identical across all datasets)
    - Non-shared (different across datasets)

    Provides methods to access, modify, and serialize this metadata.

    Attributes:
        _merged_data (Dict[str, Dict[str, Any]]): The merged metadata from all datasets.
        filenames (List[str], optional): List of filenames for the datasets.
        slice_count (int): Number of datasets represented.
    """

    def __init__(
        self,
        merged_data: Dict[str, Dict[str, Any]],
        filenames: Optional[List[str]] = None,
    ):
        """Initialize a DicomMeta instance.

        Args:
            merged_data (Dict[str, Dict[str, Any]]): The merged metadata from all datasets.
            filenames (List[str], optional): List of filenames for the datasets. Defaults to None.
        """
        self._merged_data = merged_data
        self.filenames = filenames
        # Calculate number of datasets from the first non-shared field
        for tag_entry in merged_data.values():
            if tag_entry.get("shared") is False and "Value" in tag_entry:
                self.slice_count = len(tag_entry["Value"])
                break
        else:
            # If no non-shared fields are found, default to 1
            if filenames is not None:
                self.slice_count = len(filenames)
            else:
                warnings.warn("No filenames provided, defaulting to 1 dataset")
                self.slice_count = 1

    @classmethod
    def from_datasets(
        cls, datasets: List[pydicom.Dataset], filenames: Optional[List[str]] = None
    ):
        """Create a DicomMeta instance from a list of pydicom datasets.

        Args:
            datasets (List[pydicom.Dataset]): List of pydicom datasets.
            filenames (List[str], optional): List of filenames corresponding to the datasets.
                Defaults to None.

        Returns:
            DicomMeta: A new DicomMeta instance created from the datasets.
        """
        from .merge_utils import _merge_dataset_list
        
        if not datasets:
            return cls({}, filenames)

        # Convert each dataset to a dict representation
        dicts = []
        for ds in datasets:
            tmp = ds.to_json_dict(
                bulk_data_threshold=10240, bulk_data_element_handler=lambda x: None
            )
            tmp.pop(get_tag_key(CommonTags.PixelData), None)
            dicts.append(tmp)
        # Merge the dictionaries
        merged_data = _merge_dataset_list(dicts)
        return cls(merged_data, filenames)

    def to_json(self) -> str:
        """Serialize the DicomMeta to a JSON string.

        Returns:
            str: JSON string representation of the DicomMeta.
        """
        data = {"_merged_data": self._merged_data, "slice_count": self.slice_count}
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str, filenames: List[str] = None):
        """Create a DicomMeta instance from a JSON string.

        Args:
            json_str (str): JSON string containing DicomMeta data.
            filenames (List[str], optional): List of filenames corresponding to the datasets.
                Defaults to None.

        Returns:
            DicomMeta: A new DicomMeta instance created from the JSON data.
        """
        data = json.loads(json_str)
        merged_data = data["_merged_data"]
        return cls(merged_data, filenames)


    def get_values(self, tag_input: Union[str, Tag, Tuple[int, int]]) -> List[Any]:
        """Get values for a tag across all datasets.
        
        Args:
            tag_input: The tag to retrieve, can be a Tag object, string, or (group, element) tuple
            
        Returns:
            List[Any]: List of values, one for each dataset. May contain None for datasets
                      where the tag is not present.
        """
        tag = Tag(tag_input)
        tag_key = get_tag_key(tag)
        
        # Get tag entry
        tag_entry = self._merged_data.get(tag_key)
        if tag_entry is None or "Value" not in tag_entry:
            return [None] * self.slice_count
            
        # Return values based on shared status
        if tag_entry.get("shared", False):
            # For shared tags, return the same value for all datasets
            return [tag_entry["Value"]] * self.slice_count
        else:
            # For non-shared tags, return the list of values
            return tag_entry["Value"]

    def is_shared(self, tag_input: Union[str, Tag, Tuple[int, int]]) -> bool:
        """Check if a tag has consistent values across all datasets.
        
        Args:
            tag_input: The tag to check, can be a Tag object, string, or (group, element) tuple
        
        Returns:
            bool: True if tag is shared (same value across all datasets), False otherwise
        """
        tag = Tag(tag_input)  # pydicom's Tag constructor handles various input formats
        tag_key = get_tag_key(tag)
        
        tag_entry = self._merged_data.get(tag_key)
        if tag_entry is None:
            return False
            
        return tag_entry.get("shared", False)


    def is_missing(self, tag_input: Union[str, Tag, Tuple[int, int]]) -> bool:
        """Check if a tag is missing from the metadata.
        
        Args:
            tag_input: The tag to check, can be a Tag object, string, or (group, element) tuple
        
        Returns:
            bool: True if tag is missing or has no value, False if present with a value
        """
        tag = Tag(tag_input)
        tag_key = get_tag_key(tag)
        
        tag_entry = self._merged_data.get(tag_key)
        return tag_entry is None or "Value" not in tag_entry


    def get_shared_value(self, tag_input: Union[str, Tag, Tuple[int, int]]) -> Any:
        """Get the shared value for a tag if it's shared across all datasets.
        
        Args:
            tag_input: The tag to retrieve, can be a Tag object, string, or (group, element) tuple
        
        Returns:
            Any: The shared value if tag is shared, None if not shared or missing
        """
        tag = Tag(tag_input)
        tag_key = get_tag_key(tag)
        
        tag_entry = self._merged_data.get(tag_key)
        if tag_entry is None or "Value" not in tag_entry:
            return None
            
        if tag_entry.get("shared", False):
            value = tag_entry["Value"]
            # If the value is a single-item list, extract the item
            if isinstance(value, list) and len(value) == 1:
                return value[0]
            return value
            
        return None


    def get_vr(self, tag_input: Union[str, Tag, Tuple[int, int]]) -> str:
        """Get the Value Representation (VR) for a tag.
        
        Args:
            tag_input: The tag to check, can be a Tag object, string, or (group, element) tuple
        
        Returns:
            str: The VR code (e.g., "CS", "LO", "SQ") or empty string if not found
        """
        tag = Tag(tag_input)
        tag_key = get_tag_key(tag)
        
        tag_entry = self._merged_data.get(tag_key)
        if tag_entry is None:
            return ""
            
        return tag_entry.get("vr", "")


    def __getitem__(self, tag_input: Union[str, Tag, Tuple[int, int]]) -> Tuple[Any, Optional[str]]:
        """Get a value and status for a tag (dictionary-style access).
        
        This method is useful for compatibility with status checkers that
        need to know both the value and whether it's shared across datasets.
        
        Args:
            tag_input: The tag to retrieve, can be a Tag object, string, or (group, element) tuple
            
        Returns:
            Tuple[Any, Optional[str]]: A tuple containing:
                - The value or list of values
                - Status string ('shared', 'non_shared', or None if missing)
        """
        tag = Tag(tag_input)
        tag_key = get_tag_key(tag)
        
        tag_entry = self._merged_data.get(tag_key)
        if tag_entry is None or "Value" not in tag_entry:
            return (None, None)
            
        if tag_entry.get("shared", False):
            return (tag_entry["Value"], "shared")
        else:
            return (tag_entry["Value"], "non_shared")

    def keys(self) -> List[str]:
        """Get all tag keys in the DicomMeta.

        Returns:
            List[str]: List of tag keys.
        """
        return list(self._merged_data.keys())

    def items(self):
        """Get all (key, value) pairs in the DicomMeta.

        Returns:
            Iterator: Iterator over (key, value) pairs.
        """
        return self._merged_data.items()

    def __len__(self) -> int:
        """Get the number of tags in the DicomMeta.

        Returns:
            int: Number of tags.
        """
        return len(self._merged_data)

    def set_shared_item(self, tag_input: Union[str, Tag, Tuple[int, int]], value: Any) -> None:
        """Set a shared metadata item for all datasets.
        
        Args:
            tag_input: The tag to set, can be a Tag object, string, or (group, element) tuple
            value: The value to set for the tag across all datasets
        """
        tag = Tag(tag_input)
        tag_key = get_tag_key(tag)
        vr = datadict.dictionary_VR(tag)
        
        # Get existing entry or create new one
        tag_entry = self._merged_data.get(tag_key, {})
        
        # Update the entry
        if not isinstance(value, list):
            value = [value]
        tag_entry["Value"] = value
        tag_entry["vr"] = vr
        tag_entry["shared"] = True
        
        # Store the updated entry
        self._merged_data[tag_key] = tag_entry


    def set_nonshared_item(self, tag_input: Union[str, Tag, Tuple[int, int]], values: List[Any]) -> None:
        """Set a non-shared metadata item with different values for each dataset.
        
        Args:
            tag_input: The tag to set, can be a Tag object, string, or (group, element) tuple
            values: List of values, one for each dataset
            
        Raises:
            ValueError: If the number of values doesn't match the number of datasets
        """
        if len(values) != self.slice_count:
            raise ValueError(
                f"Number of values ({len(values)}) does not match number of datasets ({self.slice_count})"
            )

        tag = Tag(tag_input)
        tag_key = get_tag_key(tag)
        vr = datadict.dictionary_VR(tag)
        
        # Get existing entry or create new one
        tag_entry = self._merged_data.get(tag_key, {})
        
        # Update the entry
        tag_entry["Value"] = values
        tag_entry["vr"] = vr
        tag_entry["shared"] = False
        
        # Store the updated entry
        self._merged_data[tag_key] = tag_entry

    def sort_files(
        self,
        sort_method: SortMethod = SortMethod.INSTANCE_NUMBER_ASC,
    ):
        """Sort the files in the DicomMeta.

        Args:
            sort_method (SortMethod): Method to use for sorting. Defaults to 
                SortMethod.INSTANCE_NUMBER_ASC.

        Raises:
            ValueError: If the sort method is not supported.
        """
        from .dicom_tags import CommonTags

        def safe_int(v):
            """Convert a value to integer safely.
            
            Args:
                v (Any): Value to convert.
                
            Returns:
                int: Converted integer value, or None if conversion fails.
            """
            try:
                return int(v)
            except (ValueError, TypeError):
                return None

        # Determine sort order based on method
        if sort_method == SortMethod.INSTANCE_NUMBER_ASC:
            # Get instance numbers
            instance_numbers = self.get_values(CommonTags.InstanceNumber)
            indices = list(range(self.slice_count))
            # Convert to integers for sorting
            int_values = [safe_int(v) for v in instance_numbers]
            # Sort based on instance numbers
            sorted_indices = [
                i for _, i in sorted(zip(int_values, indices), key=lambda x: (x[0] is None, x[0]))
            ]

        elif sort_method == SortMethod.INSTANCE_NUMBER_DESC:
            # Get instance numbers
            instance_numbers = self.get_values(CommonTags.InstanceNumber)
            indices = list(range(self.slice_count))
            # Convert to integers for sorting
            int_values = [safe_int(v) for v in instance_numbers]
            # Sort based on instance numbers (reverse)
            sorted_indices = [
                i
                for _, i in sorted(
                    zip(int_values, indices),
                    key=lambda x: (x[0] is None, -float("inf") if x[0] is None else -x[0]),
                )
            ]

        elif sort_method in (SortMethod.POSITION_RIGHT_HAND, SortMethod.POSITION_LEFT_HAND):
            # Calculate projection location along normal vector
            projection_locations = _get_projection_location(self)
            indices = list(range(self.slice_count))
            # Sort based on projection locations
            if sort_method == SortMethod.POSITION_RIGHT_HAND:
                sorted_indices = [
                    i
                    for _, i in sorted(
                        zip(projection_locations, indices),
                        key=lambda x: (x[0] is None, x[0]),
                    )
                ]
            else:  # SortMethod.POSITION_LEFT_HAND
                sorted_indices = [
                    i
                    for _, i in sorted(
                        zip(projection_locations, indices),
                        key=lambda x: (x[0] is None, -float("inf") if x[0] is None else -x[0]),
                    )
                ]
        else:
            raise ValueError(f"Unsupported sort method: {sort_method}")

        # Reorder all non-shared values according to the sorted indices
        for tag_key, tag_entry in self._merged_data.items():
            if tag_entry.get("shared") is False:
                # Reorder the values
                values = tag_entry.get("Value", [])
                tag_entry["Value"] = [values[i] if i < len(values) else None for i in sorted_indices]

        # Reorder filenames if available
        if self.filenames:
            self.filenames = [
                self.filenames[i] if i < len(self.filenames) else None for i in sorted_indices
            ]
        return sorted_indices

    def display(self, show_shared=True, show_non_shared=True):
        """Display the DicomMeta in a tabular format.

        Args:
            show_shared (bool): If True, display shared metadata. Defaults to True.
            show_non_shared (bool): If True, display non-shared metadata. Defaults to True.
        """
        _display(self, show_shared, show_non_shared)

    def _get_projection_location(self):
        """Calculate projection locations for all datasets.

        Returns:
            List[float]: Projection locations for all datasets.
        """
        return _get_projection_location(self)

    def index(self, index):
        """Create a new DicomMeta with only the specified dataset.

        Args:
            index (int): Index of the dataset to extract.

        Returns:
            DicomMeta: A new DicomMeta containing only the specified dataset.
        """
        from .merge_utils import _slice_merged_data
        return _slice_merged_data(self, index)


def read_dicom_dir(
    directory: str,
    stop_before_pixels=False,
    sort_method: SortMethod = SortMethod.INSTANCE_NUMBER_ASC,
):
    """Read all DICOM files from a directory.

    Args:
        directory (str): Path to the directory containing DICOM files.
        stop_before_pixels (bool): If True, don't read pixel data. Defaults to False.
        sort_method (SortMethod): Method to sort the DICOM files. 
                                 Defaults to SortMethod.INSTANCE_NUMBER_ASC.

    Returns:
        Tuple[DicomMeta, List[pydicom.Dataset]]: A tuple containing:
            - The merged DicomMeta object
            - The list of pydicom datasets

    Raises:
        ImportError: If pydicom is not installed.
        FileNotFoundError: If the directory doesn't exist.
        ValueError: If no DICOM files are found in the directory.
    """
    import glob

    try:
        import pydicom
    except ImportError:
        raise ImportError("pydicom is required to read DICOM files")

    # Check if directory exists
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    # Find all DICOM files in the directory - using a set to avoid duplicates
    dicom_files_set = set()
    for file_extension in ["", ".dcm", ".DCM", ".ima", ".IMA"]:
        pattern = os.path.join(directory, f"*{file_extension}")
        dicom_files_set.update(glob.glob(pattern))
    
    dicom_files = list(dicom_files_set)  # Convert back to list

    # Filter out non-DICOM files
    valid_files = []
    for file_path in dicom_files:
        try:
            # Try to read the file as DICOM
            dataset = pydicom.dcmread(
                file_path, stop_before_pixels=stop_before_pixels, force=True
            )
            # Check if it has basic DICOM attributes
            if (0x0008, 0x0016) in dataset:  # SOP Class UID
                valid_files.append(file_path)
        except Exception:
            # Not a valid DICOM file, skip
            continue

    if not valid_files:
        raise ValueError(f"No valid DICOM files found in: {directory}")

    # Read the valid DICOM files
    datasets = []
    filenames = []
    for file_path in valid_files:
        try:
            dataset = pydicom.dcmread(
                file_path, stop_before_pixels=stop_before_pixels, force=True
            )
            datasets.append(dataset)
            filenames.append(os.path.basename(file_path))
        except Exception as e:
            warnings.warn(f"Error reading {file_path}: {e}")

    # Create DicomMeta from datasets
    meta = DicomMeta.from_datasets(datasets, filenames)

    # Sort the files if needed
    if sort_method is not None:
        sorted_indices = meta.sort_files(sort_method)
        # Reorder datasets to match the meta order
        datasets = [datasets[i] for i in sorted_indices]

    return meta, datasets 