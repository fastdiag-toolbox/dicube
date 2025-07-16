import json
import os
import warnings
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pydicom
from pydicom.uid import generate_uid

from .dicom_tags import CommonTags, Tag, parse_tag
from .merge_utils import _get_value_and_name, _merge_dataset_list, _slice_merged_data


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


def _get_projection_location(meta: "DicomMeta"):
    """Calculate projection locations for datasets.

    Uses ImageOrientationPatient and ImagePositionPatient to calculate
    the projection location for each dataset along the normal vector.

    Args:
        meta (DicomMeta): DicomMeta object containing the datasets.

    Returns:
        list: Projection locations for each dataset.

    Raises:
        ValueError: If ImageOrientationPatient is not found or invalid.
    """
    # Get ImageOrientationPatient - shared
    orientation_entry = meta._merged_data.get(CommonTags.IMAGE_ORIENTATION_PATIENT.key)
    if orientation_entry and "Value" in orientation_entry:
        orientation = orientation_entry["Value"]
        # Convert orientation values to float
        orientation = [float(v) for v in orientation]
        row_orientation = np.array(orientation[:3])
        col_orientation = np.array(orientation[3:])
        normal_vector = np.cross(row_orientation, col_orientation)
    else:
        raise ValueError("ImageOrientationPatient not found or invalid.")

    # Get positions for each dataset
    positions = meta.get(CommonTags.IMAGE_POSITION_PATIENT, force_nonshared=True)
    projection_locations = []
    for pos in positions:
        if pos:
            # Convert position values to float
            pos = [float(v) for v in pos]
            projection_location = np.dot(pos, normal_vector)
            projection_locations.append(projection_location)
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

    # Prepare shared and non-shared data
    shared_data = []
    non_shared_data = {}
    non_shared_tags = []

    # Ensure filenames are available and extract base filenames without paths
    if meta.filenames:
        filenames = meta.filenames
    else:
        # If filenames are not stored, generate default filenames
        filenames = [f"Dataset_{i}" for i in range(meta.num_datasets)]

    # Define priority tags for ordering
    priority_shared_tags = [
        CommonTags.PATIENT_NAME,
        CommonTags.PATIENT_ID,
        CommonTags.STUDY_DATE,
        CommonTags.STUDY_DESCRIPTION,
        CommonTags.SERIES_DESCRIPTION,
        CommonTags.MODALITY,
        # Add other common shared tags as needed
    ]

    priority_non_shared_tags = [
        CommonTags.INSTANCE_NUMBER,
        CommonTags.SLICE_LOCATION,
        CommonTags.IMAGE_POSITION_PATIENT,
        # Add other common non-shared tags as needed
    ]

    # Process each tag in the combined dataset
    for tag_key, tag_entry in meta.items():
        # Create Tag object from tag_key
        tag_obj = Tag(int(tag_key[:4], 16), int(tag_key[4:], 16), "")
        # Try to get name from CommonTags
        try:
            tag_obj.name = CommonTags.get_tag_by_tuple(
                (tag_obj.group, tag_obj.element)
            ).name
        except KeyError:
            tag_obj.name = pydicom.datadict.keyword_for_tag(
                (tag_obj.group, tag_obj.element)
            )

        if "shared" in tag_entry:
            if tag_entry["shared"]:
                if not show_shared:
                    continue
                # Shared tag
                value = tag_entry.get("Value")
                # Skip sequences for simplicity
                if tag_entry["vr"] != "SQ":
                    shared_data.append(
                        {
                            "Tag": tag_obj.format_tag(),
                            "Name": tag_obj.name,
                            "Value": value,
                        }
                    )
            else:
                if not show_non_shared:
                    continue
                # Non-shared tag
                values = tag_entry.get("Value")
                if tag_entry["vr"] != "SQ":
                    non_shared_tags.append(tag_obj)
                    non_shared_data[tag_obj.key] = {
                        "Name": tag_obj.name,
                        "Values": values,
                    }
        else:
            # Process sequences
            if tag_entry.get("vr") == "SQ" and "Value" in tag_entry:
                # For display purposes, we might skip sequences or implement similar logic
                pass

    # Sort shared tags, prioritizing common tags
    def tag_sort_key(tag_info):
        tag_key = tag_info["Tag"].replace("(", "").replace(")", "").replace(",", "")
        if tag_key in [t.key for t in priority_shared_tags]:
            return (0, [t.key for t in priority_shared_tags].index(tag_key))
        else:
            return (1, tag_info["Tag"])

    shared_data.sort(key=tag_sort_key)

    # Sort non-shared tags, prioritizing common tags
    def non_shared_sort_key(tag_obj):
        if tag_obj.key in [t.key for t in priority_non_shared_tags]:
            return (0, [t.key for t in priority_non_shared_tags].index(tag_obj.key))
        else:
            return (1, tag_obj.format_tag())

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
                tag_obj.format_tag(): tag_obj.format_tag()
                for tag_obj in non_shared_tags
            }
            name_row = {
                tag_obj.format_tag(): non_shared_data[tag_obj.key]["Name"]
                for tag_obj in non_shared_tags
            }

            # Collect values for each dataset
            values_rows = []
            for idx in range(meta.num_datasets):
                row = {
                    tag_obj.format_tag(): non_shared_data[tag_obj.key]["Values"][idx]
                    for tag_obj in non_shared_tags
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
        num_datasets (int): Number of datasets represented.
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
                self.num_datasets = len(tag_entry["Value"])
                break
        else:
            # If no non-shared fields are found, default to 1
            self.num_datasets = 1

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
        if not datasets:
            return cls({}, filenames)

        # Convert each dataset to a dict representation
        dicts = []
        for ds in datasets:
            dicts.append(ds.to_json_dict())

        # Merge the dictionaries
        merged_data = _merge_dataset_list(dicts)
        return cls(merged_data, filenames)

    def to_json(self) -> str:
        """Serialize the DicomMeta to a JSON string.

        Returns:
            str: JSON string representation of the DicomMeta.
        """
        data = {"_merged_data": self._merged_data, "num_datasets": self.num_datasets}
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

    def get(
        self,
        key: Union[str, Tag, tuple],
        force_shared: bool = False,
        force_nonshared: bool = False,
    ) -> Any:
        """Get a value from the DicomMeta.

        Args:
            key (Union[str, Tag, tuple]): Tag key as a string, Tag object, or tuple.
            force_shared (bool): If True, treat the tag as shared even if it's not.
                Defaults to False.
            force_nonshared (bool): If True, treat the tag as non-shared even if it's shared.
                Defaults to False.

        Returns:
            Any: The value associated with the key. If the tag is shared, returns a list with
                a single value. If the tag is non-shared, returns a list of values for each dataset.

        Raises:
            KeyError: If the key is not found in the merged data.
        """
        # Convert key to string if needed
        if isinstance(key, Tag):
            tag_key = key.key
        elif isinstance(key, tuple):
            tag_key = f"{key[0]:04x}{key[1]:04x}"
        else:
            tag_key = key

        # Get tag entry
        tag_entry = self._merged_data.get(tag_key)
        if tag_entry is None:
            return [None] * self.num_datasets

        # Get value based on shared status
        if "Value" not in tag_entry:
            return [None] * self.num_datasets

        if tag_entry.get("shared") and not force_nonshared:
            # For shared tags, return a list with the same value for all datasets
            return [tag_entry["Value"]] * self.num_datasets
        else:
            # For non-shared tags, return the list of values
            if force_shared:
                # If forcing shared, return the first non-None value
                values = tag_entry["Value"]
                for value in values:
                    if value is not None:
                        return [value] * self.num_datasets
                return [None] * self.num_datasets
            else:
                return tag_entry["Value"]

    def __getitem__(self, key: Union[str, tuple, Tag]) -> tuple:
        """Get a value from the DicomMeta (dictionary-style access).

        This method is a shorthand for `get(key)` and returns the values as a tuple
        instead of a list.

        Args:
            key (Union[str, tuple, Tag]): Tag key as a string, Tag object, or tuple.

        Returns:
            tuple: The values associated with the key.

        Raises:
            KeyError: If the key is not found in the merged data.
        """
        return tuple(self.get(key))

    def _format_vr_value(self, value: Any, vr: str) -> Any:
        """Format a value based on its Value Representation (VR).

        Args:
            value (Any): The value to format.
            vr (str): The DICOM Value Representation (VR) code.

        Returns:
            Any: The formatted value according to the VR.
        """
        if value is None:
            return None

        # Handle common VR types
        if vr in ("DS", "FL", "FD"):
            return float(value)
        elif vr in ("IS", "SL", "SS", "UL", "US"):
            return int(value)
        elif vr == "SQ":
            # Sequences are handled separately
            return value
        else:
            # For all other VRs, return as is
            return value

    def set_shared_item(self, key: Union[str, tuple, Tag], value: Any):
        """Set a shared metadata item for all datasets.

        Args:
            key (Union[str, tuple, Tag]): Tag key as a string, Tag object, or tuple.
            value (Any): The value to set for the tag.
        """
        # Convert key to string if needed
        if isinstance(key, Tag):
            tag_key = key.key
            tag_vr = key.vr
        elif isinstance(key, tuple):
            tag_obj = CommonTags.get_tag_by_tuple(key)
            tag_key = tag_obj.key
            tag_vr = tag_obj.vr
        else:
            tag_key = key
            tag_obj = parse_tag(key)
            tag_vr = tag_obj.vr

        # Get existing entry or create new one
        tag_entry = self._merged_data.get(tag_key, {})
        tag_entry["vr"] = tag_vr
        tag_entry["shared"] = True

        # Format the value based on VR
        formatted_value = self._format_vr_value(value, tag_vr)
        tag_entry["Value"] = formatted_value

        # Update the merged data
        self._merged_data[tag_key] = tag_entry

    def set_nonshared_item(self, key: Union[str, tuple, Tag], values: List[Any]):
        """Set a non-shared metadata item with different values for each dataset.

        Args:
            key (Union[str, tuple, Tag]): Tag key as a string, Tag object, or tuple.
            values (List[Any]): List of values, one for each dataset.

        Raises:
            ValueError: If the length of values doesn't match the number of datasets.
        """
        if len(values) != self.num_datasets:
            raise ValueError(
                f"Length of values ({len(values)}) must match the number of datasets ({self.num_datasets})"
            )

        # Convert key to string if needed
        if isinstance(key, Tag):
            tag_key = key.key
            tag_vr = key.vr
        elif isinstance(key, tuple):
            tag_obj = CommonTags.get_tag_by_tuple(key)
            tag_key = tag_obj.key
            tag_vr = tag_obj.vr
        else:
            tag_key = key
            tag_obj = parse_tag(key)
            tag_vr = tag_obj.vr

        # Get existing entry or create new one
        tag_entry = self._merged_data.get(tag_key, {})
        tag_entry["vr"] = tag_vr
        tag_entry["shared"] = False

        # Format the values based on VR
        formatted_values = [self._format_vr_value(value, tag_vr) for value in values]
        tag_entry["Value"] = formatted_values

        # Update the merged data
        self._merged_data[tag_key] = tag_entry

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

    def _sort_sequence(self, sequence_list, sort_index):
        """Sort a sequence list based on the sort_index.

        Args:
            sequence_list (List): List of sequences to sort.
            sort_index (List[int]): List of indices to use for sorting.

        Returns:
            List: Sorted sequence list.
        """
        # If the sequence list is None or empty, return as is
        if not sequence_list:
            return sequence_list

        # Create a new sequence list sorted by the sort_index
        sorted_sequence = [sequence_list[i] if i < len(sequence_list) else None for i in sort_index]
        return sorted_sequence

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
            instance_numbers = self.get(CommonTags.INSTANCE_NUMBER, force_nonshared=True)
            indices = list(range(self.num_datasets))
            # Convert to integers for sorting
            int_values = [safe_int(v) for v in instance_numbers]
            # Sort based on instance numbers
            sorted_indices = [
                i for _, i in sorted(zip(int_values, indices), key=lambda x: (x[0] is None, x[0]))
            ]

        elif sort_method == SortMethod.INSTANCE_NUMBER_DESC:
            # Get instance numbers
            instance_numbers = self.get(CommonTags.INSTANCE_NUMBER, force_nonshared=True)
            indices = list(range(self.num_datasets))
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
            indices = list(range(self.num_datasets))
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

            # Recursively handle sequences
            if tag_entry.get("vr") == "SQ" and "Value" in tag_entry:
                self._sort_sequence(tag_entry["Value"], sorted_indices)

        # Reorder filenames if available
        if self.filenames:
            self.filenames = [
                self.filenames[i] if i < len(self.filenames) else None for i in sorted_indices
            ]

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

    # Find all DICOM files in the directory
    dicom_files = []
    for file_extension in ["", ".dcm", ".DCM", ".ima", ".IMA"]:
        pattern = os.path.join(directory, f"*{file_extension}")
        dicom_files.extend(glob.glob(pattern))

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
        meta.sort_files(sort_method)
        # Reorder datasets to match the meta order
        datasets = [datasets[i] for i in range(len(datasets))]

    return meta, datasets 