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
    """
    Enumeration of available sorting methods for DICOM datasets.

    Attributes:
        INSTANCE_NUMBER_ASC: Sort by instance number in ascending order
        INSTANCE_NUMBER_DESC: Sort by instance number in descending order
        POSITION_RIGHT_HAND: Sort by position using right-hand coordinate system
        POSITION_LEFT_HAND: Sort by position using left-hand coordinate system
    """

    INSTANCE_NUMBER_ASC = 1
    INSTANCE_NUMBER_DESC = 2
    POSITION_RIGHT_HAND = 3
    POSITION_LEFT_HAND = 4


def _get_projection_location(meta: "DicomMeta"):
    """
    Calculate projection locations for datasets.

    Uses ImageOrientationPatient and ImagePositionPatient to calculate
    the projection location for each dataset along the normal vector.

    Args:
        meta: DicomMeta object containing the datasets

    Returns:
        list: Projection locations for each dataset

    Raises:
        ValueError: If ImageOrientationPatient is not found or invalid
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
    """
    Display the shared and non-shared metadata in tabular format.

    Creates two separate tables:
    1. Shared metadata table with columns: Tag, Name, Value
    2. Non-shared metadata table where:
       - First row: Tag
       - Second row: Name
       - Following rows: Values for each dataset
       - Row labels: Filenames (without paths)

    Args:
        meta: DicomMeta object to display
        show_shared: If True, display shared metadata
        show_non_shared: If True, display non-shared metadata

    Returns:
        pandas.DataFrame: Formatted metadata tables
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
    """
    A class for managing metadata from multiple DICOM datasets.

    Uses pydicom's to_json_dict() to extract information from all levels (including sequences)
    of multiple DICOM datasets. Recursively determines which fields are:
    - Shared (identical across all datasets)
    - Non-shared (different across datasets, stored as lists)

    The merged result is stored in self._merged_data, where:
    - Keys are 8-digit hex strings (e.g., '00100010')
    - Values are dictionaries containing:
        {
            "vr": "PN",
            "shared": True/False/None,
            "Value": ...
        }
    For VR=SQ (sequences), "Value" is a list of similar merged dictionaries,
    forming a recursive structure.
    """

    def __init__(
        self,
        merged_data: Dict[str, Dict[str, Any]],
        filenames: Optional[List[str]] = None,
    ):
        """
        Initialize a DicomMeta instance.

        Args:
            merged_data: Recursively merged result from multiple dataset JSONs
            filenames: List of dataset filenames or identifiers, matching the number of datasets
        """
        self._merged_data = merged_data
        if filenames is None:
            filenames = [f"Dataset_{i}" for i in range(len(merged_data))]
        self.filenames = filenames
        self.num_datasets = len(filenames)

    @classmethod
    def from_datasets(
        cls, datasets: List[pydicom.Dataset], filenames: Optional[List[str]] = None
    ):
        """
        Create a DicomMeta instance from a list of pydicom.Dataset objects.

        Performs recursive merging of dataset metadata.

        Args:
            datasets: List of pydicom.Dataset objects
            filenames: Optional list of filenames for the datasets

        Returns:
            DicomMeta: A new instance containing the merged metadata
        """
        # Convert each dataset to JSON dictionary
        json_list = []
        for ds in datasets:
            tmp = ds.to_json_dict(
                bulk_data_threshold=10240, bulk_data_element_handler=lambda x: None
            )
            # tmp.pop(CommonTags.PIXEL_DATA.key)
            json_list.append(tmp)

        # Merge the JSON data
        merged_data = _merge_dataset_list(json_list)
        return cls(merged_data, filenames)

    def to_json(self) -> str:
        """
        Serialize the merged metadata to a JSON string.

        Useful for persistence or debugging.

        Returns:
            str: JSON string representation of the merged metadata
        """
        return json.dumps(self._merged_data)

    @classmethod
    def from_json(cls, json_str: str, filenames: List[str] = None):
        """
        Create a DicomMeta instance from a JSON string.

        Args:
            json_str: JSON string containing merged metadata
            filenames: Optional list of filenames for the datasets

        Returns:
            DicomMeta: A new instance created from the JSON string
        """
        combined_dataset = json.loads(json_str)
        return cls(combined_dataset, filenames)

    def get(
        self,
        key: Union[str, Tag, tuple],
        force_shared: bool = False,
        force_nonshared: bool = False,
    ) -> Any:
        """
        Get the value for a given key, with options to force shared or non-shared behavior.

        Args:
            key: The tag identifier, can be:
                - Tag name
                - Tag object
                - (group, element) tuple
            force_shared: If True, force the value to be shared
            force_nonshared: If True, force the value to be non-shared

        Returns:
            The value(s) for the given key:
            - Single value if shared or force_shared
            - List of values if non-shared or force_nonshared
            - None if key not found

        Raises:
            AssertionError: If both force_shared and force_nonshared are True

        Warns:
            If forcing shared/non-shared on a value that isn't naturally so
        """
        assert not (
            force_shared and force_nonshared
        ), "Cannot force both shared and non-shared."

        value, status = self[key]

        if value is None:
            return None

        if force_shared:
            if status == "non_shared":
                warnings.warn(f"{key} is not shared!")
                # Take the first value
                value = value[0] if value else None
        elif force_nonshared:
            if status == "shared":
                warnings.warn(f"{key} is not non-shared!")
                # Expand the shared value to a list
                value = [value] * self.num_datasets
        return value

    def __getitem__(self, key: Union[str, tuple, Tag]) -> tuple:
        """
        Access the (value, status) pair for a top-level tag.

        Args:
            key: The tag identifier, can be:
                - 8-digit hex string (e.g., '00100010')
                - (group, element) tuple (e.g., (0x0010,0x0010))
                - DICOM keyword (e.g., 'PatientName')

        Returns:
            tuple: (value, status_str) where:
            - If shared=True: status_str='shared', value=single value
            - If shared=False: status_str='non_shared', value=list (matching datasets count)
            - If VR=SQ: status_str='sequence', value=recursive sequence structure
              Example structure:
              {
                "vr": "SQ",
                "shared": None,
                "Value": [merged_item_0, merged_item_1, ...]
              }
        """
        # 转换 key => hex string
        tag_key = parse_tag(key).key

        entry = self._merged_data.get(tag_key)
        if entry is None:
            # 没有此tag
            return None, None

        is_shared = entry.get("shared")
        vr = entry.get("vr")
        val = entry.get("Value")

        if vr == "SQ":
            # 视为序列
            return val, "sequence"
        elif is_shared is True:
            return val, "shared"
        elif is_shared is False:
            return val, "non_shared"
        else:
            # 可能 None => 例如全为空
            return val, None

    def _format_vr_value(self, value: Any, vr: str) -> Any:
        """Format value according to VR type."""
        if value is None:
            return None

        if vr == "PN":
            # Person Name format: {'Alphabetic': 'name'}
            if isinstance(value, str):
                return {"Alphabetic": value}
            elif isinstance(value, dict) and "Alphabetic" in value:
                return value
            return {"Alphabetic": str(value)}
        elif vr == "UI":
            # UI (Unique Identifier)
            if isinstance(value, str):
                if not value.replace(".", "").isdigit():  # 如果包含非数字和点的字符
                    return generate_uid()  # 生成新的合法 UID
            return value
        return value

    def set_shared_item(self, key: Union[str, tuple, Tag], value: Any):
        """
        Set a shared DICOM tag value that is identical across all datasets.

        Args:
            key: DICOM tag identifier, can be:
                - 8-digit hex string (e.g., '00100010')
                - (group, element) tuple (e.g., (0x0010,0x0010))
                - Tag object
            value: The shared value to set

        Raises:
            ValueError: If value is a list or has incorrect length
        """
        tag = parse_tag(key)
        tag_key = tag.key
        if tag_key in self._merged_data:
            vr = self._merged_data[tag_key].get("vr")
        else:
            vr = tag.vr

        # 这些 VR 类型需要值是列表格式
        if not isinstance(value, list):
            value = [value]

        # 格式化每个值
        value = [self._format_vr_value(v, vr) for v in value]

        self._merged_data[tag_key] = {
            "vr": vr,
            "shared": True,
            "Value": value,
        }

    def set_nonshared_item(self, key: Union[str, tuple, Tag], values: List[Any]):
        """
        Set a non-shared DICOM tag value that can differ across datasets.

        Args:
            key: DICOM tag identifier, can be:
                - 8-digit hex string (e.g., '00100010')
                - (group, element) tuple (e.g., (0x0010,0x0010))
                - Tag object
            values: List of values to set, must match the number of datasets

        Raises:
            ValueError: If values is not a list or length doesn't match dataset count
        """
        if not isinstance(values, list):
            raise ValueError("Non-shared items must be lists.")
        if len(values) != self.num_datasets:
            raise ValueError(
                f"Length of values must match the number of datasets ({self.num_datasets})."
            )

        tag = parse_tag(key)
        tag_key = tag.key
        if tag_key in self._merged_data:
            vr = self._merged_data[tag_key].get("vr")
        else:
            vr = tag.vr

        self._merged_data[tag_key] = {
            "vr": vr,
            "shared": False,
            "Value": values,
        }

    def keys(self) -> List[str]:
        """
        Get a list of all top-level tag keys.

        Returns:
            List[str]: List of 8-digit hex format tag keys
        """
        return list(self._merged_data.keys())

    def items(self):
        """
        Get an iterator over (tag_key, merged_entry) pairs.

        Returns:
            Iterator: Yields (tag_key, merged_entry) tuples
        """
        return self._merged_data.items()

    def __len__(self) -> int:
        """
        Get the number of datasets.

        Returns:
            int: Number of datasets
        """
        return self.num_datasets

    def _sort_sequence(self, sequence_list, sort_index):
        """
        Recursively sort sequence values according to the sort index.

        Args:
            sequence_list: List of sequence items to sort
            sort_index: Index array for reordering values
        """
        for item in sequence_list:
            for tag_key, tag_entry in item.items():
                if tag_entry.get("shared") is False:
                    value_name, values = _get_value_and_name(tag_entry)
                    # Reorder the values according to sort_index
                    reordered_values = [values[i] for i in sort_index]
                    tag_entry[value_name] = reordered_values
                # Recursively handle nested sequences
                if tag_entry.get("vr") == "SQ" and "Value" in tag_entry:
                    self._sort_sequence(tag_entry["Value"], sort_index)

    def sort_files(
        self,
        sort_method: SortMethod = SortMethod.INSTANCE_NUMBER_ASC,
    ):
        """
        Sort datasets and filenames according to the specified method.

        Args:
            sort_method: Method to use for sorting:
                - INSTANCE_NUMBER_ASC: Sort by instance number ascending
                - INSTANCE_NUMBER_DESC: Sort by instance number descending
                - POSITION_RIGHT_HAND: Sort by position using right-hand coordinate system
                - POSITION_LEFT_HAND: Sort by position using left-hand coordinate system
        """

        def safe_int(v):
            """Convert value to int, return -1 if None"""
            if v is not None:
                return int(v)
            else:
                return -1

        """
        Sort datasets and filenames according to the specified method.
        """
        if sort_method == SortMethod.INSTANCE_NUMBER_ASC:
            values = self.get(CommonTags.INSTANCE_NUMBER, force_nonshared=True)
            # Convert values to integers, handle None
            values = [safe_int(v) for v in values]
            sort_index = np.argsort(values)
        elif sort_method == SortMethod.INSTANCE_NUMBER_DESC:
            values = self.get(CommonTags.INSTANCE_NUMBER, force_nonshared=True)
            values = [safe_int(v) for v in values]
            sort_index = np.argsort(values)[::-1]
        elif sort_method == SortMethod.POSITION_RIGHT_HAND:
            projection_locations = _get_projection_location(self)
            # Replace None with a large number to sort correctly
            projection_locations = [
                pl if pl is not None else float("inf") for pl in projection_locations
            ]
            sort_index = np.argsort(projection_locations)
        elif sort_method == SortMethod.POSITION_LEFT_HAND:
            projection_locations = _get_projection_location(self)
            projection_locations = [
                pl if pl is not None else float("-inf") for pl in projection_locations
            ]
            sort_index = np.argsort(projection_locations)[::-1]
        else:
            raise ValueError("Unknown sort method")

        # Reorder filenames
        if self.filenames:
            self.filenames = [self.filenames[i] for i in sort_index]

        # Reorder non-shared values in combined_dataset
        for tag_key, tag_entry in self._merged_data.items():
            if tag_entry.get("shared") is False:
                value_name, values = _get_value_and_name(tag_entry)
                # Reorder the values according to sort_index
                reordered_values = [values[i] for i in sort_index]
                tag_entry[value_name] = reordered_values

            # For sequences, recursively reorder non-shared values
            if tag_entry.get("vr") == "SQ" and "Value" in tag_entry:
                self._sort_sequence(tag_entry["Value"], sort_index)
        return sort_index

    def display(self, show_shared=True, show_non_shared=True):
        """
        Display the shared and non-shared metadata separately.

        Parameters:
            show_shared (bool): If True, display shared metadata.
            show_non_shared (bool): If True, display non-shared metadata.

        Shared metadata is displayed in a table with columns: Tag, Name, Value.

        Non-shared metadata is displayed in a table where:
            - The first row is the Tag
            - The second row is the Name
            - Rows starting from the third row are the values
            - The index (row labels) are the filenames (without leading directories)
        """
        return _display(self, show_shared, show_non_shared)

    def _get_projection_location(self):
        return _get_projection_location(self)

    def index(self, index):
        return _slice_merged_data(self._merged_data, index)


def read_dicom_dir(
    directory: str,
    stop_before_pixels=False,
    sort_method: SortMethod = SortMethod.INSTANCE_NUMBER_ASC,
):
    all_filenames = [os.path.join(directory, f) for f in os.listdir(directory)]
    datasets = []
    valid_filenames = []
    for filename in all_filenames:
        try:
            ds = pydicom.dcmread(filename, stop_before_pixels=stop_before_pixels)
            datasets.append(ds)
            valid_filenames.append(os.path.basename(filename))
        except pydicom.errors.InvalidDicomError:
            pass

    meta = DicomMeta.from_datasets(datasets, valid_filenames)
    sort_index = meta.sort_files(sort_method)
    datasets = [datasets[i] for i in sort_index]
    return meta, datasets 