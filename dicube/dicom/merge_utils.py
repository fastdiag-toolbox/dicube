from typing import Any, Dict, List, Optional, Tuple


###############################################################################
# Helper Functions: Check Value Equality
###############################################################################
def _all_identical(values: List[Any]) -> bool:
    """
    Check if all elements in a list are identical (including None).

    Args:
        values: List of values to compare

    Returns:
        bool: True if list is empty, has one element, or all elements are identical
    """
    if (not values) or (len(values) <= 1):
        return True
    first = values[0]
    return all(v == first for v in values[1:])


###############################################################################
# Recursively Merge Dataset JSONs
###############################################################################
def _merge_dataset_list(
    dataset_jsons: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Merge a list of pydicom JSON representations at the top level.

    Creates a merged dictionary where each tag entry contains:
    {
      "vr": str,                   # DICOM Value Representation
      "shared": True/False/None,   # None for sequences (SQ)
      "Value": [single value/list/sequence structure]
    }

    Args:
        dataset_jsons: List of pydicom JSON dictionaries to merge

    Returns:
        dict: Merged data with format {tag: merged_entry, ...}
    """
    # 1. Collect all unique tags
    all_tags = set()
    for js in dataset_jsons:
        all_tags.update(js.keys())

    merged_data = {}
    for tag in sorted(all_tags):
        # Collect values for this tag from all datasets
        # Note: Each value is like {"vr": "XX", "Value": [...]} or None
        tag_values = [ds_js.get(tag, None) for ds_js in dataset_jsons]

        # 2. Get VR if present in any dataset
        vrs = [tv["vr"] for tv in tag_values if tv is not None]
        vr = vrs[0] if vrs else None

        # 3. Merge values
        merged_data[tag] = _merge_tag_values(vr, tag_values)
    return merged_data


def _get_value_and_name(tv: Optional[Dict[str, Any]]) -> Tuple[Optional[str], Any]:
    """
    Extract the value and its field name from a tag value dictionary.

    Handles different value storage methods in DICOM:
    - Standard Value field
    - InlineBinary for binary data
    - BulkDataURI for external references

    Args:
        tv: Tag value dictionary or None

    Returns:
        tuple: (field_name, actual_value) where both may be None
    """
    if tv is not None and "Value" in tv:
        value_name = "Value"
        actual_value = tv["Value"]
    elif tv is not None and "InlineBinary" in tv:
        actual_value = tv["InlineBinary"]
        value_name = "InlineBinary"
    elif tv is not None and "BulkDataURI" in tv:
        actual_value = tv["BulkDataURI"]
        value_name = "BulkDataURI"
    else:
        value_name = None
        actual_value = None
    return value_name, actual_value


def _merge_tag_values(
    vr: Optional[str], tag_values: List[Optional[Dict[str, Any]]]
) -> Dict[str, Any]:
    """
    Merge values for a single tag across multiple datasets.

    For sequences (VR=SQ), recursively merges nested structures.
    For other VRs, determines if values are shared across datasets.

    Args:
        vr: DICOM Value Representation (VR) code
        tag_values: List of value dictionaries from each dataset

    Returns:
        dict: Merged entry with format:
            {
                "vr": str,
                "shared": bool/None,
                "Value": merged_value
            }
    """
    # If tag is missing from all datasets, return empty shell
    if all(tv is None for tv in tag_values):
        return {"vr": vr, "shared": True}

    if vr == "SQ":
        # Handle sequences recursively
        return _merge_sequence(tag_values)
    else:
        # Handle standard values
        # Extract actual values (may be list[str], list[float], or single value)
        actual_values = []
        value_name = "Value"
        for tv in tag_values:
            value_name, actual_value = _get_value_and_name(tv)
            actual_values.append(actual_value)

        # Check if all values are identical
        if _all_identical(actual_values):
            if actual_values[0] is None:
                return {"vr": vr, "shared": True}
            else:
                return {
                    "vr": vr,
                    "shared": True,
                    value_name: actual_values[0],
                }  # Store single value
        else:
            for i, v in enumerate(actual_values):
                if v is None:
                    actual_values[i] = 'None'
            # Flatten single-element lists
            if all([len(v) == 1 for v in actual_values]):
                actual_values = [v[0] for v in actual_values]
            return {
                "vr": vr,
                "shared": False,
                value_name: actual_values,  # Store list for each dataset
            }


def _merge_sequence(sq_values: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Merge sequence values across datasets.

    Each element has format: {"vr": "SQ", "Value": [item1, item2, ...]} or None.
    Returns merged structure:
    {
      "vr": "SQ",
      "shared": None,  # Shared status determined at item level
      "Value": [
         # Merged items, each a dict with {tag: {vr, shared, Value}}
      ]
    }

    Args:
        sq_values: List of sequence value dictionaries from each dataset

    Returns:
        dict: Merged sequence structure
    """
    # 1. Extract actual sequence values, replacing None with empty list
    list_of_item_lists = []
    for sq_val in sq_values:
        if sq_val and "Value" in sq_val:
            list_of_item_lists.append(sq_val["Value"])
        else:
            list_of_item_lists.append([])

    # 2. Find maximum sequence length
    max_len = max(len(items) for items in list_of_item_lists)

    # 3. Merge items at each index
    merged_items = []
    for i in range(max_len):
        # Collect i-th item from each dataset (None if index out of range)
        item_jsons = []
        for items in list_of_item_lists:
            if i < len(items):
                item_jsons.append(items[i])
            else:
                item_jsons.append(None)

        # Recursively merge items
        merged_item = _merge_item(item_jsons)
        merged_items.append(merged_item)

    return {
        "vr": "SQ",
        "shared": None,  # Sequence sharing determined at item level
        "Value": merged_items,
    }


def _merge_item(item_jsons: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Merge corresponding sequence items from multiple datasets.

    Each item_json is a simplified dataset with format:
    {"xxxx": {"vr": "...", "Value": ...}, "yyyy": {"vr": "...", "Value": ...}}

    Args:
        item_jsons: List of item dictionaries from each dataset

    Returns:
        dict: Merged item dictionary
    """
    # Replace None with empty dict
    actual_jsons = [js if js is not None else {} for js in item_jsons]
    return _merge_dataset_list(actual_jsons)


###############################################################################
# Helper Functions: Split Merged Dataset Back to Original DICOM JSON Format
###############################################################################


def _slice_merged_data(merged_dataset: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """
    Extract data for a single dataset from merged data.

    Args:
        merged_dataset: Merged dataset dictionary
        idx: Index of the dataset to extract

    Returns:
        dict: Dataset dictionary containing only the specified slice
    """
    json_dict = {}
    for tag_key, tag_entry in merged_dataset.items():
        vr = tag_entry.get("vr")
        shared = tag_entry.get("shared")

        if shared is True:
            # Shared tags have same value across all datasets
            tmp = tag_entry.copy()
            tmp.pop("shared")
            json_dict[tag_key] = tmp
        elif shared is False:
            if "Value" in tag_entry:
                valuename = "Value"
            elif "InlineBinary" in tag_entry:
                valuename = "InlineBinary"
            elif "BulkDataURI" in tag_entry:
                valuename = "BulkDataURI"
            else:
                valuename = None
            value = tag_entry.get(valuename)

            value_idx = value[idx]
            if value_idx is None:
                json_dict[tag_key] = {"vr": vr}
            elif isinstance(value_idx, list) or (valuename != "Value"):
                json_dict[tag_key] = {"vr": vr, valuename: value_idx}
            else:
                json_dict[tag_key] = {"vr": vr, valuename: [value_idx]}
        else:
            # Handle sequences and special cases
            if vr == "SQ":
                value = tag_entry.get("Value")

                if value == []:
                    json_dict[tag_key] = {"vr": vr, "Value": value}
                else:
                    json_dict[tag_key] = {
                        "vr": vr,
                        "Value": [_slice_merged_data(value[0], idx)],
                    }
    return json_dict 