from enum import Enum

import numpy as np
from pydicom.tag import Tag

from .dicom_tags import CommonTags


class DicomStatus(Enum):
    """
    Enumeration of possible DICOM series status conditions.

    Each status represents a specific condition or issue that may be present
    in a DICOM series. The conditions are grouped into categories:
    - Series UID Issues
    - Instance Number Issues
    - Spacing Issues
    - Shape Issues
    - Orientation Issues
    - Data Type Issues
    - Location Issues
    - Consistency Status
    """

    # Series UID Issues
    NON_UNIFORM_SERIES_UID = (
        "non_uniform_series_uid"  # Multiple Series UIDs in one series
    )
    MISSING_SERIES_UID = "missing_series_uid"  # No Series UIDs present

    # Instance Number Issues
    DUPLICATE_INSTANCE_NUMBERS = (
        "duplicate_instance_numbers"  # Duplicated instance numbers (e.g., 1,1,2,2,3,3)
    )
    MISSING_INSTANCE_NUMBER = "missing_instance_number"  # Missing Instance Number
    GAP_INSTANCE_NUMBER = "gap_instance_number"  # Gaps in instance numbering

    # Spacing Issues
    MISSING_SPACING = "missing_spacing"  # Missing Pixel Spacing
    NON_UNIFORM_SPACING = (
        "non_uniform_spacing"  # Inconsistent Pixel Spacing (XY intervals)
    )

    # Shape Issues
    MISSING_SHAPE = "missing_shape"  # Missing image dimensions (Columns or Rows)
    NON_UNIFORM_SHAPE = "non_uniform_shape"  # Inconsistent image dimensions

    # Orientation Issues
    MISSING_ORIENTATION = "missing_orientation"  # Missing Image Orientation Patient
    NON_UNIFORM_ORIENTATION = (
        "non_uniform_orientation"  # Inconsistent Image Orientation Patient
    )

    # Data Type Issues
    NON_UNIFORM_RESCALE_FACTOR = (
        "non_uniform_rescale_factor"  # Inconsistent intercept or slope
    )
    MISSING_DTYPE = "missing_dtype"  # Missing data type information
    NON_UNIFORM_DTYPE = "non_uniform_dtype"  # Inconsistent data types

    # Location Issues
    MISSING_LOCATION = (
        "missing_location"  # Missing Slice Location and Image Position Patient
    )
    REVERSED_LOCATION = "reversed_location"  # Z-values reversed when sorted by instance (e.g., 1,2,3,2,1)
    DWELLING_LOCATION = (
        "dwelling_location"  # Z-values show stagnation (e.g., 1,2,3,3,4,5)
    )
    GAP_LOCATION = "gap_location"  # Z-values have gaps (e.g., 1,2,3,5,6)

    # Consistency Status
    CONSISTENT = "consistent"  # All checks pass, data is consistent
    INCONSISTENT = "inconsistent"  # Other inconsistencies not covered above


def calculate_average_z_gap(z_locations: np.ndarray) -> float:
    """
    Calculate the average gap between Z-axis locations.

    Uses a robust method to estimate the typical Z-axis interval:
    1. If a single interval appears in >80% of cases, use that value
    2. Otherwise, use the larger absolute value between median and mean

    Args:
        z_locations: Sorted array of Z-axis locations

    Returns:
        float: Estimated typical Z-axis interval; 0 if cannot be calculated
    """
    if len(z_locations) < 2:
        return 0.0
    diffs = np.diff(z_locations)
    if len(diffs) == 0:
        return 0.0

    # If one interval appears in >80% of cases, use it
    uniq_diffs, counts = np.unique(diffs, return_counts=True)
    if np.max(counts) / len(diffs) > 0.8:
        return uniq_diffs[np.argmax(counts)]

    # Otherwise use the larger of median or mean
    median_diff = np.median(diffs)
    mean_diff = np.mean(diffs)
    return max([median_diff, mean_diff], key=abs)


def get_dicom_status(meta) -> DicomStatus:
    """
    Check DICOM metadata and return the corresponding status.

    Performs a series of checks on the DICOM metadata to determine its status.
    Checks include:
    - Series UID consistency
    - Instance number sequence
    - Pixel spacing uniformity
    - Image dimensions
    - Patient orientation
    - Data type consistency
    - Z-axis location sequence

    Args:
        meta: DicomMeta instance providing access to DICOM metadata

    Returns:
        DicomStatus: The status enum value representing the check results
    """
    # --------------------------  Series UID --------------------------
    if meta.is_missing(CommonTags.SeriesInstanceUID):
        return DicomStatus.MISSING_SERIES_UID
    if not meta.is_shared(CommonTags.SeriesInstanceUID):
        return DicomStatus.NON_UNIFORM_SERIES_UID

    # --------------------------  Instance Number --------------------------
    if meta.is_missing(CommonTags.InstanceNumber):
        return DicomStatus.MISSING_INSTANCE_NUMBER

    # Get instance numbers (always treat as non-shared for this check)
    instance_numbers = meta.get_values(CommonTags.InstanceNumber)
    
    # Check for single image
    if meta.slice_count == 1:
        # Single image is fine, continue to next check
        pass
    else:
        # Check for duplicate instance numbers
        if len(set(instance_numbers)) < len(instance_numbers):
            return DicomStatus.DUPLICATE_INSTANCE_NUMBERS

        # Check for gaps in instance numbering
        # First convert to integers and sort
        try:
            int_instances = [int(num) if num is not None else None for num in instance_numbers]
            sorted_instances = sorted([num for num in int_instances if num is not None])
            
            # If we have a sequence with more than one image
            if len(sorted_instances) > 1:
                # Check if they form a continuous sequence
                diffs = np.diff(sorted_instances)
                if not np.all(diffs == 1):
                    return DicomStatus.GAP_INSTANCE_NUMBER
        except (ValueError, TypeError):
            # If conversion fails, we can't check for gaps
            pass

    # --------------------------  Dtype (Bits) --------------------------
    dtype_tags = [
        CommonTags.BitsStored,
        CommonTags.BitsAllocated,
        CommonTags.HighBit,
        CommonTags.PixelRepresentation
    ]
    
    # Check if any are missing
    if any(meta.is_missing(tag) for tag in dtype_tags):
        return DicomStatus.MISSING_DTYPE
        
    # Check if any are non-shared
    if any(not meta.is_shared(tag) for tag in dtype_tags):
        return DicomStatus.NON_UNIFORM_DTYPE

    # --------------------------  Pixel Spacing --------------------------
    if meta.is_missing(CommonTags.PixelSpacing):
        return DicomStatus.MISSING_SPACING
    if not meta.is_shared(CommonTags.PixelSpacing):
        return DicomStatus.NON_UNIFORM_SPACING

    # --------------------------  Image Shape (Columns/Rows) --------------------------
    if meta.is_missing(CommonTags.Columns) or meta.is_missing(CommonTags.Rows):
        return DicomStatus.MISSING_SHAPE
    if not meta.is_shared(CommonTags.Columns) or not meta.is_shared(CommonTags.Rows):
        return DicomStatus.NON_UNIFORM_SHAPE

    # --------------------------  Orientation --------------------------
    if meta.is_missing(CommonTags.ImageOrientationPatient):
        return DicomStatus.MISSING_ORIENTATION
    if not meta.is_shared(CommonTags.ImageOrientationPatient):
        return DicomStatus.NON_UNIFORM_ORIENTATION

    # --------------------------  Location (Z direction) --------------------------
    # Need either ImagePositionPatient or SliceLocation
    has_position = not meta.is_missing(CommonTags.ImagePositionPatient)
    has_location = not meta.is_missing(CommonTags.SliceLocation)

    # If both are missing, mark as missing location
    if not has_position and not has_location:
        return DicomStatus.MISSING_LOCATION

    # Get Z locations and check for issues
    # For multi-slice datasets only
    if meta.slice_count > 1:
        # Get Z locations from the DicomMeta helper method
        z_locations = meta._get_projection_location()
        
        # Get the order of instance numbers
        instance_numbers = meta.get_values(CommonTags.InstanceNumber)
        try:
            # Convert to integers and get sort order
            int_instances = [int(num) if num is not None else float('inf') for num in instance_numbers]
            sort_idx = np.argsort(int_instances)
            
            # Sort Z locations by instance number
            sorted_z = np.array([z_locations[i] for i in sort_idx if i < len(z_locations)])
            
            # Check for direction changes
            if len(sorted_z) > 1:
                diffs_z = np.diff(sorted_z)
                
                # Check for direction changes (sign changes in differences)
                if np.min(diffs_z) < 0 < np.max(diffs_z):
                    return DicomStatus.REVERSED_LOCATION
                
                # Check for duplicate positions (zero differences)
                if np.any(diffs_z == 0):
                    return DicomStatus.DWELLING_LOCATION
                
                # Check for gaps in Z locations
                avg_gap = calculate_average_z_gap(sorted_z)
                if avg_gap > 0.0:
                    # Calculate relative deviations from average gap
                    ratio_diffs = np.abs(diffs_z - avg_gap) / (avg_gap + 1e-8)
                    # If any gap is more than 50% different from average, mark as gap
                    if np.any(ratio_diffs > 0.5):
                        return DicomStatus.GAP_LOCATION
        except (ValueError, TypeError):
            # If conversion fails, we can't check for sequence issues
            pass

    # --------------------------  Rescale Factor (Intercept/Slope) --------------------------
    # These may not exist, so only check if they're present
    has_intercept = not meta.is_missing(CommonTags.RescaleIntercept)
    has_slope = not meta.is_missing(CommonTags.RescaleSlope)
    
    if has_intercept and has_slope:
        # If present, check for consistency
        if not meta.is_shared(CommonTags.RescaleIntercept) or not meta.is_shared(CommonTags.RescaleSlope):
            return DicomStatus.NON_UNIFORM_RESCALE_FACTOR

    # --------------------------  All checks passed --------------------------
    return DicomStatus.CONSISTENT 