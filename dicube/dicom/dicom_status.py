from enum import Enum

import numpy as np

# Adjust the import based on your project structure
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
        meta: Object providing access to DICOM metadata (e.g., DicomMeta instance)
             Must support meta[Tag] -> (value, status) interface

    Returns:
        DicomStatus: The status enum value representing the check results
    """
    # --------------------------  Series UID --------------------------
    series_uid_value, series_uid_status = meta[CommonTags.SERIES_INSTANCE_UID]
    if series_uid_status is None:
        return DicomStatus.MISSING_SERIES_UID
    if series_uid_status == "non_shared":
        return DicomStatus.NON_UNIFORM_SERIES_UID

    # --------------------------  Instance Number --------------------------
    instance_num_value, instance_num_status = meta[CommonTags.INSTANCE_NUMBER]
    # 若没有取到任何 instance number
    if instance_num_status is None:
        return DicomStatus.MISSING_INSTANCE_NUMBER

    # instance_num_value 可能是单值（shared）或列表（non_shared）
    # 如果只有一张图像且 status='shared'，可将其视为单列表
    if instance_num_status == "shared":
        # 判断 meta 是否只有一张图像
        # 这里假设 meta.num_images() 或 len(meta) 可以拿到图像数量
        if len(meta) == 1:
            instance_numbers = [instance_num_value]  # 仅一帧
        else:
            return DicomStatus.DUPLICATE_INSTANCE_NUMBERS
    else:
        # non_shared: instance_num_value 应该是一个 list
        instance_numbers = instance_num_value

    # 检查是否有重复
    if len(set(instance_numbers)) < len(instance_numbers):
        return DicomStatus.DUPLICATE_INSTANCE_NUMBERS

    # 检查是否有gap
    # 例如期待连续: 1,2,3,4,...，若 diff 中有大于1的 => GAP_INSTANCE_NUMBER
    sorted_inst = np.sort(instance_numbers)
    diffs_inst = np.diff(sorted_inst)
    if len(diffs_inst) > 0 and not np.all(diffs_inst == 1):
        return DicomStatus.GAP_INSTANCE_NUMBER

    # --------------------------  Dtype (Bits) --------------------------
    # 尝试从 meta 中取 'RescaleIntercept' / 'RescaleSlope'
    # 如果压根不存在也不一定是错误
    _, bits_status1 = meta[CommonTags.BITS_STORED]
    _, bits_status2 = meta[CommonTags.BITS_ALLOCATED]
    _, bits_status3 = meta[CommonTags.HIGH_BIT]
    _, bits_status4 = meta[CommonTags.PIXEL_REPRESENTATION]

    if (
        (bits_status1 is None)
        or (bits_status2 is None)
        or (bits_status3 is None)
        or (bits_status4 is None)
    ):
        return DicomStatus.MISSING_DTYPE

    if not (
        bits_status1 == "shared"
        and bits_status2 == "shared"
        and bits_status3 == "shared"
        and bits_status4 == "shared"
    ):
        return DicomStatus.NON_UNIFORM_DTYPE

    # --------------------------  Pixel Spacing --------------------------
    spacing_value, spacing_status = meta[CommonTags.PIXEL_SPACING]
    if spacing_status is None:
        return DicomStatus.MISSING_SPACING
    if spacing_status == "non_shared":
        return DicomStatus.NON_UNIFORM_SPACING

    # --------------------------  Image Shape (Columns/Rows) --------------------------
    col_value, col_status = meta[CommonTags.COLUMNS]
    row_value, row_status = meta[CommonTags.ROWS]
    # 若缺失
    if col_status is None or row_status is None:
        return DicomStatus.MISSING_SHAPE
    # 若不一致(多帧不统一)
    if col_status == "non_shared" or row_status == "non_shared":
        return DicomStatus.NON_UNIFORM_SHAPE

    # --------------------------  Orientation --------------------------
    ori_value, ori_status = meta[CommonTags.IMAGE_ORIENTATION_PATIENT]
    if ori_status is None:
        return DicomStatus.MISSING_ORIENTATION
    if ori_status == "non_shared":
        return DicomStatus.NON_UNIFORM_ORIENTATION

    # --------------------------  Location (Z 方向) --------------------------
    # 需要 ImagePositionPatient 或 SliceLocation
    pos_value, pos_status = meta[CommonTags.IMAGE_POSITION_PATIENT]
    loc_value, loc_status = meta[CommonTags.SLICE_LOCATION]

    # 若两者均无，判定 MISSING_LOCATION
    if pos_status is None and loc_status is None:
        return DicomStatus.MISSING_LOCATION

    # 获取每张图像的 Z 值 (通常可由 ImagePositionPatient 或 SliceLocation 来推断)
    # 这里示例从 DicomMeta 提供的接口 `_get_projection_location()` 中获取
    z_locations = meta._get_projection_location()  # 返回长度与图像数相同的 Z 列表
    # 按照 instance number 排序后，再判断 Z 是否符合期待
    sort_idx = np.argsort(instance_numbers)
    sorted_z = np.array(z_locations)[sort_idx]

    # 检查是否出现"方向突变"或"反序部分"(这里只是示例，用 REVERSED_LOCATION 表示)
    diffs_z = np.diff(sorted_z)

    # 若出现正负混合 => 方向突变
    if np.min(diffs_z) < 0 < np.max(diffs_z):
        return DicomStatus.REVERSED_LOCATION

    # 检查是否有"停滞"(dwelling)，这里简单用"是否存在 0 差分"示例
    if np.any(diffs_z == 0):
        return DicomStatus.DWELLING_LOCATION

    # 检查 gap
    # 简化：若实测 diffs_z 中存在任意值> 1.5倍预期，就视为 GAP_LOCATION
    avg_gap = calculate_average_z_gap(sorted_z)
    if avg_gap == 0.0:
        return DicomStatus.DWELLING_LOCATION

    # 相对偏差
    ratio_diffs = np.abs(diffs_z - avg_gap) / (avg_gap + 1e-8)
    # 如果有某些差分比 avg_gap 大 50% (可自定义阈值)
    if np.any(ratio_diffs > 0.5):
        return DicomStatus.GAP_LOCATION

    # --------------------------  Recale Factor (Intercept/Slope) --------------------------
    # 尝试从 meta 中取 'RescaleIntercept' / 'RescaleSlope'
    # 如果压根不存在也不一定是错误
    intercept_val, intercept_status = meta[CommonTags.RESCALE_INTERCEPT]
    slope_val, slope_status = meta[CommonTags.RESCALE_SLOPE]
    if intercept_status == "non_shared" or slope_status == "non_shared":
        return DicomStatus.NON_UNIFORM_RESCALE_FACTOR

    # --------------------------  通关 --------------------------
    # 若都没问题 => 说明 Z 方向近似均匀
    return DicomStatus.CONSISTENT 