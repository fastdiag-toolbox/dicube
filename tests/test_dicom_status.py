import copy

import pytest

from dicube import CommonTags, DicomStatus, get_dicom_status, read_dicom_dir


@pytest.fixture(scope="module")
def normal_meta(sample_dicom_dir):
    """
    读取一个"正常"的DICOM文件夹，返回对应的meta。
    """
    meta, _ = read_dicom_dir(sample_dicom_dir)
    return meta


def test_consistent(normal_meta):
    """
    测试：在未修改的正常元数据下，应返回 DicomStatus.CONSISTENT
    """
    status = get_dicom_status(normal_meta)
    assert status == DicomStatus.CONSISTENT


def test_missing_series_uid(normal_meta):
    """
    触发 MISSING_SERIES_UID
    """
    meta_copy = copy.deepcopy(normal_meta)
    # 将 SERIES_INSTANCE_UID 设置为 (None, True)，相当于 shared 值，但值为 None
    meta_copy._merged_data.pop(CommonTags.SERIES_INSTANCE_UID.key)
    status = get_dicom_status(meta_copy)
    assert status == DicomStatus.MISSING_SERIES_UID


def test_non_uniform_series_uid(normal_meta):
    """
    触发 NON_UNIFORM_SERIES_UID
    """
    meta_copy = copy.deepcopy(normal_meta)
    lens = len(normal_meta)
    # 将 SERIES_INSTANCE_UID 设置为一个列表(多个UID), 即 non_shared
    meta_copy.set_nonshared_item(
        CommonTags.SERIES_INSTANCE_UID, ["1.2.3", "4.5.6"] + [""] * (lens - 2)
    )
    status = get_dicom_status(meta_copy)
    assert status == DicomStatus.NON_UNIFORM_SERIES_UID


def test_missing_instance_number(normal_meta):
    """
    触发 MISSING_INSTANCE_NUMBER
    """
    meta_copy = copy.deepcopy(normal_meta)
    meta_copy._merged_data.pop(CommonTags.INSTANCE_NUMBER.key)
    status = get_dicom_status(meta_copy)
    assert status == DicomStatus.MISSING_INSTANCE_NUMBER


def test_duplicate_instance_numbers(normal_meta):
    """
    触发 DUPLICATE_INSTANCE_NUMBERS
    """
    meta_copy = copy.deepcopy(normal_meta)
    # 假设原本有 n 张图像，就模拟让其中的 InstanceNumber 全部重复
    # 例如本来是 [1, 2, 3, ..., n] => 全部设置为 [1, 1, 1, ..., 1]
    num_datasets = meta_copy.num_datasets
    meta_copy.set_nonshared_item(CommonTags.INSTANCE_NUMBER, [1] * num_datasets)
    status = get_dicom_status(meta_copy)
    assert status == DicomStatus.DUPLICATE_INSTANCE_NUMBERS


def test_gap_instance_number(normal_meta):
    """
    触发 GAP_INSTANCE_NUMBER
    """
    meta_copy = copy.deepcopy(normal_meta)
    # 将它们改成 [1,2,3,5,6,...] 人为制造一个 gap
    # 为简单起见，我们只改前4个值: [1,2,4,5], 剩下的按原值填充也可
    num_datasets = meta_copy.num_datasets
    original = list(range(1, num_datasets + 1))
    for i in range(4, len(original)):
        original[i] += 1
    meta_copy.set_nonshared_item(CommonTags.INSTANCE_NUMBER, original)
    status = get_dicom_status(meta_copy)
    assert status == DicomStatus.GAP_INSTANCE_NUMBER


def test_missing_spacing(normal_meta):
    """
    触发 MISSING_SPACING
    """
    meta_copy = copy.deepcopy(normal_meta)
    meta_copy._merged_data.pop(CommonTags.PIXEL_SPACING.key)
    status = get_dicom_status(meta_copy)
    assert status == DicomStatus.MISSING_SPACING


def test_non_uniform_spacing(normal_meta):
    """
    触发 NON_UNIFORM_SPACING
    """
    meta_copy = copy.deepcopy(normal_meta)
    # 模拟不同帧像素间距不一致
    # 比如前半帧 [0.8, 0.8], 后半帧 [1.0, 1.0]
    num = meta_copy.num_datasets
    half = num // 2
    values = []
    for i in range(num):
        if i < half:
            values.append([0.8, 0.8])
        else:
            values.append([1.0, 1.0])
    meta_copy.set_nonshared_item(CommonTags.PIXEL_SPACING, values)
    status = get_dicom_status(meta_copy)
    assert status == DicomStatus.NON_UNIFORM_SPACING


def test_missing_shape(normal_meta):
    """
    触发 MISSING_SHAPE
    - 缺失COLUMNS 或 ROWS
    """
    meta_copy = copy.deepcopy(normal_meta)
    meta_copy._merged_data.pop(CommonTags.COLUMNS.key)
    # 或者 meta_copy[CommonTags.ROWS] = (None, True)
    status = get_dicom_status(meta_copy)
    assert status == DicomStatus.MISSING_SHAPE


def test_non_uniform_shape(normal_meta):
    """
    触发 NON_UNIFORM_SHAPE
    """
    meta_copy = copy.deepcopy(normal_meta)
    # 假设 num_datasets 帧中，一半列数是 512，一半是 256
    num = meta_copy.num_datasets
    half = num // 2
    columns_list = []
    for i in range(num):
        if i < half:
            columns_list.append(512)
        else:
            columns_list.append(256)
    meta_copy.set_nonshared_item(CommonTags.COLUMNS, columns_list)
    status = get_dicom_status(meta_copy)
    assert status == DicomStatus.NON_UNIFORM_SHAPE


def test_missing_orientation(normal_meta):
    """
    触发 MISSING_ORIENTATION
    """
    meta_copy = copy.deepcopy(normal_meta)
    meta_copy._merged_data.pop(CommonTags.IMAGE_ORIENTATION_PATIENT.key)
    status = get_dicom_status(meta_copy)
    assert status == DicomStatus.MISSING_ORIENTATION


def test_non_uniform_orientation(normal_meta):
    """
    触发 NON_UNIFORM_ORIENTATION
    """
    meta_copy = copy.deepcopy(normal_meta)
    # 第一帧: [1,0,0, 0,1,0] ; 第二帧: [1,0,0, 0,0,-1], ...
    # 只要保证有差异即可
    num = meta_copy.num_datasets
    if num < 2:
        pytest.skip("需要至少2帧才能测试非统一方向")
    orientation_list = []
    for i in range(num):
        if i % 2 == 0:
            orientation_list.append([1, 0, 0, 0, 1, 0])
        else:
            orientation_list.append([1, 0, 0, 0, 0, -1])
    meta_copy.set_nonshared_item(CommonTags.IMAGE_ORIENTATION_PATIENT, orientation_list)
    status = get_dicom_status(meta_copy)
    assert status == DicomStatus.NON_UNIFORM_ORIENTATION


def test_missing_dtype(normal_meta):
    """
    触发 MISSING_DTYPE
    - 缺失 BITS_STORED (或其它相关) 信息
    """
    meta_copy = copy.deepcopy(normal_meta)
    # 只要把BITS_STORED, BITS_ALLOCATED 等统统设 None
    meta_copy._merged_data.pop(CommonTags.BITS_STORED.key)

    status = get_dicom_status(meta_copy)
    assert status == DicomStatus.MISSING_DTYPE


def test_non_uniform_dtype(normal_meta):
    """
    触发 NON_UNIFORM_DTYPE
    - 不同帧 BitsStored / BitsAllocated / etc. 不一致
    """
    meta_copy = copy.deepcopy(normal_meta)
    num = meta_copy.num_datasets
    half = num // 2
    bits_stored_list = []
    bits_allocated_list = []
    high_bit_list = []
    pix_repr_list = []
    for i in range(num):
        if i < half:
            bits_stored_list.append(12)
            bits_allocated_list.append(16)
            high_bit_list.append(11)
            pix_repr_list.append(0)
        else:
            bits_stored_list.append(8)
            bits_allocated_list.append(8)
            high_bit_list.append(7)
            pix_repr_list.append(0)

    meta_copy.set_nonshared_item(CommonTags.BITS_STORED, bits_stored_list)
    meta_copy.set_nonshared_item(CommonTags.BITS_ALLOCATED, bits_allocated_list)
    meta_copy.set_nonshared_item(CommonTags.HIGH_BIT, high_bit_list)
    meta_copy.set_nonshared_item(CommonTags.PIXEL_REPRESENTATION, pix_repr_list)

    status = get_dicom_status(meta_copy)
    assert status == DicomStatus.NON_UNIFORM_DTYPE


def test_missing_location(normal_meta):
    """
    触发 MISSING_LOCATION
    - 既没有 IMAGE_POSITION_PATIENT, 也没有 SLICE_LOCATION
    """
    meta_copy = copy.deepcopy(normal_meta)
    meta_copy._merged_data.pop(CommonTags.IMAGE_POSITION_PATIENT.key)
    meta_copy._merged_data.pop(CommonTags.SLICE_LOCATION.key)
    status = get_dicom_status(meta_copy)
    assert status == DicomStatus.MISSING_LOCATION


def test_reversed_location(normal_meta):
    """
    触发 REVERSED_LOCATION
    - 让Z位置有正负混合排序(比如 [10,8,6,9,7] ), 代码检测到出现方向突变
    """
    meta_copy = copy.deepcopy(normal_meta)

    def mock_locations():
        num = meta_copy.num_datasets
        base = list(range(0, num))
        if num > 5:
            base[5] = 3
        return base

    meta_copy._get_projection_location = lambda: mock_locations()
    status = get_dicom_status(meta_copy)
    assert status == DicomStatus.REVERSED_LOCATION


def test_dwelling_location(normal_meta):
    """
    触发 DWELLING_LOCATION
    - 让 Z 值中有重复 => diffs_z == 0
    """
    meta_copy = copy.deepcopy(normal_meta)

    # 假设有5帧 => [1,2,3,3,4]
    # 如果帧数更大，可自行repeat这种停滞
    def mock_locations():
        num = meta_copy.num_datasets
        base = list(range(1, num + 1))
        if num >= 4:
            base[2] = base[1]  # 人为制造重复
        return base

    meta_copy._get_projection_location = lambda: mock_locations()

    status = get_dicom_status(meta_copy)
    assert status == DicomStatus.DWELLING_LOCATION


def test_gap_location(normal_meta):
    """
    触发 GAP_LOCATION
    - 让 Z 值有较大跳跃
    """
    meta_copy = copy.deepcopy(normal_meta)

    # 比如 [1,2,3,5,6] => diffs里出现超过平均* 1.5倍的跳跃
    def mock_locations():
        num = meta_copy.num_datasets
        base = list(range(1, num + 1))
        for i in range(4, num):
            base[i] += 1
        return base

    meta_copy._get_projection_location = lambda: mock_locations()

    status = get_dicom_status(meta_copy)
    assert status == DicomStatus.GAP_LOCATION


def test_non_uniform_rescale_factor(normal_meta):
    """
    触发 NON_UNIFORM_RESCALE_FACTOR
    - RescaleIntercept 或 RescaleSlope 不一致
    """
    meta_copy = copy.deepcopy(normal_meta)
    num = meta_copy.num_datasets
    if num < 2:
        pytest.skip("需要至少2帧才能测试非统一的Intercept/Slope")

    # 第一帧: Intercept=0, Slope=1; 第二帧: Intercept=10, Slope=2 ...
    # 只要保证有差异即可
    intercept_list = []
    slope_list = []
    for i in range(num):
        if i % 2 == 0:
            intercept_list.append(0)
            slope_list.append(1)
        else:
            intercept_list.append(10)
            slope_list.append(2)

    meta_copy.set_nonshared_item(CommonTags.RESCALE_INTERCEPT, intercept_list)
    meta_copy.set_nonshared_item(CommonTags.RESCALE_SLOPE, slope_list)

    status = get_dicom_status(meta_copy)
    assert status == DicomStatus.NON_UNIFORM_RESCALE_FACTOR 