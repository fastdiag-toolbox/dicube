import os
import tempfile

import numpy as np
import pytest

from dicube.core.image import DicomCubeImage
from dicube.core.pixel_header import PixelDataHeader


def test_dicom_cube_image_init():
    """
    测试 DicomCubeImage 的初始化及 get_fdata 功能
    """
    # 构造一个 10x10x10 的 3D 图像
    raw_data = np.arange(1000).reshape(10, 10, 10).astype(np.uint16)
    pixel_header = PixelDataHeader(
        RESCALE_SLOPE=2.0,
        RESCALE_INTERCEPT=-100.0,
        ORIGINAL_PIXEL_DTYPE="uint16",
        PIXEL_DTYPE="uint16",
    )
    img = DicomCubeImage(raw_data, pixel_header=pixel_header)
    assert img.raw_image.shape == (10, 10, 10)

    # 测试 get_fdata()
    fdata = img.get_fdata(dtype="float32")
    # slope=2, intercept=-100 => fdata = raw_data*2 -100
    assert fdata.shape == (10, 10, 10)
    assert abs(fdata[0, 0, 0] - (-100)) < 1e-3
    assert abs(fdata[-1, -1, -1] - (999 * 2 - 100)) < 1e-3


def test_dicom_cube_image_basic_operations():
    """
    测试 DicomCubeImage 的基本操作
    """
    raw_data = np.random.randint(0, 1000, size=(5, 64, 64), dtype=np.uint16)
    pixel_header = PixelDataHeader(
        RESCALE_SLOPE=1.0,
        RESCALE_INTERCEPT=-1024.0,
        ORIGINAL_PIXEL_DTYPE="uint16", 
        PIXEL_DTYPE="uint16",
        WINDOW_CENTER=50.0,
        WINDOW_WIDTH=400.0
    )
    
    img = DicomCubeImage(raw_data, pixel_header=pixel_header)
    
    # 测试形状属性
    assert img.shape == (5, 64, 64)
    
    # 测试元数据初始化
    img.init_meta(modality='CT', patient_name='TEST_PATIENT')
    assert img.dicom_meta is not None
    
    # 测试浮点数据获取
    fdata = img.get_fdata()
    assert fdata.shape == (5, 64, 64)
    assert fdata.dtype == np.float32


@pytest.mark.skipif(
    not os.path.exists("testdata/dicom/sample_150"),
    reason="Sample DICOM data not available"
)
def test_dicom_cube_image_from_dicom_folder():
    """
    测试从DICOM文件夹创建图像
    """
    folder_path = "testdata/dicom/sample_150"
    
    image = DicomCubeImage.from_dicom_folder(folder_path)
    assert image.raw_image.ndim == 3
    assert image.dicom_meta is not None
    assert image.pixel_header is not None
    
    # 测试数据一致性
    fdata = image.get_fdata()
    assert fdata.shape == image.raw_image.shape


def test_dicom_cube_image_to_dicom_folder():
    """
    测试 DicomCubeImage.to_dicom_folder() 功能
    """
    # 构造一个简单的 DicomCubeImage
    raw_data = np.random.randint(0, 1000, size=(5, 128, 128), dtype=np.uint16)
    pixel_header = PixelDataHeader(
        RESCALE_SLOPE=1.0,
        RESCALE_INTERCEPT=-1024.0,
        ORIGINAL_PIXEL_DTYPE="uint16",
        PIXEL_DTYPE="uint16"
    )
    
    image = DicomCubeImage(raw_data, pixel_header=pixel_header)
    image.init_meta(modality='CT', patient_name='TEST_PATIENT')
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = os.path.join(temp_dir, "test_output")
        
        # 测试无压缩输出
        image.to_dicom_folder(output_dir=output_dir)
        assert os.path.exists(output_dir)
        
        dicom_files = os.listdir(output_dir)
        assert len(dicom_files) == 5  # 5帧图像
        
        # 读回验证
        image_back = DicomCubeImage.from_dicom_folder(output_dir)
        assert image_back.shape == image.shape
        assert np.array_equal(image_back.raw_image, image.raw_image)


def test_dicom_cube_image_metadata():
    """
    测试DicomCubeImage的元数据功能
    """
    raw_data = np.random.randint(0, 1000, size=(3, 64, 64), dtype=np.uint16)
    pixel_header = PixelDataHeader(
        RESCALE_SLOPE=1.0,
        RESCALE_INTERCEPT=0.0,
        ORIGINAL_PIXEL_DTYPE="uint16",
        PIXEL_DTYPE="uint16"
    )
    
    image = DicomCubeImage(raw_data, pixel_header=pixel_header)
    
    # 测试无元数据时的init_meta
    assert image.dicom_meta is None
    image.init_meta(modality='MR', patient_name='TEST_MR_PATIENT', patient_id='12345')
    assert image.dicom_meta is not None
    
    # 验证元数据内容
    from dicube.dicom.dicom_tags import CommonTags
    patient_name = image.dicom_meta.get(CommonTags.PATIENT_NAME)
    modality = image.dicom_meta.get(CommonTags.MODALITY)
    
    assert patient_name is not None
    assert modality is not None


def test_pixel_header_validation():
    """
    测试 PixelDataHeader 的验证功能
    """
    # 测试基本的 PixelDataHeader 创建
    header = PixelDataHeader(
        RESCALE_SLOPE=2.0,
        RESCALE_INTERCEPT=-100.0,
        ORIGINAL_PIXEL_DTYPE="uint16",
        PIXEL_DTYPE="uint16",
        WINDOW_CENTER=50.0,
        WINDOW_WIDTH=400.0
    )
    
    assert header.RESCALE_SLOPE == 2.0
    assert header.RESCALE_INTERCEPT == -100.0
    assert header.WINDOW_CENTER == 50.0
    assert header.WINDOW_WIDTH == 400.0 