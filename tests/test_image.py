import os
import tempfile

import numpy as np
import pytest

import dicube
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
    img.init_meta(modality='CT', patient_name='TEST^PATIENT')
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
    
    image = dicube.load_from_dicom_folder(folder_path)
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
    image.init_meta(modality='CT', patient_name='TEST^PATIENT')
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = os.path.join(temp_dir, "test_output")
        
        # 测试无压缩输出
        dicube.save_to_dicom_folder(image, output_dir)
        assert os.path.exists(output_dir)
        
        dicom_files = os.listdir(output_dir)
        assert len(dicom_files) == 5  # 5帧图像
        
        # 读回验证
        image_back = dicube.load_from_dicom_folder(output_dir)
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
    patient_name = image.dicom_meta.get_shared_value(CommonTags.PatientName)
    modality = image.dicom_meta.get_shared_value(CommonTags.Modality)
    
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


def test_dicom_cube_image_with_space():
    """
    测试带有 Space 的 DicomCubeImage 创建和验证
    """
    from spacetransformer import Space
    
    # 创建测试数据 - 内部格式 (z,y,x)
    raw_data = np.random.randint(0, 1000, size=(10, 20, 30), dtype=np.uint16)
    pixel_header = PixelDataHeader(
        RESCALE_SLOPE=1.0,
        RESCALE_INTERCEPT=0.0,
        ORIGINAL_PIXEL_DTYPE="uint16",
        PIXEL_DTYPE="uint16"
    )
    
    # 创建 Space - 内部格式 (z,y,x)
    test_space = Space(
        shape=(10, 20, 30),
        origin=(0.0, 0.0, 0.0),
        spacing=(1.0, 1.0, 1.0),
        x_orientation=(1.0, 0.0, 0.0),
        y_orientation=(0.0, 1.0, 0.0),
        z_orientation=(0.0, 0.0, 1.0)
    )
    
    # 创建 DicomCubeImage
    image = DicomCubeImage(raw_data, pixel_header, space=test_space)
    
    # 验证内部一致性
    assert image.raw_image.shape == (10, 20, 30)
    assert image.space.shape == (10, 20, 30)
    
    # 验证 _validate_shape 不会抛出异常
    image._validate_shape()


def test_dicom_cube_image_space_mismatch():
    """
    测试 DicomCubeImage 在 Space 和数组形状不匹配时的异常处理
    """
    from spacetransformer import Space
    
    # 创建不匹配的测试数据
    raw_data = np.random.randint(0, 1000, size=(10, 20, 30), dtype=np.uint16)
    pixel_header = PixelDataHeader(
        RESCALE_SLOPE=1.0,
        RESCALE_INTERCEPT=0.0,
        ORIGINAL_PIXEL_DTYPE="uint16",
        PIXEL_DTYPE="uint16"
    )
    
    # 创建不匹配的 Space
    wrong_space = Space(
        shape=(15, 25, 35),  # 不匹配的形状
        origin=(0.0, 0.0, 0.0),
        spacing=(1.0, 1.0, 1.0)
    )
    
    # 应该抛出 DataConsistencyError
    from dicube.exceptions import DataConsistencyError
    with pytest.raises(DataConsistencyError, match="Space shape mismatch with image"):
        DicomCubeImage(raw_data, pixel_header, space=wrong_space)


def test_dicom_cube_image_space_coordinate_conversion():
    """
    测试 DicomCubeImage 的 Space 坐标系转换功能
    验证文件 I/O 过程中的坐标系转换正确性
    """
    from spacetransformer import Space
    
    # 创建测试数据 - 内部格式 (z,y,x)
    raw_data = np.random.randint(0, 1000, size=(8, 16, 24), dtype=np.uint16)
    pixel_header = PixelDataHeader(
        RESCALE_SLOPE=1.0,
        RESCALE_INTERCEPT=0.0,
        ORIGINAL_PIXEL_DTYPE="uint16",
        PIXEL_DTYPE="uint16"
    )
    
    # 创建 Space - 内部格式 (z,y,x)
    original_space = Space(
        shape=(8, 16, 24),
        origin=(1.0, 2.0, 3.0),
        spacing=(0.5, 0.8, 1.2),
        x_orientation=(1.0, 0.0, 0.0),
        y_orientation=(0.0, 1.0, 0.0),
        z_orientation=(0.0, 0.0, 1.0)
    )
    
    # 创建原始图像
    original_image = DicomCubeImage(raw_data, pixel_header, space=original_space)
    
    # 测试文件 I/O
    with tempfile.NamedTemporaryFile(suffix='.dcbs', delete=False) as tmp_file:
        temp_filename = tmp_file.name
    
    try:
        # 写入文件
        dicube.save(original_image, temp_filename, file_type='s')
        
        # 从文件读取
        loaded_image = dicube.load(temp_filename)
        
        # 验证数据一致性
        assert loaded_image.raw_image.shape == original_image.raw_image.shape
        assert loaded_image.space.shape == original_image.space.shape
        
        # 验证数组数据完全一致
        assert np.array_equal(loaded_image.raw_image, original_image.raw_image)
        
        # 验证 Space 属性一致
        assert loaded_image.space.origin == original_image.space.origin
        assert loaded_image.space.spacing == original_image.space.spacing
        assert loaded_image.space.x_orientation == original_image.space.x_orientation
        assert loaded_image.space.y_orientation == original_image.space.y_orientation
        assert loaded_image.space.z_orientation == original_image.space.z_orientation
        
    finally:
        # 清理临时文件
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)


def test_dcb_file_space_conversion():
    """
    测试 DCBFile 的 Space 坐标系转换功能
    直接测试 DCBFile 的读写过程中的坐标系转换
    """
    from spacetransformer import Space
    from dicube.storage.dcb_file import DcbSFile
    
    # 创建测试数据
    raw_data = np.random.randint(0, 1000, size=(6, 12, 18), dtype=np.uint16)
    pixel_header = PixelDataHeader(
        RESCALE_SLOPE=1.0,
        RESCALE_INTERCEPT=0.0,
        ORIGINAL_PIXEL_DTYPE="uint16",
        PIXEL_DTYPE="uint16"
    )
    
    # 创建 Space - 内部格式 (z,y,x)
    internal_space = Space(
        shape=(6, 12, 18),
        origin=(10.0, 20.0, 30.0),
        spacing=(2.0, 3.0, 4.0),
        x_orientation=(1.0, 0.0, 0.0),
        y_orientation=(0.0, 1.0, 0.0),
        z_orientation=(0.0, 0.0, 1.0)
    )
    
    with tempfile.NamedTemporaryFile(suffix='.dcbs', delete=False) as tmp_file:
        temp_filename = tmp_file.name
    
    try:
        # 写入文件
        writer = DcbSFile(temp_filename, mode='w')
        writer.write(
            images=[raw_data[i] for i in range(raw_data.shape[0])],
            pixel_header=pixel_header,
            space=internal_space
        )
        
        # 读取文件
        reader = DcbSFile(temp_filename, mode='r')
        loaded_space = reader.read_space()
        loaded_images = reader.read_images()
        
        # 验证 Space 转换正确性
        assert loaded_space.shape == internal_space.shape
        assert loaded_space.origin == internal_space.origin
        assert loaded_space.spacing == internal_space.spacing
        assert loaded_space.x_orientation == internal_space.x_orientation
        assert loaded_space.y_orientation == internal_space.y_orientation
        assert loaded_space.z_orientation == internal_space.z_orientation
        
        # 验证图像数据一致性
        assert isinstance(loaded_images, list)
        loaded_images = np.stack(loaded_images, axis=0)
        assert loaded_images.shape == raw_data.shape
        assert np.array_equal(loaded_images, raw_data)
        
    finally:
        # 清理临时文件
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)


def test_dicom_cube_image_space_round_trip():
    """
    测试 DicomCubeImage 在多次保存和读取过程中的数据一致性
    """
    from spacetransformer import Space
    
    # 创建复杂的测试数据
    raw_data = np.random.randint(0, 1000, size=(5, 10, 15), dtype=np.uint16)
    pixel_header = PixelDataHeader(
        RESCALE_SLOPE=0.5,
        RESCALE_INTERCEPT=100.0,
        ORIGINAL_PIXEL_DTYPE="uint16",
        PIXEL_DTYPE="uint16"
    )
    
    # 创建非标准的 Space
    original_space = Space(
        shape=(5, 10, 15),
        origin=(-10.0, -20.0, -30.0),
        spacing=(0.25, 0.5, 0.75),
        x_orientation=(0.8, 0.6, 0.0),
        y_orientation=(-0.6, 0.8, 0.0),
        z_orientation=(0.0, 0.0, 1.0)
    )
    
    # 创建原始图像
    original_image = DicomCubeImage(raw_data, pixel_header, space=original_space)
    
    # 多次保存和读取
    for i in range(3):
        with tempfile.NamedTemporaryFile(suffix='.dcbs', delete=False) as tmp_file:
            temp_filename = tmp_file.name
        
        try:
            # 保存
            dicube.save(original_image, temp_filename, file_type='s')
            
            # 读取
            loaded_image = dicube.load(temp_filename)
            
            # 验证数据完全一致
            assert np.array_equal(loaded_image.raw_image, original_image.raw_image)
            assert loaded_image.space.shape == original_image.space.shape
            assert np.allclose(loaded_image.space.origin, original_image.space.origin)
            assert np.allclose(loaded_image.space.spacing, original_image.space.spacing)
            assert np.allclose(loaded_image.space.x_orientation, original_image.space.x_orientation)
            assert np.allclose(loaded_image.space.y_orientation, original_image.space.y_orientation)
            assert np.allclose(loaded_image.space.z_orientation, original_image.space.z_orientation)
            
            # 用读取的图像作为下一轮的原始图像
            original_image = loaded_image
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_filename):
                os.unlink(temp_filename) 