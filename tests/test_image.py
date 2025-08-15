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
        RescaleSlope=2.0,
        RescaleIntercept=-100.0,
        OriginalPixelDtype="uint16",
        PixelDtype="uint16",
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
        RescaleSlope=1.0,
        RescaleIntercept=-1024.0,
        OriginalPixelDtype="uint16", 
        PixelDtype="uint16",
        WindowCenter=50.0,
        WindowWidth=400.0
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
        RescaleSlope=1.0,
        RescaleIntercept=-1024.0,
        OriginalPixelDtype="uint16",
        PixelDtype="uint16"
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
        RescaleSlope=1.0,
        RescaleIntercept=0.0,
        OriginalPixelDtype="uint16",
        PixelDtype="uint16"
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
    测试 PixelDataHeader 的创建和验证
    """
    header = PixelDataHeader(
        RescaleSlope=2.0,
        RescaleIntercept=-100.0,
        OriginalPixelDtype="uint16",
        PixelDtype="uint16",
        WindowCenter=50.0,
        WindowWidth=400.0
    )
    
    assert header.RescaleSlope == 2.0
    assert header.RescaleIntercept == -100.0
    assert header.WindowCenter == 50.0
    assert header.WindowWidth == 400.0


def test_dicom_cube_image_with_space():
    """
    测试带有 Space 的 DicomCubeImage 创建和验证
    """
    from spacetransformer import Space
    
    # 创建测试数据 - 内部格式 (z,y,x)
    raw_data = np.random.randint(0, 1000, size=(10, 20, 30), dtype=np.uint16)
    pixel_header = PixelDataHeader(
        RescaleSlope=1.0,
        RescaleIntercept=0.0,
        OriginalPixelDtype="uint16",
        PixelDtype="uint16"
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
        RescaleSlope=1.0,
        RescaleIntercept=0.0,
        OriginalPixelDtype="uint16",
        PixelDtype="uint16"
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
        RescaleSlope=1.0,
        RescaleIntercept=0.0,
        OriginalPixelDtype="uint16",
        PixelDtype="uint16"
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
        RescaleSlope=1.0,
        RescaleIntercept=0.0,
        OriginalPixelDtype="uint16",
        PixelDtype="uint16"
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
        RescaleSlope=0.5,
        RescaleIntercept=100.0,
        OriginalPixelDtype="uint16",
        PixelDtype="uint16"
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


def test_nifti_round_trip_consistency():
    """
    测试NIfTI文件的循环一致性 - 读入、写出、再读入，检查pixel array和affine一致性
    """
    try:
        import nibabel as nib
    except ImportError:
        pytest.skip("nibabel not installed, skipping NIfTI test")
    
    import tempfile
    from spacetransformer import Space
    
    # 创建测试数据 - 内部格式 (z,y,x)
    raw_data = np.random.randint(0, 1000, size=(10, 20, 30), dtype=np.uint16)
    pixel_header = PixelDataHeader(
        RescaleSlope=1.0,
        RescaleIntercept=0.0,
        OriginalPixelDtype="uint16",
        PixelDtype="uint16"
    )
    
    # 创建 Space - 内部格式 (z,y,x)
    test_space = Space(
        shape=(10, 20, 30),
        origin=(5.0, 10.0, 15.0),  # 非零原点以便测试
        spacing=(1.5, 0.8, 0.5),   # 非均匀间距以便测试
        x_orientation=(1.0, 0.0, 0.0),
        y_orientation=(0.0, 1.0, 0.0),
        z_orientation=(0.0, 0.0, 1.0)
    )
    
    # 创建 DicomCubeImage
    original_image = DicomCubeImage(raw_data, pixel_header, space=test_space)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # 输出路径
        nifti_path = os.path.join(temp_dir, "test.nii.gz")
        
        # 保存为NIfTI
        dicube.save_to_nifti(original_image, nifti_path)
        assert os.path.exists(nifti_path), "NIfTI file was not created"
        
        # 从NIfTI读取
        loaded_image = dicube.load_from_nifti(nifti_path)
        
        # 保存到另一个NIfTI文件
        nifti_path2 = os.path.join(temp_dir, "test2.nii.gz")
        dicube.save_to_nifti(loaded_image, nifti_path2)
        
        # 再次读取
        final_image = dicube.load_from_nifti(nifti_path2)
        
        # 验证像素数据一致性
        assert original_image.raw_image.shape == final_image.raw_image.shape, "Shape mismatch"
        assert np.array_equal(original_image.raw_image, final_image.raw_image), "Pixel data mismatch"
        
        # 验证空间信息一致性
        assert original_image.space.shape == final_image.space.shape, "Space shape mismatch"
        assert np.allclose(original_image.space.origin, final_image.space.origin, atol=1e-5), "Origin mismatch"
        assert np.allclose(original_image.space.spacing, final_image.space.spacing, atol=1e-5), "Spacing mismatch"
        assert np.allclose(original_image.space.to_nifti_affine(), final_image.space.to_nifti_affine(), atol=1e-5), "Affine mismatch"
        
        # 直接使用nibabel比较affine
        nib1 = nib.load(nifti_path)
        nib2 = nib.load(nifti_path2)
        assert np.allclose(nib1.affine, nib2.affine, atol=1e-5), "NiBabel affine mismatch"


@pytest.mark.skipif(
    not os.path.exists("testdata/nifti/s0000.nii.gz"),
    reason="Sample NIfTI data not available"
)
def test_nifti_real_data_round_trip():
    """
    使用实际的NIfTI测试文件测试循环一致性
    
    读取 testdata/nifti/s0000.nii.gz，写出到临时文件，再读回来验证一致性
    """
    try:
        import nibabel as nib
    except ImportError:
        pytest.skip("nibabel not installed, skipping NIfTI test")
    
    # 测试文件路径
    nifti_test_file = "testdata/nifti/s0000.nii.gz"
    
    # 从测试文件读取
    original_image = dicube.load_from_nifti(nifti_test_file)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # 输出到临时文件
        nifti_path = os.path.join(temp_dir, "real_data_test.nii.gz")
        
        # 保存到新的NIfTI文件
        dicube.save_to_nifti(original_image, nifti_path)
        assert os.path.exists(nifti_path), "NIfTI file was not created"
        
        # 再次读取
        loaded_image = dicube.load_from_nifti(nifti_path)
        
        # 保存到第三个文件以测试完整循环
        nifti_path2 = os.path.join(temp_dir, "real_data_test2.nii.gz")
        dicube.save_to_nifti(loaded_image, nifti_path2)
        
        # 再次读取
        final_image = dicube.load_from_nifti(nifti_path2)
        
        # 验证像素数据一致性
        assert original_image.raw_image.shape == final_image.raw_image.shape, "Shape mismatch"
        assert np.array_equal(original_image.raw_image, final_image.raw_image), "Pixel data mismatch"
        
        # 验证空间信息一致性
        assert original_image.space.shape == final_image.space.shape, "Space shape mismatch"
        assert np.allclose(original_image.space.origin, final_image.space.origin, atol=1e-5), "Origin mismatch"
        assert np.allclose(original_image.space.spacing, final_image.space.spacing, atol=1e-5), "Spacing mismatch"
        assert np.allclose(original_image.space.to_nifti_affine(), final_image.space.to_nifti_affine(), atol=1e-5), "Affine mismatch"
        
        # 直接使用nibabel比较原始文件和最终文件的affine
        nib_orig = nib.load(nifti_test_file)
        nib_final = nib.load(nifti_path2)
        assert np.allclose(nib_orig.affine, nib_final.affine, atol=1e-5), "NiBabel affine mismatch"
        assert np.allclose(nib_orig.get_fdata(), nib_final.get_fdata(), atol=1e-5), "NiBabel data mismatch"
        
        
        # 确认循环操作保持数据的一致性
        assert np.array_equal(original_image.raw_image, final_image.raw_image), "DiCube round-trip data consistency failed" 


def test_determine_optimal_nifti_dtype():
    """
    测试determine_optimal_nifti_dtype函数的数据类型选择逻辑
    """
    from dicube.storage.pixel_utils import determine_optimal_nifti_dtype
    
    # 测试不同类型的数据
    
    # 1. 小范围正整数 (应该选择uint8)
    raw_data = np.array([0, 10, 20, 255], dtype=np.uint16)
    header = PixelDataHeader(
        RescaleSlope=1.0,
        RescaleIntercept=0.0,
        OriginalPixelDtype="uint16",
        PixelDtype="uint16"
    )
    data, dtype_name = determine_optimal_nifti_dtype(raw_data, header)
    assert dtype_name == "uint8", "小范围正整数应该使用uint8"
    assert data.dtype == np.uint8, "数据类型应该是uint8"
    
    # 2. 中等范围正整数 (应该选择uint16)
    raw_data = np.array([0, 1000, 50000, 65535], dtype=np.uint16)
    header = PixelDataHeader(
        RescaleSlope=1.0,
        RescaleIntercept=0.0,
        OriginalPixelDtype="uint16",
        PixelDtype="uint16"
    )
    data, dtype_name = determine_optimal_nifti_dtype(raw_data, header)
    assert dtype_name == "uint16", "中等范围正整数应该使用uint16"
    assert data.dtype == np.uint16, "数据类型应该是uint16"
    
    # 3. 有负值的整数 (应该选择int16)
    raw_data = np.array([100, 200, 300], dtype=np.uint16)
    header = PixelDataHeader(
        RescaleSlope=1.0,
        RescaleIntercept=-1000.0,  # 应用后会有负值
        OriginalPixelDtype="uint16",
        PixelDtype="uint16"
    )
    data, dtype_name = determine_optimal_nifti_dtype(raw_data, header)
    assert dtype_name == "int16", "带负值的数据应该使用int16"
    assert data.dtype == np.int16, "数据类型应该是int16"
    assert np.allclose(data, np.array([-900, -800, -700], dtype=np.int16)), "应用重缩放后的数据不正确"
    
    # 4. 浮点数据 (应该选择float32)
    raw_data = np.array([100, 200, 300], dtype=np.uint16)
    header = PixelDataHeader(
        RescaleSlope=0.5,  # 非整数斜率
        RescaleIntercept=0.0,
        OriginalPixelDtype="uint16",
        PixelDtype="uint16"
    )
    data, dtype_name = determine_optimal_nifti_dtype(raw_data, header)
    assert dtype_name == "float32", "非整数值应该使用float32"
    assert data.dtype == np.float32, "数据类型应该是float32"
    assert np.allclose(data, np.array([50.0, 100.0, 150.0], dtype=np.float32)), "应用重缩放后的浮点数据不正确"
    
    # 5. 大数值范围 (应该选择int32)
    raw_data = np.array([1000000, 2000000, 3000000], dtype=np.int32)
    header = PixelDataHeader(
        RescaleSlope=1.0,
        RescaleIntercept=0.0,
        OriginalPixelDtype="int32",
        PixelDtype="int32"
    )
    data, dtype_name = determine_optimal_nifti_dtype(raw_data, header)
    assert dtype_name == "int32", "大范围整数应该使用int32"
    assert data.dtype == np.int32, "数据类型应该是int32" 