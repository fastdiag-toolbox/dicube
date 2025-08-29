"""
Tests for negative value support and data type optimization in DiCube.

These tests verify the new features introduced for handling negative values
without unnecessary intercept offsets, and intelligent data type selection.
"""

import os
import tempfile
import warnings

import numpy as np
import pytest

import dicube
from dicube.storage.pixel_utils import (
    derive_pixel_header_from_array, 
    is_lossless_int_convertible, 
    convert_to_minimal_int
)
from dicube.core.pixel_header import PixelDataHeader
from dicube.core.image import DicomCubeImage


class TestNegativeSupport:
    """Test negative value support in pixel data processing."""
    
    def test_signed_integer_direct_storage(self):
        """测试有符号整数的直接存储（无 intercept 偏移）"""
        test_cases = [
            (np.int8, np.array([[[-128, -1, 0, 1, 127]]], dtype=np.int8)),
            (np.int16, np.array([[[-32768, -1000, 0, 1000, 32767]]], dtype=np.int16)),
            (np.int32, np.array([[[-100000, -1000, 0, 1000, 100000]]], dtype=np.int32)),
        ]
        
        for expected_dtype, test_data in test_cases:
            result_array, header = derive_pixel_header_from_array(test_data, support_negative=True)
            
            # 应该保持原始类型，无偏移
            assert result_array.dtype == expected_dtype
            assert header.RescaleSlope == 1.0
            assert header.RescaleIntercept == 0.0
            assert header.PixelDtype == expected_dtype.__name__
            assert np.array_equal(result_array, test_data)
    
    def test_legacy_mode_compatibility(self):
        """测试向后兼容模式（使用 intercept 偏移）"""
        test_data = np.array([[[-1024, -500, 0, 500, 1000]]], dtype=np.int16)
        result_array, header = derive_pixel_header_from_array(test_data, support_negative=False)
        
        # 应该转换为 unsigned 类型并记录偏移
        assert result_array.dtype == np.uint16
        assert header.RescaleSlope == 1
        assert header.RescaleIntercept == -1024  # 最小值作为偏移
        assert header.PixelDtype == "uint16"
        
        # 验证数据转换正确性
        expected = test_data - (-1024)  # 减去最小值
        assert np.array_equal(result_array, expected.astype(np.uint16))
    
    def test_lossless_integer_conversion(self):
        """测试浮点数的无损整数转换"""
        # 可以无损转换的浮点数据
        lossless_cases = [
            (np.array([[[-10.0, 0.0, 100.0]]], dtype=np.float32), np.int8),
            (np.array([[[0.0, 128.0, 255.0]]], dtype=np.float32), np.uint8),
            (np.array([[[-1000.0, 0.0, 30000.0]]], dtype=np.float32), np.int16),
            (np.array([[[0.0, 32768.0, 65535.0]]], dtype=np.float32), np.uint16),
        ]
        
        for test_data, expected_dtype in lossless_cases:
            # 检查是否可以无损转换
            assert is_lossless_int_convertible(test_data)
            
            # 执行转换
            int_array, dtype_name = convert_to_minimal_int(test_data)
            assert int_array.dtype == expected_dtype
            assert dtype_name == expected_dtype.__name__
            
            # 通过 derive_pixel_header_from_array 测试
            result_array, header = derive_pixel_header_from_array(test_data)
            assert result_array.dtype == expected_dtype
            assert header.RescaleSlope == 1.0
            assert header.RescaleIntercept == 0.0
    
    def test_float_with_slope_mechanism(self):
        """测试需要使用 slope 机制的浮点数据"""
        # 真正的浮点数据（不能无损转换）
        float_cases = [
            np.array([[[0.1, 0.5, 0.9]]], dtype=np.float32),
            np.array([[[1.23, 4.56, 7.89]]], dtype=np.float64),
            np.array([[[-0.5, 0.0, 0.5]]], dtype=np.float32),
        ]
        
        for test_data in float_cases:
            # 检查不能无损转换
            assert not is_lossless_int_convertible(test_data)
            
            # 应该使用 slope 机制
            result_array, header = derive_pixel_header_from_array(test_data)
            assert header.RescaleSlope != 1.0 or header.RescaleIntercept != 0.0
            
            # 验证往返转换的准确性（放宽精度要求）
            reconstructed = result_array.astype(np.float32) * header.RescaleSlope + header.RescaleIntercept
            assert np.allclose(reconstructed, test_data, rtol=1e-3, atol=1e-5)
    
    def test_extreme_value_ranges(self):
        """测试极端值域情况"""
        extreme_cases = [
            # 测试各种数据类型的边界值
            np.array([[[-128, 127]]], dtype=np.int8),
            np.array([[[0, 255]]], dtype=np.uint8),
            np.array([[[-32768, 32767]]], dtype=np.int16),
            np.array([[[0, 65535]]], dtype=np.uint16),
            
            # 测试需要升级数据类型的情况
            np.array([[[0.0, 300.0]]], dtype=np.float32),  # 需要 uint16
            np.array([[[-200.0, 200.0]]], dtype=np.float32),  # 需要 int16
        ]
        
        for test_data in extreme_cases:
            result_array, header = derive_pixel_header_from_array(test_data)
            
            # 验证没有数据溢出
            if header.RescaleSlope == 1.0 and header.RescaleIntercept == 0.0:
                # 直接存储，应该完全相等
                assert np.array_equal(result_array, test_data)
            else:
                # 使用 slope 机制，验证往返准确性
                reconstructed = result_array.astype(np.float32) * header.RescaleSlope + header.RescaleIntercept
                assert np.allclose(reconstructed, test_data, rtol=1e-5)
    
    def test_constant_array_handling(self):
        """测试常数数组的特殊处理"""
        # 常数数组应该特殊处理
        const_cases = [
            np.array([[[5.0, 5.0, 5.0]]], dtype=np.float32),
            np.array([[[-100.0, -100.0, -100.0]]], dtype=np.float32),
            np.array([[[0.0, 0.0, 0.0]]], dtype=np.float32),
        ]
        
        for test_data in const_cases:
            result_array, header = derive_pixel_header_from_array(test_data)
            
            # 验证常数数组可以正确重建
            unique_vals = np.unique(test_data)
            if len(unique_vals) == 1:
                # 验证重建后的值正确
                reconstructed = result_array.astype(np.float32) * header.RescaleSlope + header.RescaleIntercept
                assert np.allclose(reconstructed, test_data, rtol=1e-6)


class TestDicomRoundtrip:
    """Test DICOM roundtrip conversion with negative values."""
    
    def test_negative_dicom_roundtrip(self):
        """测试包含负数的 DICOM 往返转换"""
        # 创建包含负数的测试数据
        test_data = np.array([
            [[-1024, -500, 0], [500, 1000, 2000]],
            [[-800, -200, 100], [800, 1500, 2500]]
        ], dtype=np.int16)
        
        pixel_header = PixelDataHeader(
            RescaleSlope=1.0,
            RescaleIntercept=0.0,
            OriginalPixelDtype="int16",
            PixelDtype="int16"
        )
        
        # 创建 DicomCubeImage
        image = DicomCubeImage(test_data, pixel_header)
        image.init_meta(modality='CT', patient_name='TEST^NEGATIVE')
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 保存为 DICOM 文件夹
            dicom_dir = os.path.join(temp_dir, "dicom_out")
            dicube.save_to_dicom_folder(image, dicom_dir)
            
            # 验证 DICOM 文件存在
            dicom_files = [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
            assert len(dicom_files) == 2  # 2 个切片
            
            # 读回并验证
            loaded_image = dicube.load_from_dicom_folder(dicom_dir)
            
            # 比较真实值（考虑到存储优化可能改变数据类型）
            original_real_values = image.get_fdata()
            loaded_real_values = loaded_image.get_fdata()
            
            assert np.allclose(original_real_values, loaded_real_values, rtol=1e-6)
    
    @pytest.mark.skipif(
        not os.path.exists("testdata/dicom/sample_150"),
        reason="Sample DICOM data not available"
    )
    def test_pydicom_compatibility(self):
        """测试生成的 DICOM 文件与 PyDicom 的兼容性"""
        import pydicom
        
        # 使用真实的 DICOM 数据测试
        image = dicube.load_from_dicom_folder("testdata/dicom/sample_150")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 保存为 DICOM 文件夹
            dicom_dir = os.path.join(temp_dir, "pydicom_test")
            dicube.save_to_dicom_folder(image, dicom_dir)
            
            # 使用 PyDicom 读取每个文件
            dicom_files = [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
            
            for dcm_file in dicom_files:
                file_path = os.path.join(dicom_dir, dcm_file)
                
                # PyDicom 应该能够正常读取
                ds = pydicom.dcmread(file_path)
                
                # 验证关键属性存在且合理
                assert hasattr(ds, 'PixelData')
                assert hasattr(ds, 'Rows') and ds.Rows > 0
                assert hasattr(ds, 'Columns') and ds.Columns > 0
                assert hasattr(ds, 'BitsAllocated') and ds.BitsAllocated in [8, 16, 32]
                assert hasattr(ds, 'PixelRepresentation') and ds.PixelRepresentation in [0, 1]
                
                # 验证像素数据可以正常解析
                pixel_array = ds.pixel_array
                assert pixel_array is not None
                assert pixel_array.shape == (ds.Rows, ds.Columns)
                
                # 验证 PixelRepresentation 与数据类型的一致性
                if ds.PixelRepresentation == 1:
                    assert np.issubdtype(pixel_array.dtype, np.signedinteger)
                else:
                    assert np.issubdtype(pixel_array.dtype, np.unsignedinteger)
                
                # 验证 RescaleSlope 和 RescaleIntercept
                slope = getattr(ds, 'RescaleSlope', 1.0)
                intercept = getattr(ds, 'RescaleIntercept', 0.0)
                
                # 计算真实值
                real_values = pixel_array.astype(np.float64) * float(slope) + float(intercept)
                
                # 验证真实值在合理范围内（CT 值通常在 -1024 到 3071 之间）
                assert real_values.min() >= -2000  # 允许一些余量
                assert real_values.max() <= 5000   # 允许一些余量


class TestDcbRoundtrip:
    """Test DCB file roundtrip with various data types."""
    
    @pytest.mark.parametrize("dtype,test_range", [
        (np.int8, (-128, 127)),
        (np.uint8, (0, 255)),
        (np.int16, (-32768, 32767)),
        (np.uint16, (0, 65535)),
        # 注意：int32/uint32 的完整范围太大，使用较小的测试范围
        (np.int32, (-100000, 100000)),
        (np.uint32, (0, 200000)),
    ])
    def test_dcb_roundtrip_all_types(self, dtype, test_range):
        """测试所有支持的数据类型的 DCB 往返转换"""
        min_val, max_val = test_range
        
        # 创建测试数据，包含边界值和中间值
        test_values = [min_val, min_val//2, 0, max_val//2, max_val]
        # 过滤掉超出当前类型范围的值
        if dtype == np.uint8 or dtype == np.uint16 or dtype == np.uint32:
            test_values = [v for v in test_values if v >= 0]
        
        test_data = np.array([test_values[:3], test_values[2:]], dtype=dtype)
        test_data = test_data.reshape(1, 2, 3)  # 确保是 3D
        
        # 处理数据
        result_array, header = derive_pixel_header_from_array(test_data)
        image = DicomCubeImage(result_array, header)
        
        with tempfile.NamedTemporaryFile(suffix='.dcbs', delete=False) as tmp_file:
            temp_filename = tmp_file.name
        
        try:
            # 保存为 DCB
            dicube.save(image, temp_filename, file_type='s')
            
            # 读回
            loaded_image = dicube.load(temp_filename)
            
            # 验证数据完整性（处理可能的形状变化）
            if result_array.shape != loaded_image.raw_image.shape:
                assert np.array_equal(result_array.flatten(), loaded_image.raw_image.flatten())
            else:
                assert np.array_equal(result_array, loaded_image.raw_image)
            
            # 验证真实值完全一致
            original_fdata = image.get_fdata()
            loaded_fdata = loaded_image.get_fdata()
            assert np.allclose(original_fdata, loaded_fdata, rtol=1e-10)
            
        finally:
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
    
    def test_mixed_positive_negative_values(self):
        """测试混合正负值的复杂场景"""
        # 创建包含正负值混合的复杂数据
        test_data = np.array([
            [[-1024, -512, -1], [0, 1, 512]],
            [[1024, 2048, 3071], [-2048, -1536, -1024]]
        ], dtype=np.int16)
        
        result_array, header = derive_pixel_header_from_array(test_data)
        
        # 应该保持为 int16 类型
        assert result_array.dtype == np.int16
        assert header.RescaleSlope == 1.0
        assert header.RescaleIntercept == 0.0
        
        # 创建图像并测试完整流程
        image = DicomCubeImage(result_array, header)
        image.init_meta(modality='CT')
        
        # 测试 DCB 往返
        with tempfile.NamedTemporaryFile(suffix='.dcbs', delete=False) as tmp_file:
            temp_filename = tmp_file.name
        
        try:
            dicube.save(image, temp_filename, file_type='s')
            loaded_image = dicube.load(temp_filename)
            
            assert np.allclose(image.get_fdata(), loaded_image.get_fdata(), rtol=1e-10)
            
        finally:
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
        
        # 测试 DICOM 往返
        with tempfile.TemporaryDirectory() as temp_dir:
            dicom_dir = os.path.join(temp_dir, "mixed_values")
            dicube.save_to_dicom_folder(image, dicom_dir)
            loaded_image = dicube.load_from_dicom_folder(dicom_dir)
            
            assert np.allclose(image.get_fdata(), loaded_image.get_fdata(), rtol=1e-6)


class TestDataTypeOptimization:
    """Test intelligent data type selection and optimization."""
    
    def test_minimal_type_selection(self):
        """测试最小数据类型的智能选择"""
                    # 测试不同值域应该选择的最小类型
        test_cases = [
            # (数据, 期望类型)
            (np.array([[[0, 100, 200]]], dtype=np.float32), np.uint8),  # 0-255 范围
            (np.array([[[-50, 0, 50]]], dtype=np.float32), np.int8),    # -128 到 127 范围
            (np.array([[[0, 1000, 30000]]], dtype=np.float32), np.uint16), # 需要 uint16
            (np.array([[[-1000, 0, 1000]]], dtype=np.float32), np.int16),  # 需要 int16
        ]
        
        for test_data, expected_dtype in test_cases:
            if is_lossless_int_convertible(test_data):
                int_array, dtype_name = convert_to_minimal_int(test_data)
                assert int_array.dtype == expected_dtype
    
    def test_type_selection_priority(self):
        """测试数据类型选择的优先级"""
        # 对于非负值，应该优先选择 unsigned 类型
        positive_data = np.array([[[0.0, 100.0, 200.0]]], dtype=np.float32)
        int_array, dtype_name = convert_to_minimal_int(positive_data)
        assert int_array.dtype == np.uint8  # 应该选择 uint8 而不是 int16
        
        # 对于包含负值的数据，必须选择 signed 类型
        negative_data = np.array([[[-10.0, 0.0, 100.0]]], dtype=np.float32)
        int_array, dtype_name = convert_to_minimal_int(negative_data)
        assert int_array.dtype == np.int8  # 必须选择有符号类型


class TestPixelRepresentation:
    """Test correct PixelRepresentation setting in DICOM output."""
    
    @pytest.mark.parametrize("dtype,expected_repr", [
        ("int8", 1),
        ("int16", 1), 
        ("int32", 1),
        ("uint8", 0),
        ("uint16", 0),
        ("uint32", 0),
    ])
    def test_pixel_representation_setting(self, dtype, expected_repr):
        """测试不同数据类型的 PixelRepresentation 设置"""
        # 创建测试数据
        if dtype.startswith('int'):
            test_data = np.array([[[-100, 0, 100]]], dtype=getattr(np, dtype))
        else:
            test_data = np.array([[[0, 100, 200]]], dtype=getattr(np, dtype))
        
        # 创建图像
        result_array, header = derive_pixel_header_from_array(test_data)
        image = DicomCubeImage(result_array, header)
        image.init_meta(modality='CT')
        
        # 检查 init_meta 设置的 PixelRepresentation
        pixel_repr = image.dicom_meta.get_shared_value(dicube.dicom.CommonTags.PixelRepresentation)
        assert pixel_repr == expected_repr
        
        # 测试保存到 DICOM 后的属性
        with tempfile.TemporaryDirectory() as temp_dir:
            dicom_dir = os.path.join(temp_dir, "repr_test")
            dicube.save_to_dicom_folder(image, dicom_dir)
            
            # 用 PyDicom 验证
            import pydicom
            dcm_files = [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
            ds = pydicom.dcmread(os.path.join(dicom_dir, dcm_files[0]))
            
            assert ds.PixelRepresentation == expected_repr


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_unsupported_dtype_error(self):
        """测试不支持的数据类型应该报错"""
        # 测试不支持的数据类型
        with pytest.raises(NotImplementedError):
            unsupported_data = np.array([[[1, 2, 3]]], dtype=np.complex64)
            derive_pixel_header_from_array(unsupported_data)
    
    def test_value_range_overflow(self):
        """测试值域溢出的处理"""
        # 创建超出所有整数类型范围的数据
        huge_data = np.array([[[1e20, 2e20, 3e20]]], dtype=np.float64)
        
        # 应该回退到浮点类型
        result_array, header = derive_pixel_header_from_array(huge_data)
        assert result_array.dtype in (np.uint8, np.uint16)  # 使用 slope 机制
        assert header.RescaleSlope != 1.0  # 必须使用缩放
    
    def test_empty_array_handling(self):
        """测试空数组的处理"""
        # 测试最小尺寸的数组
        tiny_data = np.array([[[1]]], dtype=np.int16)
        result_array, header = derive_pixel_header_from_array(tiny_data)
        
        assert result_array.shape == tiny_data.shape
        assert header.PixelDtype == "int16"


class TestPerformanceRegression:
    """Test that new logic doesn't significantly impact performance."""
    
    def test_large_array_processing(self):
        """测试大数组处理的性能（简单的回归测试）"""
        import time
        
        # 创建较大的测试数组
        large_data = np.random.randint(-1000, 3000, size=(50, 512, 512), dtype=np.int16)
        
        start_time = time.time()
        result_array, header = derive_pixel_header_from_array(large_data)
        processing_time = time.time() - start_time
        
        # 处理时间应该在合理范围内（< 1秒）
        assert processing_time < 1.0
        
        # 验证结果正确
        assert result_array.dtype == np.int16
        assert header.RescaleSlope == 1.0
        assert header.RescaleIntercept == 0.0
