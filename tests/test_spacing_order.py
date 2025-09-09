"""
Tests for DICOM PixelSpacing order consistency using SimpleITK and DiCube.

This test verifies that spacing values are correctly preserved when:
1. Creating data with SimpleITK (XYZ axis order)
2. Writing to DICOM files
3. Reading with DiCube (ZYX axis order)
"""

import os
import tempfile
import numpy as np
import pytest

try:
    import SimpleITK as sitk
    SIMPLEITK_AVAILABLE = True
except ImportError:
    SIMPLEITK_AVAILABLE = False

try:
    import dicube
    DICUBE_AVAILABLE = True
except ImportError:
    DICUBE_AVAILABLE = False


@pytest.mark.skipif(
    not SIMPLEITK_AVAILABLE or not DICUBE_AVAILABLE,
    reason="SimpleITK and dicube are required for spacing order tests"
)
def test_spacing_order_consistency():
    """测试SimpleITK和DiCube之间的spacing顺序一致性"""
    
    # 1. 用SimpleITK构建假数据，xyz的shape、size、spacing都不一样
    shape_xyz = (64, 128, 32)  # X=64, Y=128, Z=32 (SimpleITK格式)
    spacing_xyz = (0.5, 1.5, 2.5)  # X=0.5mm, Y=1.5mm, Z=2.5mm (SimpleITK格式)
    origin_xyz = (10.0, 20.0, 30.0)  # X=10, Y=20, Z=30
    
    print(f"原始设定 (SimpleITK XYZ格式):")
    print(f"  Shape (X,Y,Z): {shape_xyz}")
    print(f"  Spacing (X,Y,Z): {spacing_xyz}")
    print(f"  Origin (X,Y,Z): {origin_xyz}")
    
    # 创建测试数据 - SimpleITK内部使用ZYX顺序存储数组
    # 但GetSize()和GetSpacing()返回XYZ格式
    image_array = np.random.randint(0, 1000, size=(shape_xyz[2], shape_xyz[1], shape_xyz[0]), dtype=np.uint16)
    
    sitk_image = sitk.GetImageFromArray(image_array)
    sitk_image.SetSpacing(spacing_xyz)  # XYZ格式
    sitk_image.SetOrigin(origin_xyz)    # XYZ格式
    sitk_image.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])  # 标准方向矩阵
    
    # 验证SimpleITK设置
    print(f"\nSimpleITK验证:")
    print(f"  GetSize (X,Y,Z): {sitk_image.GetSize()}")
    print(f"  GetSpacing (X,Y,Z): {sitk_image.GetSpacing()}")
    print(f"  GetOrigin (X,Y,Z): {sitk_image.GetOrigin()}")
    
    assert sitk_image.GetSize() == shape_xyz
    assert sitk_image.GetSpacing() == spacing_xyz
    assert sitk_image.GetOrigin() == origin_xyz
    
    # 2. 用SimpleITK写入临时DICOM文件夹
    with tempfile.TemporaryDirectory() as temp_dir:
        dicom_dir = os.path.join(temp_dir, "dicom_series")
        os.makedirs(dicom_dir, exist_ok=True)
        
        # 写DICOM序列
        writer = sitk.ImageFileWriter()
        writer.KeepOriginalImageUIDOn()
        
        for i in range(sitk_image.GetSize()[2]):  # Z轴切片数
            slice_image = sitk_image[:, :, i]
            filename = os.path.join(dicom_dir, f"slice_{i:04d}.dcm")
            writer.SetFileName(filename)
            writer.Execute(slice_image)
        
        # 手动添加必要的DICOM标签
        import pydicom
        from pydicom.uid import generate_uid
        
        dicom_files = [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
        series_uid = generate_uid()
        study_uid = generate_uid()
        
        for i, dcm_file in enumerate(sorted(dicom_files)):
            filepath = os.path.join(dicom_dir, dcm_file)
            ds = pydicom.dcmread(filepath)
            
            # 添加缺失的标签
            ds.SeriesInstanceUID = series_uid
            ds.StudyInstanceUID = study_uid
            ds.SOPInstanceUID = generate_uid()
            ds.InstanceNumber = i + 1
            
            # 添加空间信息标签
            if not hasattr(ds, 'PixelSpacing'):
                # DICOM PixelSpacing = [row_spacing(Y), column_spacing(X)]
                ds.PixelSpacing = [spacing_xyz[1], spacing_xyz[0]]  # [Y, X]
            
            if not hasattr(ds, 'ImageOrientationPatient'):
                ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
            
            if not hasattr(ds, 'ImagePositionPatient'):
                # 计算每个slice的位置
                pos_z = origin_xyz[2] + i * spacing_xyz[2]
                ds.ImagePositionPatient = [origin_xyz[0], origin_xyz[1], pos_z]
            
            if not hasattr(ds, 'SliceThickness'):
                ds.SliceThickness = spacing_xyz[2]
            
            # 保存修改后的文件
            ds.save_as(filepath)
        
        print(f"手动添加DICOM标签完成")
        
        # 用SimpleITK重新读入验证数据正确性
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
        reader.SetFileNames(dicom_names)
        sitk_reloaded = reader.Execute()
        
        print(f"\nSimpleITK重新读取验证:")
        print(f"  重读Size (X,Y,Z): {sitk_reloaded.GetSize()}")
        print(f"  重读Spacing (X,Y,Z): {sitk_reloaded.GetSpacing()}")
        print(f"  重读Origin (X,Y,Z): {sitk_reloaded.GetOrigin()}")
        
        # 验证数据一致性
        np.testing.assert_allclose(
            sitk_reloaded.GetSpacing(), 
            spacing_xyz,
            rtol=1e-5,
            err_msg=f"SimpleITK重读spacing不一致: 期望 {spacing_xyz}, 实际 {sitk_reloaded.GetSpacing()}"
        )
        
        np.testing.assert_allclose(
            sitk_reloaded.GetOrigin(), 
            origin_xyz,
            rtol=1e-5,
            err_msg=f"SimpleITK重读origin不一致: 期望 {origin_xyz}, 实际 {sitk_reloaded.GetOrigin()}"
        )
        
        assert sitk_reloaded.GetSize() == shape_xyz, \
            f"SimpleITK重读size不一致: 期望 {shape_xyz}, 实际 {sitk_reloaded.GetSize()}"
        
        # 验证DICOM文件创建
        dicom_files = [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
        print(f"\n创建了 {len(dicom_files)} 个DICOM文件")
        assert len(dicom_files) == shape_xyz[2], f"期望 {shape_xyz[2]} 个文件，实际 {len(dicom_files)} 个"
        
        # 3. 用DiCube读入，确认数据一致性
        dicube_image = dicube.load_from_dicom_folder(dicom_dir)
        
        print(f"\nDiCube读取结果:")
        print(f"  Image shape (Z,Y,X): {dicube_image.raw_image.shape}")
        
        # DiCube使用ZYX轴序，所以shape应该是 (Z,Y,X)
        expected_dicube_shape = (shape_xyz[2], shape_xyz[1], shape_xyz[0])  # (Z,Y,X)
        assert dicube_image.raw_image.shape == expected_dicube_shape, \
            f"形状不匹配: 期望 {expected_dicube_shape}, 实际 {dicube_image.raw_image.shape}"
        
        # 检查Space信息
        if dicube_image.space is not None:
            dicube_spacing = dicube_image.space.spacing
            dicube_origin = dicube_image.space.origin
            dicube_shape = dicube_image.space.shape
            
            print(f"  Space spacing (X,Y,Z): {dicube_spacing}")
            print(f"  Space origin (X,Y,Z): {dicube_origin}")
            print(f"  Space shape (Z,Y,X): {dicube_shape}")
            
            # DiCube使用ZYX轴序，所以spacing会被reverse_axis_order()转换
            # 原始XYZ spacing (0.5, 1.5, 2.5) -> ZYX spacing (2.5, 1.5, 0.5)
            expected_zyx_spacing = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])  # [Z, Y, X]
            np.testing.assert_allclose(
                dicube_spacing, 
                expected_zyx_spacing,
                rtol=1e-5,
                err_msg=f"ZYX Spacing不匹配: 期望 {expected_zyx_spacing}, 实际 {dicube_spacing}"
            )
            
            np.testing.assert_allclose(
                dicube_origin, 
                origin_xyz,
                rtol=1e-5,
                err_msg=f"Origin不匹配: 期望 {origin_xyz}, 实际 {dicube_origin}"
            )
            
            # Space的shape应该是ZYX格式
            assert dicube_shape == expected_dicube_shape, \
                f"Space shape不匹配: 期望 {expected_dicube_shape}, 实际 {dicube_shape}"
            
            print(f"\n✓ 测试通过: SimpleITK -> DICOM -> DiCube 数据一致性验证成功")
            print(f"  原始SimpleITK spacing (X,Y,Z): {spacing_xyz}")
            print(f"  期望DiCube spacing (Z,Y,X): {expected_zyx_spacing}")
            print(f"  实际DiCube spacing (Z,Y,X): {dicube_spacing}")
            print(f"  说明: DiCube使用ZYX轴序，spacing顺序正确转换")
            
        else:
            pytest.fail("DiCube无法读取Space信息")


if __name__ == "__main__":
    # 可以直接运行测试
    test_spacing_order_consistency()
    print("\n🎉 测试通过!")
