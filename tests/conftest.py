import os
import warnings

import pydicom
import pytest

from dicube.dicom.dicom_meta import DicomMeta
from dicube.core.pixel_header import PixelDataHeader

try:
    from spacetransformer import Space
    SPACETRANSFORMER_AVAILABLE = True
except ImportError:
    Space = None
    SPACETRANSFORMER_AVAILABLE = False
    warnings.warn("spacetransformer not available, some tests will be skipped")


@pytest.fixture
def dummy_dicom_meta():
    """一个简单的 DicomMeta 示例"""
    meta = DicomMeta({}, [])  # 空的文件名列表
    return meta


@pytest.fixture
def dummy_space():
    """一个简单的 Space 示例"""
    if not SPACETRANSFORMER_AVAILABLE:
        pytest.skip("spacetransformer not available")
    
    space = Space(
        shape=[512, 512, 10],
        spacing=[0.5, 0.5, 1.0],
        x_orientation=[1, 0, 0],
        y_orientation=[0, 1, 0],
        z_orientation=[0, 0, 1],
        origin=(19, 12, -1),
    )
    return space


@pytest.fixture
def dummy_pixel_header():
    """一个简单的 PixelDataHeader 示例"""
    header = PixelDataHeader(
        RESCALE_SLOPE=1.0,
        RESCALE_INTERCEPT=0.0,
        ORIGINAL_PIXEL_DTYPE="uint16",
        PIXEL_DTYPE="uint8",
        WINDOW_CENTER=40,
        WINDOW_WIDTH=400,
    )
    return header


@pytest.fixture(scope="module")
def sample_dicom_dir():
    """
    Sample DICOM directory fixture
    """
    dicom_dir = "example/data/dicom/sample_150"
    if os.path.exists(dicom_dir):
        return dicom_dir
    else:
        pytest.skip(f"Sample DICOM directory not found: {dicom_dir}")


@pytest.fixture
def dicom_files(sample_dicom_dir):
    """
    Fixture that provides a list of valid DICOM files from the sample directory.
    """
    dicom_files = [
        os.path.join(sample_dicom_dir, f) 
        for f in os.listdir(sample_dicom_dir) 
        if f.endswith('.dcm')
    ]
    if not dicom_files:
        pytest.skip(f"No DICOM files found in {sample_dicom_dir}")
    return dicom_files[:10]  # 限制文件数量以加快测试


@pytest.fixture
def dicom_meta(dicom_files):
    """
    Fixture to create a DicomMeta object from the provided DICOM files.
    """
    datasets = [pydicom.dcmread(f) for f in dicom_files]
    return DicomMeta.from_datasets(
        datasets, 
        [os.path.basename(f) for f in dicom_files]
    ) 