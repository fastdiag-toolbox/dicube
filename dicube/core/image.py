# core/image.py
import warnings
from typing import Optional

import numpy as np

from ..dicom import (
    CommonTags,
    DicomMeta,
)
from .pixel_header import PixelDataHeader
from ..storage.pixel_utils import get_float_data
from spacetransformer import Space


class DicomCubeImage:
    """
    A class representing a DICOM image with associated metadata and space information.

    This class handles DICOM image data along with its pixel header, metadata, and space information.
    It provides methods for file I/O and data manipulation.
    """

    def __init__(
        self,
        raw_image: np.ndarray,
        pixel_header: PixelDataHeader,
        dicom_meta: Optional[DicomMeta] = None,
        space: Optional[Space] = None,
    ):
        """
        Initialize a DicomCubeImage instance.

        Args:
            raw_image: Raw image data array
            pixel_header: Pixel data header information
            dicom_meta: Optional DICOM metadata
            space: Optional space information
        """
        self.raw_image = raw_image
        self.pixel_header = pixel_header
        self.dicom_meta = dicom_meta
        self.space = space
        self._validate_shape()


    def init_meta(
        self,
        modality: str = "OT",
        patient_name: str = "ANONYMOUS",
        patient_id: str = "0000000",
    ) -> DicomMeta:
        """
        Initialize a basic DicomMeta when none is provided.
        Sets required DICOM fields with default values.

        Args:
            modality: 图像模态,如 CT/MR/PT 等
            patient_name: 患者姓名,默认匿名
            patient_id: 患者ID,默认0000000

        Returns:
            DicomMeta: A new DicomMeta instance with basic required fields
        """
        import datetime

        from pydicom.uid import generate_uid

        # 创建空的 DicomMeta
        num_slices = self.raw_image.shape[0] if len(self.raw_image.shape) == 3 else 1
        meta = DicomMeta({}, [f"slice_{i:04d}.dcm" for i in range(num_slices)])

        # 生成必要的 UID
        study_uid = generate_uid()
        series_uid = generate_uid()
        sop_uid = generate_uid()
        frame_uid = generate_uid()

        # 获取当前日期时间
        now = datetime.datetime.now()
        date_str = now.strftime("%Y%m%d")
        time_str = now.strftime("%H%M%S")

        # Patient Information
        meta.set_shared_item(CommonTags.PATIENT_NAME, patient_name)
        meta.set_shared_item(CommonTags.PATIENT_ID, patient_id)
        meta.set_shared_item(CommonTags.PATIENT_BIRTH_DATE, "19700101")
        meta.set_shared_item(CommonTags.PATIENT_SEX, "O")

        # Study Information
        meta.set_shared_item(CommonTags.STUDY_INSTANCE_UID, study_uid)
        meta.set_shared_item(CommonTags.STUDY_DATE, date_str)
        meta.set_shared_item(CommonTags.STUDY_TIME, time_str)
        meta.set_shared_item(CommonTags.STUDY_ID, "1")
        meta.set_shared_item(CommonTags.STUDY_DESCRIPTION, f"Default {modality} Study")

        # Series Information
        meta.set_shared_item(CommonTags.SERIES_INSTANCE_UID, series_uid)
        meta.set_shared_item(CommonTags.SERIES_NUMBER, "1")
        meta.set_shared_item(
            CommonTags.SERIES_DESCRIPTION, f"Default {modality} Series"
        )

        # Image Information
        if num_slices > 1:
            sop_uids = [generate_uid() for _ in range(num_slices)]
            instance_numbers = [str(i + 1) for i in range(num_slices)]
            meta.set_nonshared_item(CommonTags.SOP_INSTANCE_UID, sop_uids)
            meta.set_nonshared_item(CommonTags.INSTANCE_NUMBER, instance_numbers)
        else:
            meta.set_shared_item(CommonTags.SOP_INSTANCE_UID, sop_uid)
            meta.set_shared_item(CommonTags.INSTANCE_NUMBER, "1")

        meta.set_shared_item(CommonTags.FRAME_OF_REFERENCE_UID, frame_uid)

        # space Information
        if self.space is not None:
            # 设置方向信息
            orientation = self.space.to_dicom_orientation()
            meta.set_shared_item(
                CommonTags.IMAGE_ORIENTATION_PATIENT, list(orientation)
            )
            meta.set_shared_item(CommonTags.PIXEL_SPACING, list(self.space.spacing[:2]))
            meta.set_shared_item(
                CommonTags.SLICE_THICKNESS, float(self.space.spacing[2])
            )

            # 设置位置信息
            if num_slices > 1:
                positions = []
                for i in range(num_slices):
                    # 使用 space 的 z_orientation 计算每个切片的位置
                    pos = np.array(self.space.origin) + i * self.space.spacing[
                        2
                    ] * np.array(self.space.z_orientation)
                    positions.append(pos.tolist())
                meta.set_nonshared_item(CommonTags.IMAGE_POSITION_PATIENT, positions)
            else:
                meta.set_shared_item(
                    CommonTags.IMAGE_POSITION_PATIENT, list(self.space.origin)
                )
        else:
            # 如果没有 space 信息,设置默认值
            meta.set_shared_item(
                CommonTags.IMAGE_ORIENTATION_PATIENT, [1, 0, 0, 0, 1, 0]
            )
            meta.set_shared_item(CommonTags.PIXEL_SPACING, [1.0, 1.0])
            meta.set_shared_item(CommonTags.SLICE_THICKNESS, 1.0)
            if num_slices > 1:
                positions = [[0, 0, i] for i in range(num_slices)]
                meta.set_nonshared_item(CommonTags.IMAGE_POSITION_PATIENT, positions)
            else:
                meta.set_shared_item(CommonTags.IMAGE_POSITION_PATIENT, [0, 0, 0])

        # Pixel Information
        shape = self.raw_image.shape
        if len(shape) == 3:
            meta.set_shared_item(CommonTags.ROWS, shape[1])
            meta.set_shared_item(CommonTags.COLUMNS, shape[2])
        else:
            meta.set_shared_item(CommonTags.ROWS, shape[0])
            meta.set_shared_item(CommonTags.COLUMNS, shape[1])

        meta.set_shared_item(CommonTags.SAMPLES_PER_PIXEL, 1)
        meta.set_shared_item(CommonTags.PHOTOMETRIC_INTERPRETATION, "MONOCHROME2")
        meta.set_shared_item(CommonTags.BITS_ALLOCATED, 16)
        meta.set_shared_item(CommonTags.BITS_STORED, 16)
        meta.set_shared_item(CommonTags.HIGH_BIT, 15)
        meta.set_shared_item(CommonTags.PIXEL_REPRESENTATION, 0)

        # Rescale Information from pixel_header
        if self.pixel_header.RESCALE_SLOPE is not None:
            meta.set_shared_item(
                CommonTags.RESCALE_SLOPE, float(self.pixel_header.RESCALE_SLOPE)
            )
        if self.pixel_header.RESCALE_INTERCEPT is not None:
            meta.set_shared_item(
                CommonTags.RESCALE_INTERCEPT, float(self.pixel_header.RESCALE_INTERCEPT)
            )

        # Modality Information
        meta.set_shared_item(CommonTags.MODALITY, modality)

        self.dicom_meta = meta

    @property
    def shape(self):
        return self.raw_image.shape
    
    @property
    def dtype(self):
        return self.raw_image.dtype
    
    def _validate_shape(self):
        """
        Validate that the image shape matches the space shape if both are present.
        Both raw_image and space are now in (z,y,x) format internally.

        Raises:
            ValueError: If space shape doesn't match image dimensions
        """
        if self.space and self.raw_image.ndim >= 3:
            expected_shape = tuple(self.space.shape)
            if self.raw_image.shape[-len(expected_shape) :] != expected_shape:
                raise ValueError(
                    f"Space shape {expected_shape} mismatch with image {self.raw_image.shape}"
                )



    def get_fdata(self, dtype="float32") -> np.ndarray:
        """
        Get image data as floating point array with slope/intercept applied.

        Args:
            dtype: Output data type, must be one of: float16, float32, float64

        Returns:
            np.ndarray: Floating point image data with rescale factors applied
        """
        return get_float_data(self.raw_image, self.pixel_header, dtype) 