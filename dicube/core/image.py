# core/image.py
import warnings
from typing import Optional

import numpy as np

from ..dicom import (
    CommonTags,
    DicomMeta,
    DicomStatus,
)
from .pixel_header import PixelDataHeader
from ..storage.pixel_utils import get_float_data
from spacetransformer import Space


class DicomCubeImage:
    """A class representing a DICOM image with associated metadata and space information.

    This class handles DICOM image data along with its pixel header, metadata, and space information.
    It provides methods for file I/O and data manipulation.
    
    Attributes:
        raw_image (np.ndarray): The raw image data array.
        pixel_header (PixelDataHeader): Pixel data header containing metadata about the image pixels.
        dicom_meta (DicomMeta, optional): DICOM metadata associated with the image.
        space (Space, optional): Spatial information describing the image dimensions and orientation.
        dicom_status (str, optional): DICOM status string. Defaults to DicomStatus.CONSISTENT.value.
    """

    def __init__(
        self,
        raw_image: np.ndarray,
        pixel_header: PixelDataHeader,
        dicom_meta: Optional[DicomMeta] = None,
        space: Optional[Space] = None,
        dicom_status: str = DicomStatus.CONSISTENT.value,
    ):
        """Initialize a DicomCubeImage instance.

        Args:
            raw_image (np.ndarray): Raw image data array.
            pixel_header (PixelDataHeader): Pixel data header information.
            dicom_meta (DicomMeta, optional): DICOM metadata. Defaults to None.
            space (Space, optional): Spatial information. Defaults to None.
        """
        self.raw_image = raw_image
        self.pixel_header = pixel_header
        self.dicom_meta = dicom_meta
        self.space = space
        self.dicom_status = dicom_status 
        self._validate_shape()


    def init_meta(
        self,
        modality: str = "OT",
        patient_name: str = "ANONYMOUS^",
        patient_id: str = "0000000",
    ) -> DicomMeta:
        """Initialize a basic DicomMeta when none is provided.

        Sets required DICOM fields with default values.

        Args:
            modality (str): Image modality, such as CT/MR/PT. Defaults to "OT".
            patient_name (str): Patient name. Defaults to "ANONYMOUS^".
            patient_id (str): Patient ID. Defaults to "0000000".

        Returns:
            DicomMeta: A new DicomMeta instance with basic required fields.
        """
        import datetime

        from pydicom.uid import generate_uid

        # Create empty DicomMeta
        num_slices = self.raw_image.shape[0] if len(self.raw_image.shape) == 3 else 1
        meta = DicomMeta({}, [f"slice_{i:04d}.dcm" for i in range(num_slices)])

        # Generate necessary UIDs
        study_uid = generate_uid()
        series_uid = generate_uid()
        sop_uid = generate_uid()
        frame_uid = generate_uid()

        # Get current date and time
        now = datetime.datetime.now()
        date_str = now.strftime("%Y%m%d")
        time_str = now.strftime("%H%M%S")

        # Patient Information
        meta.set_shared_item(CommonTags.PatientName, {'Alphabetic':patient_name})
        meta.set_shared_item(CommonTags.PatientID, patient_id)
        meta.set_shared_item(CommonTags.PatientBirthDate, "19700101")
        meta.set_shared_item(CommonTags.PatientSex, "O")

        # Study Information
        meta.set_shared_item(CommonTags.StudyInstanceUID, study_uid)
        meta.set_shared_item(CommonTags.StudyDate, date_str)
        meta.set_shared_item(CommonTags.StudyTime, time_str)
        meta.set_shared_item(CommonTags.StudyID, "1")
        meta.set_shared_item(CommonTags.StudyDescription, f"Default {modality} Study")

        # Series Information
        meta.set_shared_item(CommonTags.SeriesInstanceUID, series_uid)
        meta.set_shared_item(CommonTags.SeriesNumber, "1")
        meta.set_shared_item(
            CommonTags.SeriesDescription, f"Default {modality} Series"
        )

        # Image Information
        if num_slices > 1:
            sop_uids = [generate_uid() for _ in range(num_slices)]
            instance_numbers = [str(i + 1) for i in range(num_slices)]
            meta.set_nonshared_item(CommonTags.SOPInstanceUID, sop_uids)
            meta.set_nonshared_item(CommonTags.InstanceNumber, instance_numbers)
        else:
            meta.set_shared_item(CommonTags.SOPInstanceUID, sop_uid)
            meta.set_shared_item(CommonTags.InstanceNumber, "1")

        meta.set_shared_item(CommonTags.FrameOfReferenceUID, frame_uid)

        # Space Information
        if self.space is not None:
            # Set orientation information
            orientation = self.space.to_dicom_orientation()
            meta.set_shared_item(
                CommonTags.ImageOrientationPatient, list(orientation)
            )
            meta.set_shared_item(CommonTags.PixelSpacing, list(self.space.spacing[:2]))
            meta.set_shared_item(
                CommonTags.SliceThickness, float(self.space.spacing[2])
            )

            # Set position information
            if num_slices > 1:
                positions = []
                for i in range(num_slices):
                    # Calculate position for each slice using space's z_orientation
                    pos = np.array(self.space.origin) + i * self.space.spacing[
                        2
                    ] * np.array(self.space.z_orientation)
                    positions.append(pos.tolist())
                meta.set_nonshared_item(CommonTags.ImagePositionPatient, positions)
            else:
                meta.set_shared_item(
                    CommonTags.ImagePositionPatient, list(self.space.origin)
                )
        else:
            # If no space information, set default values
            meta.set_shared_item(
                CommonTags.ImageOrientationPatient, [1, 0, 0, 0, 1, 0]
            )
            meta.set_shared_item(CommonTags.PixelSpacing, [1.0, 1.0])
            meta.set_shared_item(CommonTags.SliceThickness, 1.0)
            if num_slices > 1:
                positions = [[0, 0, i] for i in range(num_slices)]
                meta.set_nonshared_item(CommonTags.ImagePositionPatient, positions)
            else:
                meta.set_shared_item(CommonTags.ImagePositionPatient, [0, 0, 0])

        # Pixel Information
        shape = self.raw_image.shape
        if len(shape) == 3:
            meta.set_shared_item(CommonTags.Rows, shape[1])
            meta.set_shared_item(CommonTags.Columns, shape[2])
        else:
            meta.set_shared_item(CommonTags.Rows, shape[0])
            meta.set_shared_item(CommonTags.Columns, shape[1])

        meta.set_shared_item(CommonTags.SamplesPerPixel, 1)
        meta.set_shared_item(CommonTags.PhotometricInterpretation, "MONOCHROME2")
        meta.set_shared_item(CommonTags.BitsAllocated, 16)
        meta.set_shared_item(CommonTags.BitsStored, 16)
        meta.set_shared_item(CommonTags.HighBit, 15)
        meta.set_shared_item(CommonTags.PixelRepresentation, 0)

        # Rescale Information from pixel_header
        if self.pixel_header.RESCALE_SLOPE is not None:
            meta.set_shared_item(
                CommonTags.RescaleSlope, float(self.pixel_header.RESCALE_SLOPE)
            )
        if self.pixel_header.RESCALE_INTERCEPT is not None:
            meta.set_shared_item(
                CommonTags.RescaleIntercept, float(self.pixel_header.RESCALE_INTERCEPT)
            )

        # Modality Information
        meta.set_shared_item(CommonTags.Modality, modality)

        self.dicom_meta = meta
        return meta

    @property
    def shape(self):
        """Get the shape of the raw image.
        
        Returns:
            tuple: The shape of the raw image array.
        """
        return self.raw_image.shape
    
    @property
    def dtype(self):
        """Get the data type of the raw image.
        
        Returns:
            numpy.dtype: The data type of the raw image array.
        """
        return self.raw_image.dtype
    
    def _validate_shape(self):
        """Validate that the image shape matches the space shape if both are present.
        
        Both raw_image and space are now in (z,y,x) format internally.

        Raises:
            ValueError: If space shape doesn't match image dimensions.
        """
        if self.space and self.raw_image.ndim >= 3:
            expected_shape = tuple(self.space.shape)
            if self.raw_image.shape[-len(expected_shape) :] != expected_shape:
                raise ValueError(
                    f"Space shape {expected_shape} mismatch with image {self.raw_image.shape}"
                )

    def get_fdata(self, dtype="float32") -> np.ndarray:
        """Get image data as floating point array with slope/intercept applied.

        Args:
            dtype (str): Output data type, must be one of: float16, float32, float64. Defaults to "float32".

        Returns:
            np.ndarray: Floating point image data with rescale factors applied.
        """
        return get_float_data(self.raw_image, self.pixel_header, dtype) 