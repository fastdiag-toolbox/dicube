# core/image.py
import struct
import warnings
from typing import List, Optional

import numpy as np

from ..dicom import (
    CommonTags,
    DicomMeta,
    DicomStatus,
    SortMethod,
    get_dicom_status,
    read_dicom_dir,
)
from ..dicom.dicom_io import save_to_dicom_folder
from ..storage.dcb_file import DcbSFile, DcbFile, DcbAFile, DcbLFile
from .pixel_header import PixelDataHeader
from ..storage.pixel_utils import derive_pixel_header_from_array, get_float_data
from ..dicom import get_space_from_DicomMeta
try:
    from spacetransformer import Space, get_space_from_nifty_affine
except ImportError:
    warnings.warn("spacetransformer not found, spatial functionality will be limited")
    Space = None
    get_space_from_DicomMeta = None
    get_space_from_nifty_affine = None


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

    @property
    def shape(self):
        """
        Get the shape of the image.
        """
        return self.raw_image.shape

    @property
    def space_zyx(self):
        """
        Get the shape of the image in zyx order.
        """
        if self.space:
            return self.space.reverse_axis_order()
        else:
            return None

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

    def _validate_shape(self):
        """
        Validate that the image shape matches the space shape if both are present.

        Raises:
            ValueError: If space shape doesn't match image dimensions
        """
        if self.space and self.raw_image.ndim >= 3:
            expected_shape = tuple(self.space.shape[::-1])
            if self.raw_image.shape[-len(expected_shape) :] != expected_shape:
                raise ValueError(
                    f"Space shape {expected_shape} mismatch with image {self.raw_image.shape}"
                )

    def to_file(self, filename: str, num_threads: int = 4, file_type: str = "s"):
        """
        Write the current object to a file using JPEG XL compression.

        Args:
            filename: Path to the output file
            num_threads: Number of workers for parallel encoding
            file_type: File type, one of "s", "a", "l"
        """
        if file_type == "s":
            writer = DcbSFile(filename, mode="w")
        elif file_type == "a":
            writer = DcbAFile(filename, mode="w")
        elif file_type == "l":
            writer = DcbLFile(filename, mode="w")
        else:
            raise ValueError(f"Unsupported file type: {file_type}, must be one of 's', 'a', 'l'")
        
        writer.write(
            dicom_meta=self.dicom_meta,
            space=self.space,
            pixel_header=self.pixel_header,
            images=self.raw_image,
            num_threads=num_threads,
        )


    def to_dicom_folder(
        self,
        output_dir: str,
        filenames: Optional[List[str]] = None,
        use_j2k: bool = False,
        lossless: bool = True,
        **compress_kwargs,
    ):
        """
        Save DicomCubeImage as a DICOM directory.

        Only supports cases with dicom_meta. Will override the original DICOM's
        slope/intercept values with those from DicomCubeImage.

        Args:
            output_dir: Output directory path
            filenames: Optional list of filenames for each slice
            use_j2k: Whether to use JPEG 2000 compression
            lossless: Whether to use lossless compression when use_j2k is True
            **compress_kwargs: Additional keyword arguments passed to ds.compress()
                For example:
                - encoding_plugin: The plugin to use for compression ('pylibjpeg', 'gdcm', etc)
                - compression_level: Compression level for lossy compression
                - photometric_interpretation: Color space for compression

        Raises:
            ValueError: If dicom_meta is not present
        """
        if self.dicom_meta is None:
            warnings.warn("dicom_meta is None, initializing with default values")
            self.init_meta()

        save_to_dicom_folder(
            raw_images=self.raw_image,
            dicom_meta=self.dicom_meta,
            pixel_header=self.pixel_header,
            output_dir=output_dir,
            filenames=filenames,
            use_j2k=use_j2k,
            lossless=lossless,
            **compress_kwargs,
        )

    @classmethod
    def from_dicom_folder(
        cls, folder_path: str, sort_method: SortMethod = SortMethod.INSTANCE_NUMBER_ASC
    ):
        """
        Create a DicomCubeImage instance from a DICOM directory.

        Args:
            folder_path: Path to the DICOM directory
            sort_method: Method to sort DICOM files, defaults to instance number ascending

        Returns:
            DicomCubeImage: A new instance created from the DICOM files

        Raises:
            ValueError: If DICOM status is not supported
        """
        # 实现读取DICOM文件夹的逻辑
        meta, datasets = read_dicom_dir(folder_path, sort_method=sort_method)
        images = [d.pixel_array for d in datasets]
        status = get_dicom_status(meta)
        if status in (
            DicomStatus.NON_UNIFORM_RESCALE_FACTOR,
            DicomStatus.MISSING_DTYPE,
            DicomStatus.NON_UNIFORM_DTYPE,
            DicomStatus.MISSING_SHAPE,
            DicomStatus.INCONSISTENT,
        ):
            raise ValueError("DicomStatus not supported: %s" % status)
        if status in (
            DicomStatus.MISSING_SPACING,
            DicomStatus.NON_UNIFORM_SPACING,
            DicomStatus.MISSING_ORIENTATION,
            DicomStatus.NON_UNIFORM_ORIENTATION,
            DicomStatus.MISSING_LOCATION,
            DicomStatus.REVERSED_LOCATION,
            DicomStatus.DWELLING_LOCATION,
            DicomStatus.GAP_LOCATION,
        ):
            warnings.warn("DicomStatus: %s, cannot compute space" % status)
            space = None
        else:
            if get_space_from_DicomMeta is not None:
                space = get_space_from_DicomMeta(meta)
            else:
                space = None

        # 获取rescale参数
        slope = meta.get(CommonTags.RESCALE_SLOPE, force_shared=True)[0]
        intercept = meta.get(CommonTags.RESCALE_INTERCEPT, force_shared=True)[0]
        wind_center = meta.get(CommonTags.WINDOW_CENTER, force_shared=True)
        wind_width = meta.get(CommonTags.WINDOW_WIDTH, force_shared=True)

        # 创建pixel_header
        pixel_header = PixelDataHeader(
            RESCALE_SLOPE=float(slope) if slope is not None else 1.0,
            RESCALE_INTERCEPT=float(intercept) if intercept is not None else 0.0,
            ORIGINAL_PIXEL_DTYPE=str(images[0].dtype),
            PIXEL_DTYPE=str(images[0].dtype),
            WINDOW_CENTER=float(wind_center[0]) if wind_center is not None else None,
            WINDOW_WIDTH=float(wind_width[0]) if wind_width is not None else None,
        )

        image = DicomCubeImage(np.array(images), pixel_header, meta, space)
        return image

    @classmethod
    def from_nifti(cls, nii_path: str):
        """
        Create a DicomCubeImage instance from a NIfTI file.

        Args:
            nii_path: Path to the NIfTI file

        Returns:
            DicomCubeImage: A new instance created from the NIfTI file
        """
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError("nibabel is required to read NIfTI files")

        if get_space_from_nifty_affine is None:
            raise ImportError("spacetransformer is required for NIfTI spatial support")

        nii = nib.load(nii_path)
        space = get_space_from_nifty_affine(nii.affine, nii.shape[::-1])
        # Fix numpy array warning
        raw_image, header = derive_pixel_header_from_array(
            np.asarray(nii.dataobj, dtype=nii.dataobj.dtype)
        )
        image = DicomCubeImage(raw_image, header, space=space)
        return image

    @classmethod
    def from_file(cls, filename: str, num_threads: int = 4):
        """
        Automatically select the appropriate reader class based
        on the file magic number to create a DicomCubeImage instance.

        Args:
            filename: Input file path
            num_threads: Number of workers for parallel decoding, default is 4

        Returns:
            DicomCubeImage: A new instance created from the file

        Raises:
            ValueError: When the file format is not supported
        """
        # 读取文件头部来判断魔数
        header_size = struct.calcsize(DcbFile.HEADER_STRUCT)
        with open(filename, "rb") as f:
            header_data = f.read(header_size)
        magic = struct.unpack(DcbFile.HEADER_STRUCT, header_data)[0]

        # 根据魔数选择合适的reader类
        if magic == DcbAFile.MAGIC:
            reader = DcbAFile(filename, mode="r")
        elif magic == DcbSFile.MAGIC:
            reader = DcbSFile(filename, mode="r")
        else:
            raise ValueError(f"不支持的文件格式,魔数: {magic}")

        # 读取文件内容
        dicom_meta = reader.read_meta()
        space = reader.read_space()
        pixel_header = reader.read_pixel_header()

        images = reader.read_images(num_threads=num_threads)
        if isinstance(images, list):
            if len(images) == 0:
                raw_image = np.array(
                    [], dtype=pixel_header.ORIGINAL_PIXEL_DTYPE
                )  # empty
            else:
                raw_image = np.stack(images, axis=0)
        else:
            # Input is a (num_frames, H, W, ...) ndarray
            raw_image = images

        return cls(
            raw_image=raw_image,
            pixel_header=pixel_header,
            dicom_meta=dicom_meta,
            space=space,
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