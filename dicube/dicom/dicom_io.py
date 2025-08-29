import os
import warnings
import inspect
from typing import List, Optional

import numpy as np
from pydicom import Dataset
from pydicom.dataset import FileMetaDataset
from pydicom.uid import JPEG2000, ExplicitVRLittleEndian, JPEG2000Lossless, generate_uid

from ..dicom.dicom_meta import DicomMeta


def prepare_output_dir(output_dir: str):
    """Prepare output directory"""
    if os.path.exists(output_dir):
        import shutil

        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)


def create_file_meta(ds):
    """Create file meta information"""
    file_meta = FileMetaDataset()

    MODALITY_SOP_CLASS_MAP = {
        "CT": "1.2.840.10008.5.1.4.1.1.2",
        "MR": "1.2.840.10008.5.1.4.1.1.4",
        "US": "1.2.840.10008.5.1.4.1.1.6.1",
        "PT": "1.2.840.10008.5.1.4.1.1.128",
        "CR": "1.2.840.10008.5.1.4.1.1.1",
        "DX": "1.2.840.10008.5.1.4.1.1.1.1",
        "NM": "1.2.840.10008.5.1.4.1.1.20",
    }

    modality = ds.Modality if hasattr(ds, "Modality") else "CT"
    default_sop_uid = MODALITY_SOP_CLASS_MAP.get(modality, MODALITY_SOP_CLASS_MAP["CT"])

    file_meta.MediaStorageSOPClassUID = (
        ds.SOPClassUID if hasattr(ds, "SOPClassUID") else default_sop_uid
    )
    file_meta.MediaStorageSOPInstanceUID = (
        ds.SOPInstanceUID if hasattr(ds, "SOPInstanceUID") else generate_uid()
    )
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    return file_meta


def ensure_required_tags(ds):
    """Ensure required DICOM tags exist"""
    if not hasattr(ds, "SOPClassUID"):
        ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
    if not hasattr(ds, "SOPInstanceUID"):
        ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID





def set_pixel_data_attributes(ds: Dataset, pixel_header):
    """Set pixel data attributes based on pixel header information.
    
    Args:
        ds (Dataset): DICOM dataset to update
        pixel_header: Pixel header containing data type information
    """
    if pixel_header:
        ds.RescaleSlope = pixel_header.RescaleSlope
        ds.RescaleIntercept = pixel_header.RescaleIntercept
        
        # Set correct pixel data attributes based on pixel_header
        pixel_dtype = pixel_header.PixelDtype
        
        # Determine bits based on data type
        dtype_to_bits = {
            "uint8": (8, 8, 7, 0),
            "int8": (8, 8, 7, 1),
            "uint16": (16, 16, 15, 0),
            "int16": (16, 16, 15, 1),
            "uint32": (32, 32, 31, 0),
            "int32": (32, 32, 31, 1),
        }
        
        bits_allocated, bits_stored, high_bit, pixel_rep = dtype_to_bits.get(pixel_dtype, (16, 16, 15, 0))
        ds.BitsAllocated = bits_allocated
        ds.BitsStored = bits_stored
        ds.HighBit = high_bit
        ds.PixelRepresentation = pixel_rep
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"


def create_dicom_dataset(meta_dict: dict, pixel_header, transfer_syntax_uid=None):
    """Create DICOM dataset with unified pixel attribute handling.
    
    Args:
        meta_dict: Metadata dictionary for the dataset
        pixel_header: Pixel header containing data type information
        transfer_syntax_uid: Optional transfer syntax UID (defaults to ExplicitVRLittleEndian)
    """
    ds = Dataset.from_json(meta_dict)

    if hasattr(ds, "file_meta"):
        warnings.warn("Found original file metadata, will be overridden")

    # Create file metadata
    file_meta = create_file_meta(ds)
    if transfer_syntax_uid:
        file_meta.TransferSyntaxUID = transfer_syntax_uid
    ds.file_meta = file_meta
    
    ensure_required_tags(ds)
    
    # Set unified pixel data attributes
    set_pixel_data_attributes(ds, pixel_header)

    return ds


def save_dicom(ds: Dataset, output_path: str):
    """Save DICOM file.
    
    Args:
        ds: DICOM dataset
        output_path: Output file path
    """
    
    sig = inspect.signature(Dataset.save_as)
    if "enforce_file_format" in sig.parameters:  # pydicom >= 3.0
        ds.save_as(output_path, enforce_file_format=True)
    else:
        # Ensure valid transfer syntax UID
        if hasattr(ds, 'file_meta') and hasattr(ds.file_meta, 'TransferSyntaxUID'):
            # Check if valid, replace with standard ExplicitVRLittleEndian if invalid
            try:
                from pydicom.uid import UID
                uid = UID(ds.file_meta.TransferSyntaxUID)
                if not hasattr(uid, 'is_little_endian'):
                    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            except (ValueError, AttributeError):
                # If UID is invalid, use standard ExplicitVRLittleEndian
                ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.save_as(output_path, write_like_original=False)

def save_to_dicom_folder(
    raw_images: np.ndarray,
    dicom_meta: DicomMeta,
    pixel_header,
    output_dir: str,
    filenames: Optional[List[str]] = None,
):
    """Save image data as DICOM folder.
    
    Args:
        raw_images: Raw image data, can be 2D or 3D array
        dicom_meta: DICOM metadata
        pixel_header: Pixel header information
        output_dir: Output directory
        filenames: Custom filename list, if None default names will be used
    """
    prepare_output_dir(output_dir)

    if raw_images.ndim == 2:
        raw_images = raw_images[np.newaxis, ...]

    for idx in range(len(raw_images)):
        # Create dataset with unified pixel attribute handling
        ds = create_dicom_dataset(dicom_meta.index(idx), pixel_header)
        img = raw_images[idx]

        # Ensure image data type matches pixel header expectations
        target_dtype = getattr(np, pixel_header.PixelDtype, img.dtype)
        if img.dtype != target_dtype:
            img = img.astype(target_dtype)

        ds.PixelData = img.tobytes()

        output_path = os.path.join(
            output_dir, filenames[idx] if filenames else f"slice_{idx:04d}.dcm"
        )
        save_dicom(ds, output_path)