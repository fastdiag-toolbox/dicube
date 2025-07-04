from spacetransformer import Space
from .dicom_status import DicomStatus, get_dicom_status
from .dicom_tags import CommonTags
import numpy as np


def get_space_from_DicomMeta(meta):
    """
    Create a Space object from DICOM metadata.

    Extracts geometric information from DICOM tags including:
    - Image Position (Patient) for origin
    - Pixel Spacing and Slice Thickness for spacing
    - Image Orientation (Patient) for direction cosines
    - Rows, Columns, and number of slices for shape

    Args:
        meta: DicomMeta object containing DICOM metadata
             Must support meta[Tag] -> (value, status) interface

    Returns:
        Space: A new Space instance with geometry matching the DICOM data

    Raises:
        ValueError: If required DICOM tags are missing or invalid
    """

    num_images = len(meta)
    status = get_dicom_status(meta)
    if status not in (DicomStatus.CONSISTENT, DicomStatus.NON_UNIFORM_RESCALE_FACTOR):
        return None
    spacing = meta.get(CommonTags.PIXEL_SPACING, force_shared=True)
    spacing = [float(s) for s in spacing]
    positions = np.array(
        meta.get(CommonTags.IMAGE_POSITION_PATIENT, force_nonshared=True)
    )
    orientation = meta.get(CommonTags.IMAGE_ORIENTATION_PATIENT, force_shared=True)
    orientation = [float(s) for s in orientation]
    origin = positions[0].tolist()
    if num_images > 1:
        diff = positions[-1] - positions[0]
        z_orientation = diff / np.linalg.norm(diff).tolist()
        z_step_vector = diff / (num_images - 1)
        spacing.append(float(np.linalg.norm(z_step_vector)))
    else:
        thickness = meta.get(CommonTags.SLICE_THICKNESS, force_shared=True)
        if thickness is None:
            thickness = 1
        spacing.append(float(thickness))
        z_orientation = np.cross(orientation[:3], orientation[3:6]).tolist()
    shape = [
        int(meta.get(CommonTags.COLUMNS, force_shared=True)[0]),
        int(meta.get(CommonTags.ROWS, force_shared=True)[0]),
        num_images,
    ]
    space = Space(
        origin=origin,
        spacing=spacing,
        x_orientation=orientation[:3],
        y_orientation=orientation[3:6],
        z_orientation=z_orientation,
        shape=shape,
    )
    return space
