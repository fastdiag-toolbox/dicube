from spacetransformer import Space
from .dicom_status import DicomStatus, get_dicom_status
from .dicom_tags import CommonTags
import numpy as np


def get_space_from_DicomMeta(meta, axis_order="xyz"):
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

    num_images = meta.slice_count
    status = get_dicom_status(meta)
    if status not in (DicomStatus.CONSISTENT, DicomStatus.NON_UNIFORM_RESCALE_FACTOR):
        return None
    spacing = meta.get_shared_value(CommonTags.PixelSpacing)
    spacing = [float(s) for s in spacing]
    positions = np.array(
        meta.get_values(CommonTags.ImagePositionPatient)
    )
    orientation = meta.get_shared_value(CommonTags.ImageOrientationPatient)
    orientation = [float(s) for s in orientation]
    origin = positions[0].tolist()
    if num_images > 1:
        diff = positions[-1] - positions[0]
        z_orientation = diff / np.linalg.norm(diff).tolist()
        z_step_vector = diff / (num_images - 1)
        spacing.append(float(np.linalg.norm(z_step_vector)))
    else:
        thickness = meta.get_shared_value(CommonTags.SliceThickness)
        if thickness is None:
            thickness = 1
        spacing.append(float(thickness))
        z_orientation = np.cross(orientation[:3], orientation[3:6]).tolist()
    shape = [
        int(meta.get_shared_value(CommonTags.Columns)),
        int(meta.get_shared_value(CommonTags.Rows)),
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
    if axis_order == "xyz":
        space = space
    elif axis_order == "zyx":
        space = space.reverse_axis_order()
    else:
        raise ValueError(f"Invalid axis order: {axis_order}")
    return space