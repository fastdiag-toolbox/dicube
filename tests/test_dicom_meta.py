import pydicom

from dicube.dicom.dicom_meta import DicomMeta, SortMethod
from dicube.dicom.dicom_tags import CommonTags


def test_dicommeta_from_datasets(dicom_meta):
    """
    Test that DicomMeta can be correctly instantiated from DICOM datasets.
    """
    assert dicom_meta.num_datasets > 0, "No datasets were loaded."
    assert isinstance(dicom_meta, DicomMeta), "Object is not of type DicomMeta."


def test_get_shared_metadata(dicom_meta):
    """
    Test the retrieval of shared metadata from DicomMeta.
    """
    patient_name = dicom_meta.get(CommonTags.PATIENT_NAME)
    assert patient_name is not None, "Patient name not found in shared metadata."


def test_get_nonshared_metadata(dicom_meta):
    """
    Test the retrieval of non-shared metadata from DicomMeta.
    """
    instance_numbers = dicom_meta.get(CommonTags.INSTANCE_NUMBER, force_nonshared=True)
    assert isinstance(instance_numbers, list), "Instance numbers should be a list."
    assert (
        len(instance_numbers) == dicom_meta.num_datasets
    ), "Mismatch in number of instance numbers."


def test_sort_files(dicom_meta):
    """
    Test the sorting functionality of DicomMeta.
    """
    # Test ascending sort by instance number
    dicom_meta.sort_files(SortMethod.INSTANCE_NUMBER_ASC)
    instance_numbers_sorted = dicom_meta.get(
        CommonTags.INSTANCE_NUMBER, force_nonshared=True
    )
    assert instance_numbers_sorted == sorted(
        instance_numbers_sorted
    ), "Instance numbers are not sorted correctly."

    # Test descending sort by instance number
    dicom_meta.sort_files(SortMethod.INSTANCE_NUMBER_DESC)
    instance_numbers_sorted_desc = dicom_meta.get(
        CommonTags.INSTANCE_NUMBER, force_nonshared=True
    )
    assert instance_numbers_sorted_desc == sorted(
        instance_numbers_sorted_desc, reverse=True
    ), "Instance numbers are not sorted correctly in descending order."


def test_projection_location(dicom_meta):
    """
    Test the calculation of projection location for datasets.
    """
    projection_locations = dicom_meta._get_projection_location()
    assert isinstance(
        projection_locations, list
    ), "Projection locations should be a list."
    assert (
        len(projection_locations) == dicom_meta.num_datasets
    ), "Mismatch in number of projection locations."


def test_dicom_json_convert(dicom_files, dicom_meta):
    """
    Test JSON conversion functionality.
    """
    json_back = dicom_meta.index(0)
    json_convert = pydicom.dcmread(dicom_files[0]).to_json_dict(
        bulk_data_threshold=10240, bulk_data_element_handler=lambda x: None
    )
    json_back.pop(CommonTags.PIXEL_DATA.key)
    json_convert.pop(CommonTags.PIXEL_DATA.key)
    assert (
        json_convert == json_back
    ), "read dicom json different from json convert from dicommeta."


def test_dicom_meta_basic_operations(dummy_dicom_meta):
    """
    Test basic DicomMeta operations.
    """
    # Test empty meta
    assert dummy_dicom_meta.num_datasets == 0
    
    # Test setting shared items
    dummy_dicom_meta.set_shared_item(CommonTags.PATIENT_NAME, "TEST_PATIENT")
    patient_name = dummy_dicom_meta.get(CommonTags.PATIENT_NAME)
    # DICOM PATIENT_NAME is returned in Person Name format
    assert patient_name == [{'Alphabetic': 'TEST_PATIENT'}]


def test_dicom_meta_tags_access():
    """
    Test DICOM tag access functionality.
    """
    # Test that common tags are accessible
    assert hasattr(CommonTags, 'PATIENT_NAME')
    assert hasattr(CommonTags, 'MODALITY')
    assert hasattr(CommonTags, 'INSTANCE_NUMBER')
    
    # Test tag properties
    patient_tag = CommonTags.PATIENT_NAME
    assert hasattr(patient_tag, 'key')
    assert hasattr(patient_tag, 'vr') 