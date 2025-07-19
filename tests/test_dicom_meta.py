import pydicom

from dicube.dicom.dicom_meta import DicomMeta, SortMethod
from dicube.dicom.dicom_tags import CommonTags


def test_dicommeta_from_datasets(dicom_meta):
    """
    Test that DicomMeta can be correctly instantiated from DICOM datasets.
    """
    assert dicom_meta.slice_count > 0, "No datasets were loaded."
    assert isinstance(dicom_meta, DicomMeta), "Object is not of type DicomMeta."


def test_get_shared_metadata(dicom_meta):
    """
    Test the retrieval of shared metadata from DicomMeta.
    """
    # Check if patient name is shared
    assert dicom_meta.is_shared(CommonTags.PatientName), "Patient name should be shared."
    # Get shared value directly
    patient_name = dicom_meta.get_shared_value(CommonTags.PatientName)
    assert patient_name is not None, "Patient name not found in shared metadata."


def test_get_nonshared_metadata(dicom_meta):
    """
    Test the retrieval of non-shared metadata from DicomMeta.
    """
    # Get values for a non-shared tag
    instance_numbers = dicom_meta.get_values(CommonTags.InstanceNumber)
    assert isinstance(instance_numbers, list), "Instance numbers should be a list."
    assert (
        len(instance_numbers) == dicom_meta.slice_count
    ), "Mismatch in number of instance numbers."


def test_sort_files(dicom_meta):
    """
    Test the sorting functionality of DicomMeta.
    """
    # Test ascending sort by instance number
    dicom_meta.sort_files(SortMethod.INSTANCE_NUMBER_ASC)
    instance_numbers_sorted = dicom_meta.get_values(CommonTags.InstanceNumber)
    
    # Convert to integers for comparison
    int_numbers = [int(num) if num is not None else float('inf') for num in instance_numbers_sorted]
    assert int_numbers == sorted(int_numbers), "Instance numbers are not sorted correctly."

    # Test descending sort by instance number
    dicom_meta.sort_files(SortMethod.INSTANCE_NUMBER_DESC)
    instance_numbers_sorted_desc = dicom_meta.get_values(CommonTags.InstanceNumber)
    
    # Convert to integers for comparison
    int_numbers_desc = [int(num) if num is not None else float('-inf') for num in instance_numbers_sorted_desc]
    assert int_numbers_desc == sorted(int_numbers_desc, reverse=True), "Instance numbers are not sorted correctly in descending order."


def test_projection_location(dicom_meta):
    """
    Test the calculation of projection location for datasets.
    """
    projection_locations = dicom_meta._get_projection_location()
    assert isinstance(
        projection_locations, list
    ), "Projection locations should be a list."
    assert (
        len(projection_locations) == dicom_meta.slice_count
    ), "Mismatch in number of projection locations."


def test_dicom_json_convert(dicom_files, dicom_meta):
    """
    Test JSON conversion functionality.
    """
    json_back = dicom_meta.index(0)
    json_convert = pydicom.dcmread(dicom_files[0]).to_json_dict(
        bulk_data_threshold=10240, bulk_data_element_handler=lambda x: None
    )
    json_convert.pop('7FE00010') # remove pixel data
    assert (
        json_convert == json_back
    ), "read dicom json different from json convert from dicommeta."


def test_dicom_meta_basic_operations(dummy_dicom_meta):
    """
    Test basic DicomMeta operations.
    """
    
    # Test setting shared items
    dummy_dicom_meta.set_shared_item(CommonTags.PatientName, "TEST^PATIENT")
    
    # Verify item is shared
    assert dummy_dicom_meta.is_shared(CommonTags.PatientName), "PATIENT_NAME should be shared."
    
    # Get shared value directly
    patient_name = dummy_dicom_meta.get_shared_value(CommonTags.PatientName)
    assert patient_name == "TEST^PATIENT", "Unexpected patient name value."
    
    # Get values as list
    patient_name_list = dummy_dicom_meta.get_values(CommonTags.PatientName)
    assert patient_name_list == ["TEST^PATIENT"]*dummy_dicom_meta.slice_count, "Unexpected patient name list."


def test_is_missing(dummy_dicom_meta):
    """
    Test is_missing functionality.
    """
    # This tag should be missing
    assert dummy_dicom_meta.is_missing(CommonTags.Modality), "MODALITY should be missing."
    
    # Add a tag and verify it's not missing
    dummy_dicom_meta.set_shared_item(CommonTags.Modality, "CT")
    assert not dummy_dicom_meta.is_missing(CommonTags.Modality), "MODALITY should not be missing."


def test_get_vr(dummy_dicom_meta):
    """
    Test get_vr functionality.
    """
    # Set a tag with known VR
    dummy_dicom_meta.set_shared_item(CommonTags.PatientName, "TEST^PATIENT")
    
    # Get the VR
    vr = dummy_dicom_meta.get_vr(CommonTags.PatientName)
    assert vr == "PN", "Unexpected VR for PATIENT_NAME."


def test_dicom_meta_tags_access():
    """
    Test DICOM tag access functionality.
    """
    # Test that common tags are accessible
    assert hasattr(CommonTags, 'PatientName')
    assert hasattr(CommonTags, 'Modality')
    assert hasattr(CommonTags, 'InstanceNumber')
    
    # Test tag properties
    patient_tag = CommonTags.PatientName
    assert hasattr(patient_tag, 'group')
    assert hasattr(patient_tag, 'element')
    
    # Test string format
    tag_str = f"({patient_tag.group:04X},{patient_tag.element:04X})"
    assert tag_str == "(0010,0010)", "Unexpected tag string format." 