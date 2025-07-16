from pydicom.tag import Tag
from typing import Union, Tuple, Set


def get_tag_key(tag: Tag) -> str:
    """Get the hexadecimal string representation of a DICOM Tag (format: 'ggggeeee').
    
    Args:
        tag: pydicom Tag object
        
    Returns:
        str: Hexadecimal string, e.g., '00100020' for PatientID
    """
    return f"{tag:08X}"  # or format(tag, "08X")


class CommonTags:
    """Common DICOM tags used throughout the library.
    
    This class provides convenient access to frequently used DICOM tags
    organized by category (patient, study, series, instance, etc.).
    All tags are pydicom Tag objects.
    """
    
    # Patient tags
    PatientID = Tag("PatientID")
    PatientName = Tag("PatientName")
    PatientBirthDate = Tag("PatientBirthDate")
    PatientSex = Tag("PatientSex")
    PatientAge = Tag("PatientAge")
    PatientWeight = Tag("PatientWeight")
    
    # Study tags
    StudyInstanceUID = Tag("StudyInstanceUID")
    StudyID = Tag("StudyID")
    StudyDate = Tag("StudyDate")
    StudyTime = Tag("StudyTime")
    AccessionNumber = Tag("AccessionNumber")
    StudyDescription = Tag("StudyDescription")
    
    # Series tags
    SeriesInstanceUID = Tag("SeriesInstanceUID")
    SeriesNumber = Tag("SeriesNumber")
    Modality = Tag("Modality")
    SeriesDescription = Tag("SeriesDescription")
    
    # Instance tags
    SOPInstanceUID = Tag("SOPInstanceUID")
    SOPClassUID = Tag("SOPClassUID")
    InstanceNumber = Tag("InstanceNumber")
    
    # Image tags
    Rows = Tag("Rows")
    Columns = Tag("Columns")
    BitsAllocated = Tag("BitsAllocated")
    BitsStored = Tag("BitsStored")
    HighBit = Tag("HighBit")
    SamplesPerPixel = Tag("SamplesPerPixel")
    PhotometricInterpretation = Tag("PhotometricInterpretation")
    PixelRepresentation = Tag("PixelRepresentation")
    
    # Spatial tags
    ImagePositionPatient = Tag("ImagePositionPatient")
    ImageOrientationPatient = Tag("ImageOrientationPatient")
    PixelSpacing = Tag("PixelSpacing")
    SliceThickness = Tag("SliceThickness")
    SpacingBetweenSlices = Tag("SpacingBetweenSlices")
    SliceLocation = Tag("SliceLocation")
    
    # Value transformations
    RescaleIntercept = Tag("RescaleIntercept")
    RescaleSlope = Tag("RescaleSlope")
    WindowCenter = Tag("WindowCenter")
    WindowWidth = Tag("WindowWidth")
    PatientPosition = Tag("PatientPosition")
    BodyPartExamined = Tag("BodyPartExamined")
    
    # Pixel data
    PixelData = Tag("PixelData")
    
    # Enhanced MR specific tags
    DimensionIndexSequence = Tag("DimensionIndexSequence")
    FrameContentSequence = Tag("FrameContentSequence")
    
    # UID tags
    ImplementationClassUID = Tag("ImplementationClassUID")
    
    # Other important tags
    TransferSyntaxUID = Tag("TransferSyntaxUID")
    MediaStorageSOPClassUID = Tag("MediaStorageSOPClassUID")
    MediaStorageSOPInstanceUID = Tag("MediaStorageSOPInstanceUID")
    SpecificCharacterSet = Tag("SpecificCharacterSet")
        
    # Manufacturer Information
    Manufacturer = Tag("Manufacturer")
    ManufacturerModelName = Tag("ManufacturerModelName")
    SoftwareVersions = Tag("SoftwareVersions")
    
    # Other Common Tags
    FrameOfReferenceUID = Tag("FrameOfReferenceUID")
    ReferencedImageSequence = Tag("ReferencedImageSequence")
    ReferencedSOPInstanceUID = Tag("ReferencedSOPInstanceUID")
    AcquisitionNumber = Tag("AcquisitionNumber")
    ContrastBolusAgent = Tag("ContrastBolusAgent")
    
    # Tag sets for hierarchical DICOM levels
    PATIENT_LEVEL_TAGS: Set[Tag] = {
        PatientID, PatientName, PatientBirthDate, PatientSex
    }
    
    STUDY_LEVEL_TAGS: Set[Tag] = {
        StudyInstanceUID, StudyID, StudyDate, StudyTime, AccessionNumber
    }
    
    SERIES_LEVEL_TAGS: Set[Tag] = {
        SeriesInstanceUID, SeriesNumber, Modality
    }
    
    INSTANCE_LEVEL_TAGS: Set[Tag] = {
        SOPInstanceUID, SOPClassUID, InstanceNumber
    } 