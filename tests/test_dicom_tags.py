import re
from typing import Tuple

import pytest
import pydicom
from pydicom.tag import Tag
from pydicom import datadict

from dicube.dicom.dicom_tags import CommonTags, get_tag_key


def test_tag_from_tag() -> None:
    """Test creating a Tag from Tag object."""
    original_tag = Tag(0x0010, 0x0020)  # PatientID
    tag = Tag(original_tag)
    assert tag == original_tag
    assert tag.group == 0x0010
    assert tag.element == 0x0020


def test_tag_from_tuple() -> None:
    """Test creating a Tag from (group, element) tuple."""
    tag_tuple = (0x0010, 0x0020)  # PatientID
    tag = Tag(tag_tuple)
    assert tag.group == 0x0010
    assert tag.element == 0x0020


def test_tag_from_hex_string() -> None:
    """Test creating a Tag from hex string."""
    # Test 8-character hex string format
    hex_string = "00100020"  # PatientID
    tag = Tag(hex_string)
    assert tag.group == 0x0010
    assert tag.element == 0x0020


def test_tag_from_dicom_format() -> None:
    """Test creating a Tag from DICOM format string."""
    # Test (gggg,eeee) format
    dicom_format = "(0010,0020)"  # PatientID
    
    # Extract group and element from the DICOM format string
    match = re.match(r"\(([0-9A-Fa-f]{4}),([0-9A-Fa-f]{4})\)", dicom_format)
    if match:
        group = int(match.group(1), 16)
        element = int(match.group(2), 16)
        tag = Tag((group, element))
        assert tag.group == 0x0010
        assert tag.element == 0x0020
    else:
        pytest.fail("Failed to parse DICOM format string")


def test_tag_from_keyword() -> None:
    """Test creating a Tag from DICOM keyword."""
    keyword = "PatientID"
    tag = Tag(keyword)
    assert tag.group == 0x0010
    assert tag.element == 0x0020


def test_tag_invalid() -> None:
    """Test creating a Tag from invalid input."""
    with pytest.raises(ValueError):
        Tag("not_a_tag")


def test_get_tag_name() -> None:
    """Test getting the name (keyword) for a tag."""
    tag = Tag(0x0010, 0x0020)  # PatientID
    name = datadict.keyword_for_tag(tag)
    assert name == "PatientID"


def test_get_tag_vr() -> None:
    """Test getting the VR for a tag."""
    tag = Tag(0x0010, 0x0020)  # PatientID
    vr = datadict.dictionary_VR(tag)
    assert vr == "LO"  # Long String

    tag = Tag(0x0010, 0x0010)  # PatientName
    vr = datadict.dictionary_VR(tag)
    assert vr == "PN"  # Person Name


def test_common_tags() -> None:
    """Test the CommonTags class."""
    # Test a few common tags
    assert CommonTags.PatientName.group == 0x0010
    assert CommonTags.PatientName.element == 0x0010

    assert CommonTags.PatientID.group == 0x0010
    assert CommonTags.PatientID.element == 0x0020

    assert CommonTags.Modality.group == 0x0008
    assert CommonTags.Modality.element == 0x0060


def test_tag_equality() -> None:
    """Test tag equality."""
    tag1 = Tag("PatientID")
    tag2 = Tag((0x0010, 0x0020))
    tag3 = Tag("00100020")
    
    assert tag1 == tag2
    assert tag2 == tag3
    assert tag1 == tag3


def test_get_tag_key() -> None:
    """Test getting a tag key in hexadecimal format."""
    tag = Tag("PatientID")
    key = get_tag_key(tag)
    assert key == "00100020"
    
    tag = Tag("StudyInstanceUID")
    key = get_tag_key(tag)
    assert key == "0020000D" 