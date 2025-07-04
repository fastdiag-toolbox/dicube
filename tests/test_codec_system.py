"""Tests for the unified codec system."""

import pytest
import numpy as np
from dicube.codecs import get_codec, list_codecs, register_codec
from dicube.codecs import ImageCodec


class TestCodecSystem:
    """Test the unified codec system."""
    
    def test_list_codecs(self):
        """Test listing available codecs."""
        codecs = list_codecs()
        assert isinstance(codecs, list)
        assert len(codecs) > 0
        # Should contain at least jph (which is usually available)
        assert 'jph' in codecs
    
    def test_get_codec_valid(self):
        """Test getting a valid codec."""
        # Test with a codec that should be available
        codec = get_codec('jph')
        assert codec.name == 'jph'
        assert codec.id == 2
        assert hasattr(codec, 'encode')
        assert hasattr(codec, 'decode')
    
    def test_get_codec_invalid(self):
        """Test getting an invalid codec."""
        with pytest.raises(ValueError) as excinfo:
            get_codec('nonexistent')
        assert "Unknown codec" in str(excinfo.value)
        assert "nonexistent" in str(excinfo.value)
    
    def test_get_codec_case_insensitive(self):
        """Test that codec names are case insensitive."""
        codec1 = get_codec('jph')
        codec2 = get_codec('JPH')
        codec3 = get_codec('Jph')
        assert codec1.name == codec2.name == codec3.name
    

    
    def test_codec_interface(self):
        """Test that codecs implement the ImageCodec interface."""
        for codec_name in list_codecs():
            codec = get_codec(codec_name)
            
            # Check required attributes
            assert hasattr(codec, 'id')
            assert hasattr(codec, 'name')
            assert hasattr(codec, 'extensions')
            assert isinstance(codec.id, int)
            assert isinstance(codec.name, str)
            assert isinstance(codec.extensions, tuple)
            
            # Check required methods
            assert hasattr(codec, 'encode')
            assert hasattr(codec, 'decode')
            assert hasattr(codec, 'is_available')
            assert hasattr(codec, 'get_version')
    
    def test_codec_functionality(self):
        """Test basic codec functionality."""
        # Use a codec that should be available
        codec = get_codec('jph')
        
        if not codec.is_available():
            pytest.skip(f"Codec {codec.name} is not available")
        
        # Create test image
        test_image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        
        # Test encode/decode
        encoded = codec.encode(test_image)
        assert isinstance(encoded, bytes)
        assert len(encoded) > 0
        
        decoded = codec.decode(encoded)
        assert isinstance(decoded, np.ndarray)
        assert decoded.shape == test_image.shape
        
        # Test version info
        version = codec.get_version()
        assert isinstance(version, str)
        assert len(version) > 0
    
    def test_codec_string_representation(self):
        """Test codec string representation."""
        codec = get_codec('jph')
        repr_str = repr(codec)
        assert 'JphCodec' in repr_str
        assert 'id=2' in repr_str
        assert "name='jph'" in repr_str


class MockCodec:
    """Mock codec for testing registration."""
    
    id = 99
    name = "mock"
    extensions = (".mock",)
    
    def encode(self, image, /, **kwargs):
        return b"mock_encoded"
    
    def decode(self, data, /, **kwargs):
        return np.zeros((10, 10), dtype=np.uint8)
    

    
    def is_available(self):
        return True
    
    def get_version(self):
        return "1.0.0"


class TestCodecRegistration:
    """Test codec registration functionality."""
    
    def test_register_codec(self):
        """Test registering a custom codec."""
        # Register mock codec
        mock_codec = MockCodec()
        register_codec(mock_codec)
        
        # Check it's in the list
        assert 'mock' in list_codecs()
        
        # Check we can get it
        retrieved = get_codec('mock')
        assert retrieved.name == 'mock'
        assert retrieved.id == 99
        
        # Check it works
        assert retrieved.is_available()
        assert retrieved.get_version() == "1.0.0"
        
        # Clean up would happen automatically in real usage
        # but for testing we'll leave it registered 