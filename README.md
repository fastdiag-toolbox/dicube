# DiCube: Medical Image Storage Library

DiCube is a Python library for efficient storage and processing of 3D medical images with complete DICOM metadata preservation. It provides a high-compression, single-file format that combines DICOM compatibility with modern compression techniques.

## Overview

DiCube was extracted from the larger DICOMCube project to focus specifically on medical image storage. It works alongside:
- **spacetransformer**: For 3D spatial transformations and coordinate systems
- **medmask**: For medical image segmentation mask processing

## Architecture

### Core Modules

```
dicube/
├── core/                 # Core data structures
│   ├── image.py         # DicomCubeImage (main interface)
│   └── pixel_header.py  # PixelDataHeader (image metadata)
├── storage/             # File storage formats
│   ├── dcb_file.py      # DCB file format implementations
│   └── pixel_utils.py   # Pixel processing utilities
├── dicom/               # DICOM functionality
│   ├── dicom_meta.py    # DicomMeta (metadata container)
│   ├── dicom_status.py  # DICOM consistency checking
│   ├── dicom_tags.py    # DICOM tag definitions
│   ├── dicom_io.py      # DICOM file I/O
│   └── merge_utils.py   # Metadata merging utilities
├── codecs/              # Compression codecs
│   ├── jxl/            # JPEG XL codec (dcba format)
│   └── jph/            # HTJ2K codec (dcbs format)
└── exceptions.py        # Custom exceptions
```

## File Formats

DiCube supports multiple file formats optimized for different use cases:

### .dcba (JPEG XL format)
- **Magic**: `DCMCUBEA`
- **Codec**: JPEG XL compression
- **Use case**: High-quality images with excellent compression
- **Features**: Lossless and lossy modes, HDR support

### .dcbs (HTJ2K format)  
- **Magic**: `DCMCUBES`
- **Codec**: High Throughput JPEG 2000
- **Use case**: High-speed encoding/decoding
- **Features**: Optimized for throughput

### .dcbr (ROI format)
- **Magic**: `DCMCUBER` 
- **Codec**: JPEG XL with separate foreground/background
- **Use case**: Region-of-interest processing
- **Features**: Different quality settings for FG/BG

## Key Classes and Interfaces

### DicomCubeImage
Main interface for medical image handling:

```python
from dicube import DicomCubeImage

# Create from DICOM directory
image = DicomCubeImage.from_dicom_folder('path/to/dicom/')

# Create from NIfTI file
image = DicomCubeImage.from_nifti('image.nii.gz')

# Save to compressed format
image.to_file('output.dcba')  # JPEG XL
image.to_file('output.dcbs')  # HTJ2K

# Load from file
loaded_image = DicomCubeImage.from_file('output.dcba')

# Export back to DICOM
image.to_dicom_folder('output_dicom/')

# Get pixel data
pixel_data = image.get_fdata()  # Returns float array
raw_data = image.raw_image       # Returns original dtype
```

### DicomMeta
DICOM metadata container with efficient shared/non-shared value handling:

```python
from dicube import DicomMeta, read_dicom_dir

# Read DICOM directory
meta = read_dicom_dir('dicom_folder/')

# Access shared values (same across all slices)
patient_name = meta.get('PatientName')  # Returns single value

# Access non-shared values (different per slice)
positions = meta.get('ImagePositionPatient', force_nonshared=True)  # Returns list

# Check status
from dicube import get_dicom_status
status = get_dicom_status(meta)
```

### File Format Structure

All DiCube formats share a common binary structure:

```
Header (100 bytes):
├── magic: 8 bytes
├── version: 4 bytes  
├── dicom_status_offset: 8 bytes
├── dicom_status_length: 8 bytes
├── dicommeta_offset: 8 bytes
├── dicommeta_length: 8 bytes
├── space_offset: 8 bytes
├── space_length: 8 bytes
├── pixel_header_offset: 8 bytes
├── pixel_header_length: 8 bytes
├── frame_offsets_offset: 8 bytes
├── frame_offsets_length: 8 bytes
├── frame_lengths_offset: 8 bytes
├── frame_lengths_length: 8 bytes
└── frame_count: 8 bytes

Data sections:
├── DicomStatus (text)
├── DicomMeta (compressed JSON)
├── Space (JSON)
├── PixelDataHeader (JSON)
├── Frame offsets (binary)
├── Frame lengths (binary)
└── Compressed frame data
```

## Integration with spacetransformer

DiCube uses `spacetransformer.Space` for 3D coordinate system handling:

```python
from spacetransformer import Space

# DicomCubeImage automatically creates Space from DICOM
image = DicomCubeImage.from_dicom_folder('dicom/')
space = image.space  # spacetransformer.Space object

# Apply spatial transformations
transformed_space = space.apply_flip(axis=2)
transformed_space = space.apply_rotate(axis=0, angle=90, unit='degree')

# Update image with new space
image.space = transformed_space
```

## DICOM Status Checking

DiCube provides comprehensive DICOM consistency checking:

```python
from dicube import DicomStatus, get_dicom_status

status = get_dicom_status(meta)

# Possible status values:
# DicomStatus.CONSISTENT - All checks pass
# DicomStatus.MISSING_SERIES_UID - No series UID
# DicomStatus.DUPLICATE_INSTANCE_NUMBERS - Non-unique instance numbers
# DicomStatus.NON_UNIFORM_SPACING - Inconsistent pixel spacing
# DicomStatus.GAP_LOCATION - Missing slices in Z direction
# ... and more
```

## Compression Codecs

### JPEG XL (jxl/)
- Files: `imageio_jxl.py`, Cython bindings
- Functions: `imencode_jxl()`, `imdecode_jxl()`, `imencode_jxl_roi()`, `imdecode_jxl_roi()`
- Build: Uses Cython for C++ bindings

### HTJ2K (jph/)
- Files: `_encode.py`, `_decode.py`, pybind11 bindings  
- Functions: `imencode_jph()`, `imdecode_jph()`
- Build: Uses pybind11 for C++ bindings

## Best Practices

### For Medical Images
1. Always preserve DICOM metadata when possible
2. Use `.dcba` format for archival storage (lossless JPEG XL)
3. Use `.dcbs` format for processing pipelines (fast HTJ2K)
4. Check DICOM status before processing: `get_dicom_status(meta)`

### For Integration
1. Use spacetransformer for all spatial operations
2. Use medmask for segmentation mask processing  
3. Convert coordinates between voxel and world space using `Space` transforms
4. Validate file format compatibility with `DicomCubeImage.from_file()`

### Performance Tips
1. Use `num_threads` parameter for parallel compression
2. For large datasets, process in chunks to manage memory
3. Check DicomStatus before processing to avoid corrupted data
4. Use ROI format (.dcbr) for images with distinct foreground/background

## Error Handling

```python
from dicube.exceptions import (
    DicomCubeError,
    InvalidCubeFileError, 
    CodecError,
    MetaDataError,
    DataConsistencyError
)

try:
    image = DicomCubeImage.from_file('corrupted.dcba')
except InvalidCubeFileError:
    print("Not a valid DiCube file")
except CodecError:
    print("Compression/decompression failed")
except MetaDataError:
    print("Missing or invalid metadata")
```

## Dependencies

### Required
- numpy: Array operations
- pydicom: DICOM file handling
- spacetransformer: Spatial transformations
- zstandard: Metadata compression

### Optional (for full functionality)
- JPEG XL library: For .dcba format
- OpenJPH library: For .dcbs format
- nibabel: For NIfTI file support

## Migration Notes

This library was extracted from DICOMCube with the following changes:
1. **Removed**: All mask-related functionality (moved to medmask)
2. **Removed**: 3D morphological operations 
3. **Updated**: Space handling now uses spacetransformer
4. **Simplified**: Focus on core image storage functionality
5. **Maintained**: Full backward compatibility for .dcb* file formats

## Build System

The library includes C++ extensions that require compilation:

### Cython Extensions (jxl/)
- `_jpegxl.pyx` → `_jpegxl.cpython-*.so`
- `_shared.pyx` → `_shared.cpython-*.so`

### pybind11 Extensions (jph/)  
- `encode_complete.cpp` → `ojph_complete.cpython-*.so`
- `decode_complete.cpp` → `ojph_decode_complete.cpython-*.so`

Build configuration should handle cross-platform compilation and library linking for JPEG XL and OpenJPH dependencies. 