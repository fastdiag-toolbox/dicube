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
|   ├── io.py            # DicomCubeImageIO (from and to many file formats)
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
│   └── jph/            # HTJ2K codec (dcbs format)
└── exceptions.py        # Custom exceptions
```

## File Formats

DiCube defines three file format specifications for different use cases:

### .dcbs (Speed format) - **Currently Implemented**
- **Magic**: `DCMCUBES`
- **Target**: I/O speed suitable for deep learning training while high compression ratio.
- **Codec**: High Throughput JPEG 2000 (HTJ2K)
- **Use case**: High-speed encoding/decoding for processing pipelines
- **Features**: Optimized for throughput, lossless compression

### .dcba (Archive format) - **Placeholder**
- **Magic**: `DCMCUBEA`
- **Target**: 20% better compression ratio than dcbs
- **Use case**: Long-term storage and archiving
- **Status**: Awaiting suitable codec that meets compression targets

### .dcbl (Lossy format) - **Placeholder**
- **Magic**: `DCMCUBEL`
- **Target**: 60%+ compression ratio with imperceptible quality loss
- **Use case**: High-compression scenarios where minor quality trade-offs are acceptable
- **Status**: Awaiting suitable codec that meets quality/compression targets

> **Codec Selection Philosophy**: We take a conservative approach to codec adoption, requiring extensive testing and clear performance benefits before implementation. This avoids the complexity issues seen in DICOM's numerous format variations.

## Key Classes and Interfaces

### DicomCubeImage
Main interface for medical image handling:

```python
import dicube

# Create from DICOM directory
image = dicube.load_from_dicom_folder('path/to/dicom/')

# Create from NIfTI file
image = dicube.load_from_nifti('image.nii.gz')

# Save to compressed format (currently only dcbs is implemented)
dicube.save(image, 'output.dcbs', file_type='s')  # HTJ2K (Speed format)

# Load from file
loaded_image = dicube.load('output.dcbs')

# Export back to DICOM
dicube.save_to_dicom_folder(image, 'output_dicom/')

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



## Integration with spacetransformer

DiCube uses `spacetransformer.Space` for 3D coordinate system handling:

```python
from spacetransformer import Space

# DicomCubeImage automatically creates Space from DICOM
image = dicube.load_from_dicom_folder('dicom/')
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

### HTJ2K (jph/)
- **Status**: Currently implemented for .dcbs format
- **Files**: `_encode.py`, `_decode.py`, pybind11 bindings  
- **Functions**: `imencode_jph()`, `imdecode_jph()`
- **Build**: Uses pybind11 for C++ bindings to OpenJPH library
- **Performance**: Optimized for high-speed encoding/decoding

## Best Practices

### For Medical Images
1. Always preserve DICOM metadata when possible
2. Currently use `.dcbs` format for all storage needs (fast HTJ2K)
3. Check DICOM status before processing: `get_dicom_status(meta)`
4. Monitor for updates as `.dcba` and `.dcbl` formats become available

### For Integration
1. Use spacetransformer for all spatial operations
2. Use medmask for segmentation mask processing  
3. Convert coordinates between voxel and world space using `Space` transforms
4. Validate file format compatibility with `dicube.load()`

### Performance Tips
1. Use `num_threads` parameter for parallel compression
2. For large datasets, process in chunks to manage memory
3. Check DicomStatus before processing to avoid corrupted data
4. Use HTJ2K's high-speed capabilities for processing pipelines

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
    image = dicube.load('corrupted.dcbs')
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
- OpenJPH library: For .dcbs format implementation
- nibabel: For NIfTI file support

 