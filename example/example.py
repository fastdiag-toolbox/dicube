import numpy as np
import pandas as pd
import time
import tempfile
import os
from pathlib import Path

# DICOM libraries
import pydicom
import dicube

# NIfTI libraries  
import nibabel as nib

# os.chdir("../")

print("✅ All libraries imported successfully")
print(os.getcwd())


def load_dicom_with_pydicom(dicom_folder):
    """Load DICOM files using pydicom and return pixel arrays"""
    dicom_files = sorted([f for f in Path(dicom_folder).glob("*.dcm")])
    datasets = []
    pixel_arrays = []
    
    for dcm_file in dicom_files:
        ds = pydicom.dcmread(dcm_file)
        datasets.append(ds)
        pixel_arrays.append(ds.pixel_array)
    
    return datasets, np.array(pixel_arrays)

def save_dicom_with_pydicom(datasets, pixel_arrays, output_folder):
    """Save DICOM files using pydicom"""
    os.makedirs(output_folder, exist_ok=True)
    
    for i, (ds, pixel_array) in enumerate(zip(datasets, pixel_arrays)):
        # Update pixel data
        ds.PixelData = pixel_array.tobytes()
        
        # Save to new location
        output_path = Path(output_folder) / f"slice_{i:04d}.dcm"
        ds.save_as(output_path)

def save_dicom_as_nifti(datasets, pixel_arrays, output_file):
    """Save DICOM data as NIfTI file"""
    # Get the first dataset for metadata
    first_ds = datasets[0]
    
    # Create affine matrix from DICOM metadata
    # This is a simplified affine matrix - you might need to adjust based on your DICOM data
    pixel_spacing = getattr(first_ds, 'PixelSpacing', [1.0, 1.0])
    slice_thickness = getattr(first_ds, 'SliceThickness', 1.0)
    
    # Create a basic affine matrix
    affine = np.array([
        [pixel_spacing[0], 0, 0, 0],
        [0, pixel_spacing[1], 0, 0],
        [0, 0, slice_thickness, 0],
        [0, 0, 0, 1]
    ])
    
    # Create NIfTI image
    nii = nib.Nifti1Image(pixel_arrays, affine)
    
    # Save as .nii.gz
    nib.save(nii, output_file)
    
    return nii

def save_dicom_as_nifti_advanced(datasets, pixel_arrays, output_file):
    """Save DICOM data as NIfTI file with better geometric information"""
    if len(datasets) == 0:
        raise ValueError("No DICOM datasets provided")
    
    # Get the first dataset for metadata
    first_ds = datasets[0]
    
    # Extract DICOM geometric information
    pixel_spacing = getattr(first_ds, 'PixelSpacing', [1.0, 1.0])
    slice_thickness = getattr(first_ds, 'SliceThickness', 1.0)
    
    # Get image orientation and position if available
    image_orientation = getattr(first_ds, 'ImageOrientationPatient', [1, 0, 0, 0, 1, 0])
    image_position = getattr(first_ds, 'ImagePositionPatient', [0, 0, 0])
    
    # Create a more accurate affine matrix
    # This is a simplified version - for production use, you might want to use dicom2nifti library
    affine = np.array([
        [pixel_spacing[0], 0, 0, image_position[0]],
        [0, pixel_spacing[1], 0, image_position[1]],
        [0, 0, slice_thickness, image_position[2]],
        [0, 0, 0, 1]
    ])
    
    # Create NIfTI image
    nii = nib.Nifti1Image(pixel_arrays, affine)
    
    # Add some header information
    header = nii.header
    header.set_xyzt_units('mm', 'sec')  # Set units to millimeters and seconds
    
    # Save as .nii.gz
    nib.save(nii, output_file)
    
    return nii

print("✅ PyDICOM helper functions defined")


# Test data path
dicom_folder = "./testdata2/dicube-testdata/dicom/sample_150"

if not os.path.exists(dicom_folder):
    print(f"❌ DICOM folder not found: {dicom_folder}")
else:
    print(f"✅ DICOM folder found: {dicom_folder}")
    
    results = []
    
    # with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = "../tmpdir"
    tmpdir = Path(tmpdir)
    
    # ======= PyDICOM Method =======
    print("\n🔄 Testing PyDICOM...")
    
    # Load with pydicom
    start_time = time.time()
    datasets, pixel_arrays = load_dicom_with_pydicom(dicom_folder)
    pydicom_load_time = time.time() - start_time
    
    print(f"  Loaded {len(datasets)} DICOM files")
    print(f"  Shape: {pixel_arrays.shape}, dtype: {pixel_arrays.dtype}")
    print(f"  Load time: {pydicom_load_time:.3f}s")
    
    # Save with pydicom
    pydicom_output = tmpdir / "pydicom_output"
    start_time = time.time()
    save_dicom_with_pydicom(datasets, pixel_arrays, pydicom_output)
    pydicom_save_time = time.time() - start_time
    
    # Calculate folder size
    pydicom_size = sum(f.stat().st_size for f in pydicom_output.glob("*.dcm"))
    
    print(f"  Save time: {pydicom_save_time:.3f}s")
    print(f"  Folder size: {pydicom_size / 1024 / 1024:.2f} MB")
    
    results.append({
        'Method': 'PyDICOM',
        'Load Time (s)': pydicom_load_time,
        'Save Time (s)': pydicom_save_time,
        'File Size (MB)': pydicom_size / 1024 / 1024,
        'Compression Ratio': pixel_arrays.nbytes / pydicom_size
    })
    
    # ======= Save DICOM as NIfTI =======
    print("\n🔄 Saving DICOM as NIfTI...")
    
    # Save as NIfTI file using advanced function
    nifti_output = tmpdir / "dicom_to_nifti.nii.gz"
    start_time = time.time()
    try:
        nii = save_dicom_as_nifti_advanced(datasets, pixel_arrays, nifti_output)
        nifti_save_time = time.time() - start_time
        
        nifti_size = nifti_output.stat().st_size
        print(f"  NIfTI save time: {nifti_save_time:.3f}s")
        print(f"  NIfTI file size: {nifti_size / 1024 / 1024:.2f} MB")
        print(f"  NIfTI file saved to: {nifti_output}")
        print(f"  NIfTI shape: {nii.shape}")
        print(f"  NIfTI affine matrix:\n{nii.affine}")
        
        results.append({
            'Method': 'DICOM to NIfTI',
            'Load Time (s)': pydicom_load_time,
            'Save Time (s)': nifti_save_time,
            'File Size (MB)': nifti_size / 1024 / 1024,
            'Compression Ratio': pixel_arrays.nbytes / nifti_size
        })
        
    except Exception as e:
        print(f"  ❌ Error saving NIfTI file: {e}")
        # Fallback to basic function
        print("  🔄 Trying basic NIfTI conversion...")
        nii = save_dicom_as_nifti(datasets, pixel_arrays, nifti_output)
        nifti_save_time = time.time() - start_time
        
        nifti_size = nifti_output.stat().st_size
        print(f"  Basic NIfTI save time: {nifti_save_time:.3f}s")
        print(f"  Basic NIfTI file size: {nifti_size / 1024 / 1024:.2f} MB")
        print(f"  Basic NIfTI file saved to: {nifti_output}")
        
        results.append({
            'Method': 'DICOM to NIfTI (Basic)',
            'Load Time (s)': pydicom_load_time,
            'Save Time (s)': nifti_save_time,
            'File Size (MB)': nifti_size / 1024 / 1024,
            'Compression Ratio': pixel_arrays.nbytes / nifti_size
        })
    
    # ======= DicomCubeImage Method =======
    print("\n🔄 Testing DicomCubeImage...")
    
    # Load with DicomCubeImage
    start_time = time.time()
    dicube_image = dicube.load_from_dicom_folder(dicom_folder)
    dicube_load_time = time.time() - start_time
    
    print(f"  Shape: {dicube_image.shape}, dtype: {dicube_image.raw_image.dtype}")
    print(f"  Load time: {dicube_load_time:.3f}s")
    
    # Save as DCB file
    dcb_file = tmpdir / "test.dcb"
    start_time = time.time()
    dicube.save(dicube_image, str(dcb_file), file_type="s")  # "s" for speed (OJPH)
    dicube_save_time = time.time() - start_time
    
    # Load DCB file
    start_time = time.time()
    dcb_file = Path(str(dcb_file)+".dcbs")
    loaded_image = dicube.load(str(dcb_file))
    dcb_load_time = time.time() - start_time
    
    dcb_size = dcb_file.stat().st_size
    
    print(f"  DCB save time: {dicube_save_time:.3f}s")
    print(f"  DCB load time: {dcb_load_time:.3f}s")
    print(f"  DCB file size: {dcb_size / 1024 / 1024:.2f} MB")
    
    # Verify data consistency
    data_consistent = np.array_equal(dicube_image.raw_image, loaded_image.raw_image)
    print(f"  Data consistent: {data_consistent}")
    
    results.append({
        'Method': 'DCB (OJPH)',
        'Load Time (s)': dcb_load_time,
        'Save Time (s)': dicube_save_time,
        'File Size (MB)': dcb_size / 1024 / 1024,
        'Compression Ratio': pixel_arrays.nbytes / dcb_size
    })
    
    # Display comparison table
    df = pd.DataFrame(results)
    df = df.round(3)
    print("\n📊 DICOM Comparison Results:")
    print(df.to_string(index=False))

def load_nifti_with_nibabel(nifti_file):
    """Load NIfTI file using nibabel"""
    nii = nib.load(nifti_file)
    data = np.asarray(nii.dataobj, dtype=nii.dataobj.dtype)
    return nii, data

def save_nifti_with_nibabel(data, affine, output_file):
    """Save NIfTI file using nibabel"""
    nii = nib.Nifti1Image(data, affine)
    nib.save(nii, output_file)

print("✅ NiBabel helper functions defined")

# Test data path
nifti_file = "./example/dicom_converted.nii.gz"
if not os.path.exists(nifti_file):
    #读取dcm，写一个nifti文件
    dicom_folder = "./testdata2/dicube-testdata/dicom/sample_150"
    datasets, pixel_arrays = load_dicom_with_pydicom(dicom_folder)
    save_dicom_as_nifti(datasets, pixel_arrays, nifti_file)


if not os.path.exists(nifti_file):
    print(f"❌ NIfTI file not found: {nifti_file}")
else:
    print(f"✅ NIfTI file found: {nifti_file}")
    
    results = []
    original_size = os.path.getsize(nifti_file)
    print(f"  Original file size: {original_size / 1024 / 1024:.2f} MB")
    
    # with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = "../tmpdir"
    tmpdir = Path(tmpdir)
    
    # ======= NiBabel Method =======
    print("\n🔄 Testing NiBabel...")
    
    # Load with nibabel
    start_time = time.time()
    nii, nifti_data = load_nifti_with_nibabel(nifti_file)
    nibabel_load_time = time.time() - start_time
    
    print(f"  Shape: {nifti_data.shape}, dtype: {nifti_data.dtype}")
    print(f"  Load time: {nibabel_load_time:.3f}s")
    
    # Save with nibabel
    nibabel_output = tmpdir / "nibabel_output.nii.gz"
    start_time = time.time()
    save_nifti_with_nibabel(nifti_data, nii.affine, nibabel_output)
    nibabel_save_time = time.time() - start_time
    
    nibabel_size = nibabel_output.stat().st_size
    
    print(f"  Save time: {nibabel_save_time:.3f}s")
    print(f"  File size: {nibabel_size / 1024 / 1024:.2f} MB")
    
    results.append({
        'Method': 'NiBabel',
        'Load Time (s)': nibabel_load_time,
        'Save Time (s)': nibabel_save_time,
        'File Size (MB)': nibabel_size / 1024 / 1024,
        'Compression Ratio': nifti_data.nbytes / nibabel_size
    })
    
    # ======= DicomCubeImage Method =======
    print("\n🔄 Testing DicomCubeImage...")
    
    # Load with DicomCubeImage
    start_time = time.time()
    dicube_image = dicube.load_from_nifti(nifti_file)
    dicube_load_time = time.time() - start_time
    
    print(f"  Shape: {dicube_image.shape}, dtype: {dicube_image.raw_image.dtype}")
    print(f"  Load time: {dicube_load_time:.3f}s")
    
    # Save as DCB file
    dcb_file = tmpdir / "test_nifti.dcb"
    start_time = time.time()
    dicube.save(dicube_image, str(dcb_file), file_type="s")  # "s" for speed (OJPH)
    dicube_save_time = time.time() - start_time
    
    # Load DCB file
    start_time = time.time()
    dcb_file = Path(str(dcb_file)+".dcbs")
    loaded_image = dicube.load(str(dcb_file))
    dcb_load_time = time.time() - start_time
    
    dcb_size = dcb_file.stat().st_size
    
    print(f"  DCB save time: {dicube_save_time:.3f}s")
    print(f"  DCB load time: {dcb_load_time:.3f}s")
    print(f"  DCB file size: {dcb_size / 1024 / 1024:.2f} MB")
    
    # Verify data consistency
    data_consistent = np.array_equal(dicube_image.raw_image, loaded_image.raw_image)
    print(f"  Data consistent: {data_consistent}")
    
    results.append({
        'Method': 'DCB (OJPH)',
        'Load Time (s)': dcb_load_time,
        'Save Time (s)': dicube_save_time,
        'File Size (MB)': dcb_size / 1024 / 1024,
        'Compression Ratio': nifti_data.nbytes / dcb_size
    })
    

# Display comparison table
df = pd.DataFrame(results)
df = df.round(3)
print("\n📊 NIfTI Comparison Results:")
print(df.to_string(index=False))