#!/usr/bin/env python3
"""
Wheel Verification Script

Validates that built wheel packages correctly contain all necessary files and dependencies.
"""

import os
import sys
import zipfile
import tempfile
import subprocess
from pathlib import Path


def check_wheel_contents(wheel_path):
    """Check wheel package contents"""
    print(f"Checking wheel package: {wheel_path}")
    
    with zipfile.ZipFile(wheel_path, 'r') as wheel:
        files = wheel.namelist()
        
        print("\n=== Wheel Package Contents ===")
        for file in sorted(files):
            print(f"  {file}")
        
        # Check required files
        required_patterns = [
            "dicube/",
            "dicube/codecs/",
            "dicube/codecs/jph/",
            ".dist-info/",
        ]
        
        extension_files = [
            f for f in files 
            if f.startswith("dicube/codecs/jph/") and (f.endswith(".so") or f.endswith(".pyd"))
        ]
        
        print("\n=== Check Results ===")
        
        # Check required directories
        for pattern in required_patterns:
            matching_files = [f for f in files if f.startswith(pattern)]
            if matching_files:
                print(f"✓ Found {pattern}: {len(matching_files)} files")
            else:
                print(f"✗ Missing {pattern}")
        
        # Check extension files
        if extension_files:
            print(f"✓ Found extension files:")
            for ext_file in extension_files:
                print(f"    {ext_file}")
        else:
            print("✗ No compiled extension files found (.so/.pyd)")
        
        return len(extension_files) > 0


def test_wheel_installation(wheel_path):
    """Test wheel package installation and import"""
    print(f"\n=== Testing Wheel Installation ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Install in temporary environment
        env = os.environ.copy()
        env['PYTHONPATH'] = temp_dir
        
        try:
            # Install wheel
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "--target", temp_dir,
                "--force-reinstall",
                "--no-deps",
                str(wheel_path)
            ], capture_output=True, text=True, env=env)
            
            if result.returncode != 0:
                print(f"✗ Installation failed:")
                print(result.stderr)
                return False
            
            print("✓ Installation successful")
            
            # Test import
            test_script = f"""
import sys
sys.path.insert(0, '{temp_dir}')

try:
    import dicube
    print("✓ dicube import successful")
    
    from dicube.codecs.jph import JphCodec
    print("✓ JphCodec import successful")
    
    codec = JphCodec()
    print(f"✓ JphCodec instantiation successful: {{codec.name}}")
    
    if codec.is_available():
        print("✓ JphCodec functionality check passed")
        
        # Simple test
        import numpy as np
        test_image = np.ones((10, 10), dtype=np.uint8)
        encoded = codec.encode(test_image)
        decoded = codec.decode(encoded)
        
        if decoded.shape == test_image.shape:
            print("✓ Encode/decode test passed")
        else:
            print(f"✗ Encode/decode test failed: shape mismatch {{decoded.shape}} != {{test_image.shape}}")
    else:
        print("✗ JphCodec functionality check failed")
        
except ImportError as e:
    print(f"✗ Import failed: {{e}}")
except Exception as e:
    print(f"✗ Test failed: {{e}}")
"""
            
            result = subprocess.run([
                sys.executable, "-c", test_script
            ], capture_output=True, text=True, env=env)
            
            print(result.stdout)
            if result.stderr:
                print("Error output:")
                print(result.stderr)
            
            return result.returncode == 0
            
        except Exception as e:
            print(f"✗ Testing process error: {e}")
            return False


def check_dependencies(wheel_path):
    """Check wheel package dependencies"""
    print(f"\n=== Checking Dependencies ===")
    
    # Extract platform tags
    wheel_name = Path(wheel_path).name
    parts = wheel_name.split('-')
    if len(parts) >= 4:
        platform_tag = parts[-1].replace('.whl', '')
        abi_tag = parts[-2]
        python_tag = parts[-3]
        print(f"Python tag: {python_tag}")
        print(f"ABI tag: {abi_tag}")
        print(f"Platform tag: {platform_tag}")
    
    # Check shared library dependencies on Linux
    if sys.platform.startswith('linux'):
        try:
            # Extract .so files and check dependencies
            with zipfile.ZipFile(wheel_path, 'r') as wheel:
                so_files = [f for f in wheel.namelist() if f.endswith('.so')]
                
                for so_file in so_files:
                    print(f"\nChecking dependencies for {so_file}:")
                    
                    # Extract to temporary file
                    with tempfile.NamedTemporaryFile(suffix='.so', delete=False) as temp_so:
                        temp_so.write(wheel.read(so_file))
                        temp_so_path = temp_so.name
                    
                    try:
                        # Use ldd to check dependencies
                        result = subprocess.run([
                            'ldd', temp_so_path
                        ], capture_output=True, text=True)
                        
                        if result.returncode == 0:
                            lines = result.stdout.strip().split('\n')
                            for line in lines:
                                line = line.strip()
                                if '=>' in line:
                                    lib_name = line.split('=>')[0].strip()
                                    lib_path = line.split('=>')[1].strip()
                                    if 'not found' in lib_path:
                                        print(f"  ✗ {lib_name} -> {lib_path}")
                                    else:
                                        print(f"  ✓ {lib_name}")
                        else:
                            print(f"  Cannot check dependencies: {result.stderr}")
                    
                    finally:
                        os.unlink(temp_so_path)
        
        except Exception as e:
            print(f"Dependency check failed: {e}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python check_wheels.py <wheel_file>")
        sys.exit(1)
    
    wheel_path = Path(sys.argv[1])
    if not wheel_path.exists():
        print(f"Error: Wheel file does not exist: {wheel_path}")
        sys.exit(1)
    
    if not wheel_path.suffix == '.whl':
        print(f"Error: Not a valid wheel file: {wheel_path}")
        sys.exit(1)
    
    print("=== DiCube Wheel Verification Tool ===")
    
    # Check wheel contents
    contents_ok = check_wheel_contents(wheel_path)
    
    if contents_ok:
        # Test installation and import
        install_ok = test_wheel_installation(wheel_path)
        
        # Check dependencies
        check_dependencies(wheel_path)
        
        if install_ok:
            print("\n=== ✓ All Checks Passed ===")
            sys.exit(0)
        else:
            print("\n=== ✗ Functionality Tests Failed ===")
            sys.exit(1)
    else:
        print("\n=== ✗ Wheel Contents Check Failed ===")
        sys.exit(1)


if __name__ == "__main__":
    main() 