#!/usr/bin/env python3
"""
Local Development Build Script

For developers to locally compile DiCube project's OpenJPH extensions.
Provides simplified build commands and common build options.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, cwd=None, env=None):
    """Run command and display output"""
    print(f"Executing: {' '.join(cmd)}")
    print(f"Working directory: {cwd or os.getcwd()}")
    
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=False,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Command failed with exit code: {result.returncode}")
        sys.exit(result.returncode)
    
    return result


def check_dependencies():
    """Check build dependencies"""
    print("Checking build dependencies...")
    
    # Check Python dependencies
    try:
        import numpy
        import pybind11
        print(f"✓ NumPy: {numpy.__version__}")
        print(f"✓ pybind11: {pybind11.__version__}")
    except ImportError as e:
        print(f"✗ Missing Python dependency: {e}")
        print("Please install dependencies: pip install numpy pybind11[global]")
        print("Or if using mamba: mamba install numpy pybind11")
        sys.exit(1)
    
    # Check CMake
    try:
        result = subprocess.run(
            ["cmake", "--version"], 
            capture_output=True, 
            text=True,
            check=True
        )
        version_line = result.stdout.split('\n')[0]
        print(f"✓ {version_line}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ CMake not found, please install CMake 3.12+")
        print("  mamba install cmake")
        sys.exit(1)
    
    # Check OpenJPH source code
    project_root = Path(__file__).parent.parent
    openjph_path = project_root / "source" / "OpenJPH" / "CMakeLists.txt"
    if openjph_path.exists():
        print(f"✓ OpenJPH source: {openjph_path.parent}")
    else:
        print("✗ OpenJPH source not found")
        print("Please initialize git submodule:")
        print("  git submodule update --init --recursive")
        sys.exit(1)


def build_project(build_type="Release", clean=False, parallel_jobs=None):
    """Build the project"""
    project_root = Path(__file__).parent.parent
    build_dir = project_root / "build"
    
    if clean and build_dir.exists():
        print("Cleaning build directory...")
        import shutil
        shutil.rmtree(build_dir)
    
    build_dir.mkdir(exist_ok=True)
    
    # Configuration phase
    cmake_args = [
        "cmake",
        str(project_root),
        f"-DCMAKE_BUILD_TYPE={build_type}",
        "-DOPENJPH_BUILD_STATIC=ON",
        "-DICUBE_BUILD_PYTHON_EXTENSIONS=ON",
    ]
    
    # Check for Ninja
    try:
        subprocess.run(["ninja", "--version"], capture_output=True, check=True)
        cmake_args.append("-GNinja")
        print("Using Ninja build system")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Using default build system (Make)")
    
    print("Configuring project...")
    run_command(cmake_args, cwd=build_dir)
    
    # Build phase
    build_args = ["cmake", "--build", "."]
    if parallel_jobs:
        build_args.extend(["--parallel", str(parallel_jobs)])
    elif parallel_jobs is None:
        build_args.append("--parallel")
    
    print("Building project...")
    run_command(build_args, cwd=build_dir)
    
    print(f"Build complete! Extension files are at: {project_root}/dicube/codecs/jph/")


def install_editable():
    """Install in editable mode"""
    project_root = Path(__file__).parent.parent
    
    print("Installing in editable mode...")
    run_command([sys.executable, "-m", "pip", "install", "-e", "."], cwd=project_root)
    
    print("Installation complete!")
    print("Test import:")
    print("  python -c \"import dicube; print('Import successful')\"")


def main():
    parser = argparse.ArgumentParser(description="DiCube Local Build Script")
    parser.add_argument(
        "--build-type", 
        choices=["Debug", "Release", "RelWithDebInfo"],
        default="Release",
        help="Build type (default: Release)"
    )
    parser.add_argument(
        "--clean", 
        action="store_true",
        help="Clean build directory before building"
    )
    parser.add_argument(
        "--jobs", "-j",
        type=int,
        help="Number of parallel compile jobs (default: auto-detect)"
    )
    parser.add_argument(
        "--install", 
        action="store_true",
        help="Install in editable mode after build"
    )
    parser.add_argument(
        "--check-only",
        action="store_true", 
        help="Only check dependencies, don't build"
    )
    
    args = parser.parse_args()
    
    print("=== DiCube Local Build Script ===")
    
    # Check dependencies
    check_dependencies()
    
    if args.check_only:
        print("Dependency check complete!")
        return
    
    # Build project
    build_project(
        build_type=args.build_type,
        clean=args.clean,
        parallel_jobs=args.jobs
    )
    
    # Install if requested
    if args.install:
        install_editable()
    
    print("=== Build Complete ===")


if __name__ == "__main__":
    main() 