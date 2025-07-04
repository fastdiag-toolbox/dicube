#!/usr/bin/env python3
"""
Setup script for dicube - Medical Image Storage Library

DiCube provides efficient storage and processing of 3D medical images 
with complete DICOM metadata preservation using modern compression techniques.
"""

from setuptools import setup, find_packages, Extension
import os
import sys
import numpy as np

# 尝试导入pybind11
try:
    import pybind11
    from pybind11.setup_helpers import Pybind11Extension, build_ext
    from pybind11 import get_cmake_dir
    HAS_PYBIND11 = True
except ImportError:
    HAS_PYBIND11 = False

# Read README file for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as f:
        return f.read()

# Base requirements
install_requires = [
    "numpy>=1.21.0",
    "pydicom>=2.3.0",
    "zstandard>=0.18.0",
]

# Optional requirements for specific codecs
extras_require = {
    "jxl": [
        # JPEG XL support - requires building C++ extensions
        "Cython>=0.29.0",
        "pybind11>=2.6.0",
    ],
    "jph": [
        # HTJ2K support - requires building C++ extensions  
        "pybind11>=2.6.0",
    ],
    "nifti": [
        "nibabel>=3.2.0",  # For NIfTI file support
    ],
    "dev": [
        "pytest>=6.0.0",
        "pytest-cov>=2.12.0",
        "black>=21.0.0",
        "flake8>=3.9.0",
        "mypy>=0.910",
        "sphinx>=4.0.0",
        "sphinx-rtd-theme>=0.5.0",
    ],
    "all": [
        "Cython>=0.29.0",
        "pybind11>=2.6.0", 
        "nibabel>=3.2.0",
    ]
}

# C++ 扩展模块配置
ext_modules = []

if HAS_PYBIND11:
    # 编译和链接参数
    extra_compile_args = ["-O3", "-std=c++17", "-fPIC"]
    extra_link_args = []
    
    # 系统头文件路径
    include_dirs = [
        "/usr/include/jxl",
        "/usr/local/include/openjph",
        "/usr/include",
        "/usr/local/include",
        np.get_include(),  # NumPy 头文件路径
        pybind11.get_include(),  # pybind11 头文件路径
    ]
    
    # 库文件路径
    library_dirs = [
        "/usr/lib",
        "/usr/local/lib",
        "/usr/lib/x86_64-linux-gnu",
    ]
    
    # JXL 编码模块
    jxl_encode_ext = Pybind11Extension(
        "dicube.codecs.jxl.jxl_encode",
        sources=["dicube/codecs/jxl/jxl_encode.cpp"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=["jxl", "jxl_threads"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    )
    
    # JXL 解码模块
    jxl_decode_ext = Pybind11Extension(
        "dicube.codecs.jxl.jxl_decode",
        sources=["dicube/codecs/jxl/jxl_decode.cpp"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=["jxl", "jxl_threads"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    )
    
    # JPH 编码模块（保持原有的）
    jph_encode_ext = Pybind11Extension(
        "dicube.codecs.jph.ojph_complete",
        sources=["dicube/codecs/jph/encode_complete.cpp"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=["openjph"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    )
    
    # JPH 解码模块（保持原有的）
    jph_decode_ext = Pybind11Extension(
        "dicube.codecs.jph.ojph_decode_complete",
        sources=["dicube/codecs/jph/decode_complete.cpp"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=["openjph"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    )
    
    ext_modules = [
        jxl_encode_ext,
        jxl_decode_ext,
        jph_encode_ext,
        jph_decode_ext,
    ]

setup(
    name="dicube",
    version="1.0.0",
    description="Medical Image Storage Library with DICOM compatibility",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="liaofz",
    author_email="",
    url="https://github.com/username/dicube",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research", 
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext} if HAS_PYBIND11 else {},
    include_package_data=True,
    package_data={
        "dicube": [
            "*.md",
            "*.txt",
            "example/data/*",
        ],
    },
    entry_points={
        "console_scripts": [
            # Add CLI tools if needed in the future
        ],
    },
    zip_safe=False,  # Needed for C++ extensions
) 