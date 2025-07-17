cmake_minimum_required(VERSION 3.15.0)

project(dicube-openjph 
    VERSION 0.1.0
    DESCRIPTION "DiCube with statically linked OpenJPH"
    LANGUAGES CXX
)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ================================================================================================
# 选项配置
# ================================================================================================

option(DICUBE_BUILD_PYTHON_EXTENSIONS "Build Python extensions" ON)

# ================================================================================================
# 编译器设置
# ================================================================================================

if(MSVC)
    # Windows MSVC 设置
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
    add_compile_options(/W3 /EHsc)
    # 静态链接 CRT
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
else()
    # GCC/Clang 设置
    add_compile_options(-fexceptions -Wall -Wextra -fPIC)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -DNDEBUG)
    endif()
endif()

# ================================================================================================
# OpenJPH 构建
# ================================================================================================

# 检查 OpenJPH 源码
set(OPENJPH_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/source/OpenJPH")
if(NOT EXISTS "${OPENJPH_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR 
        "OpenJPH source not found. Please run: git submodule update --init --recursive"
    )
endif()

# OpenJPH 构建选项
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(OJPH_ENABLE_TIFF_SUPPORT OFF CACHE BOOL "" FORCE)
set(OJPH_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(OJPH_BUILD_EXECUTABLES OFF CACHE BOOL "" FORCE)
set(OJPH_BUILD_STREAM_EXPAND OFF CACHE BOOL "" FORCE)
set(OJPH_DISABLE_SIMD OFF CACHE BOOL "" FORCE)  # 启用 SIMD 优化

# 添加 OpenJPH
add_subdirectory(${OPENJPH_SOURCE_DIR} openjph EXCLUDE_FROM_ALL)

# 设置 OpenJPH 目标和头文件
set(OPENJPH_TARGET openjph)
set(OPENJPH_INCLUDE_DIRS 
    "${OPENJPH_SOURCE_DIR}/src/core/common"
    "${CMAKE_BINARY_DIR}/openjph/src/core/common"  # 生成的配置文件
)

# ================================================================================================
# Python 扩展构建
# ================================================================================================

if(DICUBE_BUILD_PYTHON_EXTENSIONS)
    # 查找 Python 和 pybind11
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(pybind11 REQUIRED)
    
    # 查找 NumPy
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
        OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    if(NOT NUMPY_INCLUDE_DIR)
        message(FATAL_ERROR "NumPy not found")
    endif()
    
    # 构建 Python 扩展
    pybind11_add_module(ojph_complete dicube/codecs/jph/encode_complete.cpp)
    pybind11_add_module(ojph_decode_complete dicube/codecs/jph/decode_complete.cpp)
    
    # 设置包含目录和链接
    foreach(target ojph_complete ojph_decode_complete)
        target_include_directories(${target} PRIVATE
            ${OPENJPH_INCLUDE_DIRS}
            ${NUMPY_INCLUDE_DIR}
        )
        target_link_libraries(${target} PRIVATE ${OPENJPH_TARGET})
        
        # 确保 .pyd/.so 文件直接放在正确的目录，不带 Release 子目录
        if(MSVC)
            # Windows特殊处理，防止产生 Release 子目录
            set_target_properties(${target} PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/dicube/codecs/jph
                RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/dicube/codecs/jph
                RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/dicube/codecs/jph
                LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/dicube/codecs/jph
                LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/dicube/codecs/jph
                LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/dicube/codecs/jph
            )
        else()
            # Linux/macOS
            set_target_properties(${target} PROPERTIES
                LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/dicube/codecs/jph
            )
        endif()
    endforeach()
    
    message(STATUS "Python extensions configured (Python ${Python_VERSION})")
endif()

# ================================================================================================
# 构建摘要
# ================================================================================================

message(STATUS "=== Build Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Python extensions: ${DICUBE_BUILD_PYTHON_EXTENSIONS}")
if(DICUBE_BUILD_PYTHON_EXTENSIONS)
    message(STATUS "Python: ${Python_VERSION}")
endif()
message(STATUS "===========================") 