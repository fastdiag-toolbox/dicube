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

option(OPENJPH_BUILD_STATIC "Build OpenJPH as static library" ON)
option(DICUBE_BUILD_PYTHON_EXTENSIONS "Build Python extensions" ON)

# ================================================================================================
# 编译器设置
# ================================================================================================

if(MSVC)
    # Windows MSVC 设置
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
    add_compile_options(/W3 /EHsc)
    # 静态链接 CRT（避免运行时依赖）
    if(OPENJPH_BUILD_STATIC)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    endif()
else()
    # GCC/Clang 设置
    add_compile_options(
        -fexceptions
        -Wall 
        -Wextra
        -fPIC  # 位置无关代码，用于共享库
    )
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -DNDEBUG)
    endif()
endif()

# ================================================================================================
# OpenJPH 静态库构建
# ================================================================================================

# 检查 OpenJPH 源码是否存在
set(OPENJPH_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/source/OpenJPH")
if(NOT EXISTS "${OPENJPH_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR 
        "OpenJPH source not found at ${OPENJPH_SOURCE_DIR}. "
        "Please ensure the git submodule is properly initialized:\n"
        "  git submodule update --init --recursive"
    )
endif()

# 设置 OpenJPH 构建选项
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries" FORCE)
set(OJPH_ENABLE_TIFF_SUPPORT OFF CACHE BOOL "Disable TIFF support" FORCE)
set(OJPH_BUILD_TESTS OFF CACHE BOOL "Disable OpenJPH tests" FORCE)
set(OJPH_BUILD_EXECUTABLES OFF CACHE BOOL "Disable OpenJPH executables" FORCE)
set(OJPH_BUILD_STREAM_EXPAND OFF CACHE BOOL "Disable stream expand" FORCE)

# 性能优化选项（可以根据需要调整）
set(OJPH_DISABLE_SIMD OFF CACHE BOOL "Enable SIMD optimizations" FORCE)

# 添加 OpenJPH 子项目
message(STATUS "Adding OpenJPH subproject...")
add_subdirectory(${OPENJPH_SOURCE_DIR} openjph)

# 获取 OpenJPH 库目标
# OpenJPH 的主要库目标通常是 'openjph'
if(TARGET openjph)
    set(OPENJPH_TARGET openjph)
    message(STATUS "Found OpenJPH target: openjph")
elseif(TARGET OpenJPH::openjph)
    set(OPENJPH_TARGET OpenJPH::openjph)
    message(STATUS "Found OpenJPH target: OpenJPH::openjph")
else()
    # 尝试查找可能的目标名称
    get_property(ALL_TARGETS DIRECTORY ${OPENJPH_SOURCE_DIR} PROPERTY BUILDSYSTEM_TARGETS)
    message(STATUS "Available targets from OpenJPH: ${ALL_TARGETS}")
    message(FATAL_ERROR "Could not find OpenJPH library target")
endif()

# 获取 OpenJPH 头文件路径
set(OPENJPH_INCLUDE_DIRS 
    "${OPENJPH_SOURCE_DIR}/src/core/common"
    "${OPENJPH_SOURCE_DIR}/src/core/codestream"
    "${OPENJPH_SOURCE_DIR}/src/core/coding"
    "${OPENJPH_SOURCE_DIR}/src/core/transform"
    "${OPENJPH_SOURCE_DIR}/src/core/others"
    "${CMAKE_BINARY_DIR}/openjph/src/core/common"  # 生成的配置文件
)

# 创建openjph头文件目录的符号链接，以支持 #include <openjph/ojph_file.h> 格式
set(OPENJPH_HEADER_LINK_DIR "${CMAKE_BINARY_DIR}/openjph_headers")
file(MAKE_DIRECTORY "${OPENJPH_HEADER_LINK_DIR}")

# 创建openjph子目录的符号链接
if(NOT EXISTS "${OPENJPH_HEADER_LINK_DIR}/openjph")
    if(WIN32)
        # Windows使用目录连接
        execute_process(
            COMMAND cmd /c mklink /J "${OPENJPH_HEADER_LINK_DIR}\\openjph" "${OPENJPH_SOURCE_DIR}\\src\\core\\common"
            RESULT_VARIABLE LINK_RESULT
        )
    else()
        # Unix系统使用符号链接
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E create_symlink 
                "${OPENJPH_SOURCE_DIR}/src/core/common"
                "${OPENJPH_HEADER_LINK_DIR}/openjph"
            RESULT_VARIABLE LINK_RESULT
        )
    endif()
    
    if(LINK_RESULT EQUAL 0)
        message(STATUS "Created OpenJPH header symlink: ${OPENJPH_HEADER_LINK_DIR}/openjph")
        list(APPEND OPENJPH_INCLUDE_DIRS "${OPENJPH_HEADER_LINK_DIR}")
    else()
        message(WARNING "Failed to create OpenJPH header symlink, falling back to direct include")
    endif()
endif()

# ================================================================================================
# Python 扩展构建配置
# ================================================================================================

if(DICUBE_BUILD_PYTHON_EXTENSIONS)
    # 使用scikit-build-core推荐的Python查找方式
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
    
    # 添加调试信息
    message(STATUS "Python_EXECUTABLE: ${Python_EXECUTABLE}")
    message(STATUS "Python_LIBRARIES: ${Python_LIBRARIES}")
    message(STATUS "Python_INCLUDE_DIRS: ${Python_INCLUDE_DIRS}")
    message(STATUS "Python_VERSION: ${Python_VERSION}")
    
    # 查找 pybind11
    find_package(pybind11 REQUIRED)
    
    # 查找 NumPy
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
        OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    if(NUMPY_INCLUDE_DIR)
        message(STATUS "NumPy include directory: ${NUMPY_INCLUDE_DIR}")
    else()
        message(FATAL_ERROR "NumPy not found")
    endif()
    
    # 编译第一个pybind11扩展 - 编码器
    pybind11_add_module(ojph_complete
        dicube/codecs/jph/encode_complete.cpp
    )
    
    target_include_directories(ojph_complete PRIVATE
        ${OPENJPH_INCLUDE_DIRS}
        ${NUMPY_INCLUDE_DIR}
    )
    
    target_link_libraries(ojph_complete PRIVATE
        ${OPENJPH_TARGET}
    )
    
    # 编译第二个pybind11扩展 - 解码器
    pybind11_add_module(ojph_decode_complete
        dicube/codecs/jph/decode_complete.cpp
    )
    
    target_include_directories(ojph_decode_complete PRIVATE
        ${OPENJPH_INCLUDE_DIRS}
        ${NUMPY_INCLUDE_DIR}
    )
    
    target_link_libraries(ojph_decode_complete PRIVATE
        ${OPENJPH_TARGET}
    )
    
    # 设置输出属性
    set_target_properties(ojph_complete PROPERTIES
        OUTPUT_NAME "ojph_complete"
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/dicube/codecs/jph
    )
    
    set_target_properties(ojph_decode_complete PROPERTIES
        OUTPUT_NAME "ojph_decode_complete"
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/dicube/codecs/jph
    )
    
    message(STATUS "Python extensions configured successfully")
else()
    message(STATUS "Python extensions are disabled")
endif()

# ================================================================================================
# 构建信息输出
# ================================================================================================

message(STATUS "=== DiCube Build Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "OpenJPH static build: ${OPENJPH_BUILD_STATIC}")
message(STATUS "Build Python extensions: ${DICUBE_BUILD_PYTHON_EXTENSIONS}")
if(DICUBE_BUILD_PYTHON_EXTENSIONS)
    message(STATUS "Python executable: ${Python_EXECUTABLE}")
    message(STATUS "Python version: ${Python_VERSION}")
    message(STATUS "NumPy include: ${NUMPY_INCLUDE_DIR}")
    message(STATUS "pybind11 version: ${pybind11_VERSION}")
endif()
message(STATUS "OpenJPH include dirs: ${OPENJPH_INCLUDE_DIRS}")
message(STATUS "OpenJPH target: ${OPENJPH_TARGET}")
message(STATUS "==================================") 