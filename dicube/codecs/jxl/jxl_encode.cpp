// jxl_encode.cpp - JPEG XL 编码模块
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <jxl/encode.h>
#include <jxl/thread_parallel_runner.h>

#include <vector>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <cstring>
#include <cctype>

namespace py = pybind11;

// 前向声明
py::bytes imencode_jxl(py::array image,
                       int quality,
                       int effort,
                       std::string colorspace,
                       int bit_depth);

/// 返回 libjxl 版本字符串
std::string jpegxl_version() {
    uint32_t ver = JxlEncoderVersion();
    std::ostringstream oss;
    oss << "libjxl " << (ver / 1000000) << "."
        << ((ver / 1000) % 1000) << "."
        << (ver % 1000);
    return oss.str();
}

/// 快速无损编码函数：使用 JxlFastLosslessEncode API
/// 
/// 此函数针对无损编码进行了优化，适用于需要快速编码的场景。
/// 在计算密集的编码过程中释放 GIL 以提高性能。
/// 注意：JxlFastLosslessEncode 在当前版本的 libjxl 中可能不可用，
/// 因此暂时使用标准编码流程作为替代。
py::bytes imencode_jxl_fast(py::array image,
                            int quality = 100,
                            int effort = 6,
                            int bit_depth = -1) {
    // 暂时使用标准编码流程，因为 JxlFastLosslessEncode 在当前 libjxl 版本中不可用
    return imencode_jxl(image, quality, effort, "", bit_depth);
    
    /* 原始的 JxlFastLosslessEncode 实现（注释掉，等待 libjxl 版本更新）
    // 获取 numpy 数组信息
    auto buf = image.request();
    if (buf.ndim < 2 || buf.ndim > 4) {
        throw std::runtime_error("Invalid image dimensions: expected 2D, 3D or 4D array");
    }

    // 解析图像尺寸和通道数
    size_t height = buf.shape[0];
    size_t width = buf.shape[1];
    size_t channels = (buf.ndim == 3) ? buf.shape[2] : 1;
    
    // 若为 4D，取第一帧
    if (buf.ndim == 4) {
        height = buf.shape[1];
        width = buf.shape[2];
        channels = buf.shape[3];
    }

    // 在释放 GIL 的区域内执行编码
    py::bytes result;
    {
        py::gil_scoped_release release;
        
        unsigned char* output_data = nullptr;
        size_t output_size = JxlFastLosslessEncode(
            static_cast<unsigned char*>(buf.ptr), 
            width, 
            width * channels, 
            height, 
            channels,
            bit_depth > 0 ? bit_depth : -1,
            false,  // little_endian
            effort,
            &output_data,
            nullptr,  // runner
            nullptr   // runner_opaque
        );

        if (output_size == 0 || !output_data) {
            throw std::runtime_error("Failed to encode with fast lossless encoding");
        }

        // 创建 py::bytes 对象（需要重新获取 GIL）
        {
            py::gil_scoped_acquire acquire;
            result = py::bytes(reinterpret_cast<const char*>(output_data), output_size);
        }

        // 释放内存
        free(output_data);
    }

    return result;
    */
}

/// 完整编码函数：支持有损和无损编码，以及各种参数配置
///
/// 此函数提供了对 JPEG XL 编码的完全控制，支持颜色空间、质量、位深等设置。
/// 在编码过程中释放 GIL 以提高性能。
py::bytes imencode_jxl(py::array image,
                       int quality = 100,
                       int effort = 6,
                       std::string colorspace = "",
                       int bit_depth = -1) {
    // 获取 numpy 数组信息
    auto buf = image.request();
    if (buf.ndim < 2 || buf.ndim > 4) {
        throw std::runtime_error("Invalid image dimensions: expected 2D, 3D or 4D array");
    }

    // 解析图像尺寸和通道数
    size_t height = buf.shape[0];
    size_t width = buf.shape[1];
    size_t channels = (buf.ndim == 3) ? buf.shape[2] : 1;
    
    // 若为 4D，取第一帧
    if (buf.ndim == 4) {
        height = buf.shape[1];
        width = buf.shape[2];
        channels = buf.shape[3];
    }

    void* data = buf.ptr;
    size_t data_size = buf.size * buf.itemsize;

    // 在释放 GIL 的区域内执行编码
    py::bytes result;
    {
        py::gil_scoped_release release;
        
        // 创建编码器
        std::unique_ptr<JxlEncoder, decltype(&JxlEncoderDestroy)> encoder(
            JxlEncoderCreate(nullptr), JxlEncoderDestroy);
        if (!encoder) {
            throw std::runtime_error("Failed to create JxlEncoder");
        }

        // 设置基本图像信息
        JxlBasicInfo basic_info;
        JxlEncoderInitBasicInfo(&basic_info);
        basic_info.xsize = static_cast<uint32_t>(width);
        basic_info.ysize = static_cast<uint32_t>(height);
        
        if (channels == 1) {
            basic_info.num_color_channels = 1;
            basic_info.num_extra_channels = 0;
        } else {
            basic_info.num_color_channels = 3;  // 默认 RGB
            basic_info.num_extra_channels = (channels > 3) ? (channels - 3) : 0;
        }

        // 无损模式设置
        if (quality >= 100) {
            basic_info.uses_original_profile = JXL_TRUE;
        }

        JxlEncoderStatus status = JxlEncoderSetBasicInfo(encoder.get(), &basic_info);
        if (status != JXL_ENC_SUCCESS) {
            throw std::runtime_error("Failed to set basic info");
        }

        // 设置颜色空间
        JxlColorEncoding color_encoding;
        if (buf.format == py::format_descriptor<uint8_t>::format()) {
            JxlColorEncodingSetToSRGB(&color_encoding, (channels == 1));
        } else {
            JxlColorEncodingSetToLinearSRGB(&color_encoding, (channels == 1));
        }
        
        status = JxlEncoderSetColorEncoding(encoder.get(), &color_encoding);
        if (status != JXL_ENC_SUCCESS) {
            throw std::runtime_error("Failed to set color encoding");
        }

        // 创建帧设置
        JxlEncoderFrameSettings* frame_settings = JxlEncoderFrameSettingsCreate(encoder.get(), nullptr);
        if (!frame_settings) {
            throw std::runtime_error("Failed to create frame settings");
        }

        // 设置编码参数
        status = JxlEncoderFrameSettingsSetOption(frame_settings,
                                                 JXL_ENC_FRAME_SETTING_EFFORT, effort);
        if (status != JXL_ENC_SUCCESS) {
            throw std::runtime_error("Failed to set encoding effort");
        }

        // 设置质量模式
        if (quality >= 100) {
            status = JxlEncoderSetFrameLossless(frame_settings, JXL_TRUE);
            if (status != JXL_ENC_SUCCESS) {
                throw std::runtime_error("Failed to set lossless mode");
            }
        } else {
            status = JxlEncoderSetFrameLossless(frame_settings, JXL_FALSE);
            if (status != JXL_ENC_SUCCESS) {
                throw std::runtime_error("Failed to set lossy mode");
            }
            
            float distance = JxlEncoderDistanceFromQuality(quality);
            status = JxlEncoderSetFrameDistance(frame_settings, distance);
            if (status != JXL_ENC_SUCCESS) {
                throw std::runtime_error("Failed to set quality distance");
            }
        }

        // 设置位深
        if (bit_depth > 0) {
            basic_info.bits_per_sample = static_cast<uint32_t>(bit_depth);
            status = JxlEncoderSetBasicInfo(encoder.get(), &basic_info);
            if (status != JXL_ENC_SUCCESS) {
                throw std::runtime_error("Failed to set bit depth");
            }
        }

        // 处理颜色空间参数
        if (!colorspace.empty()) {
            std::string cs;
            cs.reserve(colorspace.size());
            for (char c : colorspace) {
                cs.push_back(std::toupper(c));
            }
            
            if (cs == "L" || cs == "GRAY") {
                basic_info.num_color_channels = 1;
                basic_info.num_extra_channels = (channels > 1) ? (channels - 1) : 0;
                status = JxlEncoderSetBasicInfo(encoder.get(), &basic_info);
                if (status != JXL_ENC_SUCCESS) {
                    throw std::runtime_error("Failed to update basic info with colorspace");
                }
            }
        }

        // 设置像素格式
        JxlPixelFormat pixel_format;
        std::memset(&pixel_format, 0, sizeof(pixel_format));
        pixel_format.num_channels = static_cast<uint32_t>(channels);
        pixel_format.endianness = JXL_NATIVE_ENDIAN;
        
        if (buf.format == py::format_descriptor<uint8_t>::format()) {
            pixel_format.data_type = JXL_TYPE_UINT8;
        } else if (buf.format == py::format_descriptor<uint16_t>::format()) {
            pixel_format.data_type = JXL_TYPE_UINT16;
        } else if (buf.format == py::format_descriptor<float>::format()) {
            pixel_format.data_type = JXL_TYPE_FLOAT;
        } else {
            pixel_format.data_type = JXL_TYPE_UINT8;
        }

        // 添加图像帧
        status = JxlEncoderAddImageFrame(frame_settings, &pixel_format, data, data_size);
        if (status != JXL_ENC_SUCCESS) {
            throw std::runtime_error("Failed to add image frame");
        }

        // 关闭输入
        JxlEncoderCloseInput(encoder.get());

        // 编码处理
        size_t initial_buffer_size = std::max(static_cast<size_t>(32768), 
                                            data_size / (quality >= 100 ? 4 : 16));
        std::vector<uint8_t> out_buffer(initial_buffer_size);
        uint8_t* next_out = out_buffer.data();
        size_t avail_out = out_buffer.size();

        while (true) {
            status = JxlEncoderProcessOutput(encoder.get(), &next_out, &avail_out);
            
            if (status == JXL_ENC_NEED_MORE_OUTPUT) {
                size_t offset = next_out - out_buffer.data();
                out_buffer.resize(out_buffer.size() * 2);
                next_out = out_buffer.data() + offset;
                avail_out = out_buffer.size() - offset;
                continue;
            } else if (status == JXL_ENC_SUCCESS) {
                break;
            } else {
                throw std::runtime_error("Encoding failed");
            }
        }

        size_t output_size = out_buffer.size() - avail_out;
        
        // 创建结果（需要重新获取 GIL）
        {
            py::gil_scoped_acquire acquire;
            result = py::bytes(reinterpret_cast<const char*>(out_buffer.data()), output_size);
        }
    }

    return result;
}

PYBIND11_MODULE(jxl_encode, m) {
    m.doc() = "JPEG XL 编码模块，采用现代 C++ 风格并优化性能";
    
    m.def("jpegxl_version", &jpegxl_version, "获取 libjxl 库版本信息");
    
    m.def("imencode_jxl", &imencode_jxl,
          py::arg("image"),
          py::arg("quality") = 100,
          py::arg("effort") = 6,
          py::arg("colorspace") = "",
          py::arg("bit_depth") = -1,
          "将 NumPy 图像数组编码为 JPEG XL 格式，支持有损和无损编码");
    
    m.def("imencode_jxl_fast", &imencode_jxl_fast,
          py::arg("image"),
          py::arg("quality") = 100,
          py::arg("effort") = 6,
          py::arg("bit_depth") = -1,
          "快速无损编码 NumPy 图像数组为 JPEG XL 格式");
} 