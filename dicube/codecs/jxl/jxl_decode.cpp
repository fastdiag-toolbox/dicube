// jxl_decode.cpp - JPEG XL 解码模块
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <jxl/decode.h>
#include <jxl/thread_parallel_runner.h>

#include <vector>
#include <memory>
#include <stdexcept>
#include <cstring>

namespace py = pybind11;

/// JPEG XL 解码函数
///
/// 此函数将 JPEG XL 字节流解码为 NumPy 数组。
/// 在解码过程中释放 GIL 以提高性能，使用 RAII 进行资源管理。
py::array imdecode_jxl(py::bytes encoded_data, bool keep_orientation = false) {
    // 获取字节流数据
    std::string encoded_str = encoded_data;
    const uint8_t* data = reinterpret_cast<const uint8_t*>(encoded_str.data());
    size_t data_size = encoded_str.size();

    if (data_size == 0) {
        throw std::runtime_error("Empty input data");
    }

    py::array result;
    
    // 在释放 GIL 的区域内执行解码
    {
        py::gil_scoped_release release;
        
        // 创建解码器（使用智能指针进行 RAII 管理）
        std::unique_ptr<JxlDecoder, decltype(&JxlDecoderDestroy)> decoder(
            JxlDecoderCreate(nullptr), JxlDecoderDestroy);
        if (!decoder) {
            throw std::runtime_error("Failed to create JxlDecoder");
        }

        // 设置多线程运行器
        size_t num_threads = JxlThreadParallelRunnerDefaultNumWorkerThreads();
        std::unique_ptr<void, decltype(&JxlThreadParallelRunnerDestroy)> runner(
            JxlThreadParallelRunnerCreate(nullptr, num_threads), 
            JxlThreadParallelRunnerDestroy);
        
        if (runner) {
            JxlDecoderStatus status = JxlDecoderSetParallelRunner(decoder.get(), 
                                                                 JxlThreadParallelRunner, 
                                                                 runner.get());
            if (status != JXL_DEC_SUCCESS) {
                throw std::runtime_error("Failed to set parallel runner");
            }
        }

        // 设置方向保持选项
        if (keep_orientation) {
            JxlDecoderSetKeepOrientation(decoder.get(), JXL_TRUE);
        }

        // 订阅事件
        JxlDecoderStatus status = JxlDecoderSubscribeEvents(decoder.get(), 
                                                           JXL_DEC_BASIC_INFO | JXL_DEC_FULL_IMAGE);
        if (status != JXL_DEC_SUCCESS) {
            throw std::runtime_error("Failed to subscribe to events");
        }

        // 设置输入数据
        status = JxlDecoderSetInput(decoder.get(), data, data_size);
        if (status != JXL_DEC_SUCCESS) {
            throw std::runtime_error("Failed to set input data");
        }

        // 获取基本信息
        JxlBasicInfo basic_info;
        status = JxlDecoderProcessInput(decoder.get());
        if (status != JXL_DEC_BASIC_INFO) {
            throw std::runtime_error("Failed to get basic info event");
        }
        
        status = JxlDecoderGetBasicInfo(decoder.get(), &basic_info);
        if (status != JXL_DEC_SUCCESS) {
            throw std::runtime_error("Failed to get basic info");
        }

        // 确定输出格式和通道数
        size_t num_channels = basic_info.num_color_channels + basic_info.num_extra_channels;
        
        // 根据原始位深选择合适的输出类型
        JxlDataType output_type;
        if (basic_info.bits_per_sample <= 8) {
            output_type = JXL_TYPE_UINT8;
        } else if (basic_info.bits_per_sample <= 16) {
            output_type = JXL_TYPE_UINT16;
        } else {
            output_type = JXL_TYPE_FLOAT;  // 高位深使用浮点
        }

        JxlPixelFormat pixel_format = {
            static_cast<uint32_t>(num_channels),
            output_type,
            JXL_NATIVE_ENDIAN,
            0
        };

        // 计算输出尺寸
        std::vector<ssize_t> shape;
        if (basic_info.have_animation) {
            // 动画模式（暂时只处理第一帧）
            shape = {1, 
                    static_cast<ssize_t>(basic_info.ysize), 
                    static_cast<ssize_t>(basic_info.xsize)};
            if (num_channels > 1) {
                shape.push_back(static_cast<ssize_t>(num_channels));
            }
        } else {
            // 静态图像
            shape = {static_cast<ssize_t>(basic_info.ysize), 
                    static_cast<ssize_t>(basic_info.xsize)};
            if (num_channels > 1) {
                shape.push_back(static_cast<ssize_t>(num_channels));
            }
        }

        // 获取输出缓冲区大小
        size_t buffer_size;
        status = JxlDecoderImageOutBufferSize(decoder.get(), &pixel_format, &buffer_size);
        if (status != JXL_DEC_SUCCESS) {
            throw std::runtime_error("Failed to get output buffer size");
        }

        // 创建输出缓冲区
        std::vector<uint8_t> output_buffer(buffer_size);

        // 设置输出缓冲区
        status = JxlDecoderSetImageOutBuffer(decoder.get(), &pixel_format, 
                                           output_buffer.data(), buffer_size);
        if (status != JXL_DEC_SUCCESS) {
            throw std::runtime_error("Failed to set output buffer");
        }

        // 执行解码
        status = JxlDecoderProcessInput(decoder.get());
        while (status != JXL_DEC_SUCCESS) {
            if (status == JXL_DEC_ERROR) {
                throw std::runtime_error("Decoding failed");
            } else if (status == JXL_DEC_NEED_MORE_INPUT) {
                throw std::runtime_error("Incomplete input data");
            } else if (status == JXL_DEC_FULL_IMAGE) {
                // 继续处理
                status = JxlDecoderProcessInput(decoder.get());
            } else {
                // 其他状态，继续处理
                status = JxlDecoderProcessInput(decoder.get());
            }
        }

        // 创建 NumPy 数组（需要重新获取 GIL）
        {
            py::gil_scoped_acquire acquire;
            
            if (output_type == JXL_TYPE_UINT8) {
                auto output_array = py::array_t<uint8_t>(shape);
                std::memcpy(output_array.mutable_data(), output_buffer.data(), buffer_size);
                result = std::move(output_array);
            } else if (output_type == JXL_TYPE_UINT16) {
                auto output_array = py::array_t<uint16_t>(shape);
                std::memcpy(output_array.mutable_data(), output_buffer.data(), buffer_size);
                result = std::move(output_array);
            } else {  // JXL_TYPE_FLOAT
                auto output_array = py::array_t<float>(shape);
                std::memcpy(output_array.mutable_data(), output_buffer.data(), buffer_size);
                result = std::move(output_array);
            }
        }
    }

    return result;
}

PYBIND11_MODULE(jxl_decode, m) {
    m.doc() = "JPEG XL 解码模块，采用现代 C++ 风格并优化性能";
    
    m.def("imdecode_jxl", &imdecode_jxl,
          py::arg("encoded_data"),
          py::arg("keep_orientation") = false,
          "将 JPEG XL 字节流解码为 NumPy 数组");
} 