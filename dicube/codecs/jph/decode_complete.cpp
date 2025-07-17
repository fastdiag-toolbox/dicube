// decode_complete.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <stdexcept>
#include <cstdint>

// 避免与Python的ssize_t定义冲突
// 我们创建自己的ssize_t类型别名，而不是重新定义ssize_t
#if defined(_MSC_VER) 
#include <BaseTsd.h>
// 使用pybind11的ssize_t，避免重复定义
using ojph_ssize_t = Py_ssize_t;
#else
// 在非Windows平台上使用标准ssize_t
using ojph_ssize_t = ssize_t;
#endif



#include "ojph_file.h"
#include "ojph_codestream.h"
#include "ojph_mem.h"
#include "ojph_params.h"

namespace py = pybind11;
using namespace ojph;

/*-----------------------------------------------------------------------------
  MemInfile 类：用于从内存中读取 JPEG2000 数据，同时保存对数据的引用，
  以防止在解码期间数据被释放。
-----------------------------------------------------------------------------*/
class MemInfile : public mem_infile {
public:
    py::bytes data_holder;  // 保持对数据的引用
    MemInfile() : mem_infile() {}

    void open(py::bytes data) {
        data_holder = data;  // 保存引用，确保数据在此对象生命周期内有效
        py::buffer_info info(py::buffer(data).request());
        const ui8* ptr = reinterpret_cast<const ui8*>(info.ptr);
        size_t size = static_cast<size_t>(info.size);
        mem_infile::open(ptr, size);
    }

    void close() {
        mem_infile::close();
    }
};

/*-----------------------------------------------------------------------------
  模板函数 decode_image_impl<T>
  该函数完成以下工作：
    1. 调用 cs.create() 开始解码流程。
    2. 对于每个组件（通道），依次调用 cs.pull(comp) 读取每一行，
       并将返回的行数据（line->i32 数组）转换为 T 类型，存入平面缓存；
    3. 对于单通道图像，平面数据即为最终行优先排列的数据；
       对于多通道图像，需要将平面排列的数据转换为交错排列（即每个像素的各分量依次存放）。
-----------------------------------------------------------------------------*/
template<typename T>
py::array decode_image_impl(codestream &cs, MemInfile &infile,
                            int level, int height, int width, int num_components) {
    cs.create();

    // 分配平面数据缓存：总大小 = num_components * height * width
    std::vector<T> planar_data(num_components * height * width);

    {
        // 在此区域释放 GIL，提高解码循环性能
        py::gil_scoped_release release;
        for (int comp = 0; comp < num_components; comp++) {
            for (int row = 0; row < height; row++) {
                ui32 comp_index = static_cast<ui32>(comp);
                line_buf* line = cs.pull(comp_index);
                if (!line)
                    throw std::runtime_error("Failed to get line buffer during decoding");
                // 假定 line->size == width
                for (int x = 0; x < width; x++) {
                    T value = static_cast<T>(line->i32[x]);
                    // 索引计算： comp * (height * width) + row * width + x
                    planar_data[comp * (height * width) + row * width + x] = value;
                }
            }
        }
    }
    cs.close();
    infile.close();

    if (num_components == 1) {
        // 构造 numpy 数组形状 (height, width)
        std::vector<ojph_ssize_t> shape = { static_cast<ojph_ssize_t>(height), static_cast<ojph_ssize_t>(width) };
        std::vector<ojph_ssize_t> strides = { static_cast<ojph_ssize_t>(sizeof(T)) * width, static_cast<ojph_ssize_t>(sizeof(T)) };
        py::array result(py::buffer_info(planar_data.data(), sizeof(T),
                                         py::format_descriptor<T>::format(),
                                         2, shape, strides));
        return result.attr("copy")();
    }
    else {
        // 对于多通道图像，将平面排列的数据转换为交错排列。
        // 平面排列顺序为：[ comp0(row0...rowN), comp1(row0...rowN), ... ]
        // 目标输出排列为 (height, width, num_components)
        std::vector<T> interleaved(num_components * height * width);
        for (int row = 0; row < height; row++) {
            for (int x = 0; x < width; x++) {
                for (int comp = 0; comp < num_components; comp++) {
                    interleaved[row * (width * num_components) + x * num_components + comp] =
                        planar_data[comp * (height * width) + row * width + x];
                }
            }
        }
        std::vector<ojph_ssize_t> shape = { static_cast<ojph_ssize_t>(height),
                                       static_cast<ojph_ssize_t>(width),
                                       static_cast<ojph_ssize_t>(num_components) };
        std::vector<ojph_ssize_t> strides = { static_cast<ojph_ssize_t>(sizeof(T)) * width * num_components,
                                         static_cast<ojph_ssize_t>(sizeof(T)) * num_components,
                                         static_cast<ojph_ssize_t>(sizeof(T)) };
        py::array result(py::buffer_info(interleaved.data(), sizeof(T),
                                         py::format_descriptor<T>::format(),
                                         3, shape, strides));
        return result.attr("copy")();
    }
}

/*-----------------------------------------------------------------------------
  decode_image 函数
  参数：
    - data: JPEG2000 字节数据（py::bytes）
    - level: 分辨率级别（0 表示全分辨率，默认为 0）
    - resilient: 是否启用鲁棒解码（默认为 false）
  
  流程：
    1. 使用 MemInfile 打开数据。
    2. 创建 codestream 并调用 read_headers()。
    3. 从 SIZ 标记中获得图像尺寸、通道数、bit depth、signed 等信息，
       并根据 level 调整分辨率。
    4. 根据 bit depth 和 signed 属性选择合适的输出数据类型，
       调用模板函数 decode_image_impl<T>() 执行解码。
    5. 返回生成的 numpy 数组。
-----------------------------------------------------------------------------*/
py::array decode_image(py::bytes data,
                       int level = 0,
                       bool resilient = false) {
    MemInfile infile;
    infile.open(data);

    codestream cs;
    if (resilient)
        cs.enable_resilience();

    cs.read_headers(&infile);

    // 使用 auto&& 绑定返回值，防止绑定临时对象错误
    auto &&siz = cs.access_siz();
    if (level > 0)
        cs.restrict_input_resolution(level, level);
    int height = siz.get_recon_height(0);
    int width = siz.get_recon_width(0);
    int num_components = siz.get_num_components();

    int bit_depth = siz.get_bit_depth(0);
    bool is_signed = siz.is_signed(0);
    if (bit_depth <= 8 && !is_signed) {
        return decode_image_impl<uint8_t>(cs, infile, level, height, width, num_components);
    } else if (bit_depth <= 8 && is_signed) {
        return decode_image_impl<int8_t>(cs, infile, level, height, width, num_components);
    } else if (bit_depth <= 16 && !is_signed) {
        return decode_image_impl<uint16_t>(cs, infile, level, height, width, num_components);
    } else if (bit_depth <= 16 && is_signed) {
        return decode_image_impl<int16_t>(cs, infile, level, height, width, num_components);
    } else if (bit_depth <= 32 && !is_signed) {
        return decode_image_impl<uint32_t>(cs, infile, level, height, width, num_components);
    } else if (bit_depth <= 32 && is_signed) {
        return decode_image_impl<int32_t>(cs, infile, level, height, width, num_components);
    } else {
        throw std::runtime_error("Unsupported bit depth");
    }
}

/*-----------------------------------------------------------------------------
  PYBIND11_MODULE 宏
  将该模块导出为 Python 模块 "ojph_decode_complete"（可根据需要修改模块名）。
-----------------------------------------------------------------------------*/
PYBIND11_MODULE(ojph_decode_complete, m) {
    m.doc() = "OpenJPH 解码模块：将解码循环全部在 C++ 侧执行，返回 numpy 数组";
    m.def("decode_image", &decode_image,
          py::arg("data"),
          py::arg("level") = 0,
          py::arg("resilient") = false,
          "Decode JPEG2000 bytes to a numpy array using OpenJPH.");
}
