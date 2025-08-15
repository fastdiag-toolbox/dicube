// encode_complete.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cstdint>

// 此文件不需要ssize_t定义


#include "ojph_file.h"
#include "ojph_codestream.h"
#include "ojph_mem.h"
#include "ojph_params.h"

namespace py = pybind11;
using namespace ojph;

// 内存输出文件类，用于存储编码后数据
class MemOutfile : public mem_outfile {
private:
    std::vector<ui8> buffer;
    size_t current_pos;
public:
    MemOutfile() : current_pos(0) {}

    void open() {
        buffer.clear();
        current_pos = 0;
    }

    size_t write(const void* data, size_t size) override {
        const ui8* ptr = static_cast<const ui8*>(data);
        if (current_pos + size > buffer.size()) {
            buffer.resize(current_pos + size);
        }
        std::copy(ptr, ptr + size, buffer.begin() + current_pos);
        current_pos += size;
        return size;
    }

    int seek(si64 offset, enum outfile_base::seek origin) override {
        switch (origin) {
            case outfile_base::OJPH_SEEK_SET:
                if (offset < 0 || static_cast<size_t>(offset) > buffer.size())
                    return -1;
                current_pos = static_cast<size_t>(offset);
                break;
            case outfile_base::OJPH_SEEK_CUR:
                if (static_cast<si64>(current_pos) + offset < 0 ||
                    current_pos + offset > buffer.size())
                    return -1;
                current_pos += offset;
                break;
            case outfile_base::OJPH_SEEK_END:
                if (static_cast<si64>(buffer.size()) + offset < 0 ||
                    buffer.size() + offset > buffer.size())
                    return -1;
                current_pos = buffer.size() + offset;
                break;
        }
        return 0;
    }

    si64 tell() override {
        return static_cast<si64>(current_pos);
    }

    // 返回 py::bytes 对象
    py::bytes read() {
        return py::bytes(reinterpret_cast<const char*>(buffer.data()), buffer.size());
    }
};

/// encode_image 函数：
///   - image: 要编码的 numpy 数组（2D 或 3D），支持 uint8、uint16、int16
///   - reversible: 是否使用可逆变换（默认 true）
///   - num_decompositions: 小波分解层数（默认 5）
///   - block_size: 码块尺寸 (width, height)（默认 (64, 64)）
///   - precinct_size: 若两个值均 > 0，则设置 precinct 尺寸（默认 (0, 0) 表示不使用）
///   - progression_order: 进程顺序（默认 "RPCL"）
///   - color_transform: 是否使用颜色变换（默认 false）
///   - profile: 配置文件名称（默认为空字符串）
///
/// 本函数在 C++ 内部逐行逐通道读取数据，且在释放 GIL 的区域内执行密集计算，
/// 以避免 Python 层大量跨语言调用产生的性能开销。
py::bytes encode_image(py::array image,
                       bool reversible = true,
                       int num_decompositions = 5,
                       std::pair<int, int> block_size = {64, 64},
                       std::pair<int, int> precinct_size = {0, 0},
                       std::string progression_order = "RPCL",
                       bool color_transform = false,
                       std::string profile = "")
{
    // 获取 numpy 数组信息（要求 2D 或 3D 且连续）
    auto buf = image.request();
    if (buf.ndim != 2 && buf.ndim != 3)
        throw std::runtime_error("Image must be 2D or 3D array");

    int height = buf.shape[0];
    int width  = buf.shape[1];
    int num_components = (buf.ndim == 3) ? buf.shape[2] : 1;

    // 检查数据类型，目前支持 uint8 ("B")、uint16 ("H")、int16 ("h")
    // 同时支持带字节序前缀的格式（如 "<H", ">H", "=H" 等）
    std::string format = buf.format;
    bool is_uint8 = (format == "B" || format == "<B" || format == ">B" || format == "=B");
    bool is_uint16 = (format == "H" || format == "<H" || format == ">H" || format == "=H");
    bool is_int16 = (format == "h" || format == "<h" || format == ">h" || format == "=h");
    
    if (!is_uint8 && !is_uint16 && !is_int16)
        throw std::runtime_error("Unsupported dtype. Only uint8, uint16, and int16 are supported.");

    // 创建输出内存文件
    MemOutfile outfile;
    outfile.open();

    // 创建 codestream 对象
    codestream cs;
    if (!profile.empty())
        cs.set_profile(profile.c_str());

    // 设置 SIZ 参数（API 返回的是引用）
    auto &&siz = cs.access_siz();
    siz.set_image_extent(point(width, height));
    siz.set_tile_size(size(width, height));  // 单一 tile
    siz.set_num_components(num_components);
    for (int c = 0; c < num_components; c++) {
        if (is_uint8)
            siz.set_component(c, point(1, 1), 8, false);
        else if (is_uint16)
            siz.set_component(c, point(1, 1), 16, false);
        else if (is_int16)
            siz.set_component(c, point(1, 1), 16, true);
    }

    // 设置 COD 参数
    auto &&cod = cs.access_cod();
    cod.set_num_decomposition(num_decompositions);
    cod.set_block_dims(block_size.first, block_size.second);
    if (precinct_size.first > 0 && precinct_size.second > 0) {
        ojph::size prec_size(precinct_size.first, precinct_size.second);
        cod.set_precinct_size(num_decompositions + 1, &prec_size);
    }
    cod.set_progression_order(progression_order.c_str());
    cod.set_reversible(reversible);
    cod.set_color_transform(color_transform);

    // 写入 codestream 头信息
    cs.write_headers(&outfile);

    // 在释放 GIL 的区域内逐行、逐通道拷贝数据
    {
        py::gil_scoped_release release;
        line_buf* line = nullptr;
        // 假设 numpy 数组为 C-contiguous
        char* data_ptr = static_cast<char*>(buf.ptr);
        // 对于 3D 数组，stride0, stride1, stride2 分别为：
        // buf.strides[0]（每行字节数）、buf.strides[1]（每个像素的字节数，即 num_components*itemsize）、buf.strides[2]（itemsize）
        for (int y = 0; y < height; y++) {
            for (int c = 0; c < num_components; c++) {
                // 获取当前行、当前通道数据起始地址
                const char* row_ptr;
                if (buf.ndim == 3)
                    row_ptr = data_ptr + y * buf.strides[0] + c * buf.strides[2];
                else
                    row_ptr = data_ptr + y * buf.strides[0];
                // 对于 3D 数组，相邻像素在同一通道的间隔为：
                int step = (buf.ndim == 3) ? (buf.strides[1] / buf.itemsize) : 1;

                // 获取下一个 line buffer
                ui32 comp = static_cast<ui32>(c);
                line = cs.exchange(line, comp);
                if (!line)
                    throw std::runtime_error("Failed to get line buffer");

                // 对每个像素，按照正确的步长提取当前通道的值
                for (int x = 0; x < width; x++) {
                    int32_t value = 0;
                    if (is_uint8) {
                        const uint8_t* src = reinterpret_cast<const uint8_t*>(row_ptr);
                        value = src[x * step];
                    } else if (is_uint16) {
                        const uint16_t* src = reinterpret_cast<const uint16_t*>(row_ptr);
                        value = src[x * step];
                    } else if (is_int16) {
                        const int16_t* src = reinterpret_cast<const int16_t*>(row_ptr);
                        value = src[x * step];
                    }
                    line->i32[x] = value;
                }
            }
        }
        // 调用一次 exchange 结束数据传输
        ui32 zero = 0;
        cs.exchange(line, zero);
    } // 结束无 GIL区域

    cs.flush();
    cs.close();

    // 重置文件指针并读取编码后的数据
    outfile.seek(0, outfile_base::OJPH_SEEK_SET);
    py::bytes encoded = outfile.read();
    outfile.close();

    return encoded;
}

PYBIND11_MODULE(ojph_complete, m) {
    m.doc() = "OpenJPH 编码模块，内部完成行和通道循环以减少 Python 层开销";
    m.def("encode_image", &encode_image,
          py::arg("image"),
          py::arg("reversible") = true,
          py::arg("num_decompositions") = 5,
          py::arg("block_size") = std::make_pair(64, 64),
          py::arg("precinct_size") = std::make_pair(0, 0),
          py::arg("progression_order") = "RPCL",
          py::arg("color_transform") = false,
          py::arg("profile") = "",
          "使用 OpenJPH 编码图像，将逐行和逐通道的循环全部在 C++ 内部完成。");
}
