# OpenJPH 静态编译集成实施计划

## 项目目标

实现dicube项目中OpenJPH的静态编译集成，支持两种使用模式：
1. **普通用户模式**：通过pip安装预编译的binary包，支持Windows/Linux/macOS
2. **开发者模式**：下载源码自己编译，可修改OpenJPH源码

## 当前状态分析

### 现有结构
```
dicube/
├── source/OpenJPH/                    # git submodule（已有）
├── dicube/codecs/jph/                 # Python包装代码（已有）
│   ├── codec.py                       # 主要接口
│   ├── encode_complete.cpp            # C++编码实现
│   ├── decode_complete.cpp            # C++解码实现
│   └── *.so                          # 已编译的扩展
├── setup.py                          # 当前构建配置
└── pyproject.toml                     # 现代配置文件
```

### 现有问题
- setup.py依赖系统已安装的libopenjph
- 无法为普通用户提供开箱即用的安装体验
- 缺少跨平台构建配置

## 解决方案设计

### 1. 构建系统现代化

**选择 pyproject.toml 作为主配置**
- 符合Python现代标准
- 更好的工具链支持
- 配置与代码分离
- 保留setup.py处理复杂C++构建逻辑

### 2. 静态编译架构

#### 核心策略
- **静态链接OpenJPH**：将OpenJPH源码编译为静态库并链接到Python扩展
- **零外部依赖**：用户无需安装额外C++库
- **跨平台支持**：Linux/Windows/macOS自动构建

#### 构建流程
1. **检测模式**：
   - 开发模式：检测source/OpenJPH是否存在
   - 分发模式：从预编译wheel安装

2. **OpenJPH编译**：
   - 使用CMake构建静态库
   - 平台特定编译选项
   - 优化设置

3. **Python扩展编译**：
   - pybind11绑定
   - 静态链接OpenJPH
   - 包装到wheel

### 3. 双模式支持

#### A. 普通用户模式
```bash
pip install dicube
```
- 下载预编译wheel
- 开箱即用，无编译时间
- 支持Windows/Linux/macOS

#### B. 开发者模式  
```bash
git clone --recursive your-repo
cd dicube
pip install -e .
```
- 本地编译安装
- 可修改OpenJPH源码
- 完整开发环境

### 4. 技术实现细节

#### 构建系统架构
```
pyproject.toml          # 主配置，依赖管理，元数据
├── setup.py           # C++构建逻辑（如果需要复杂逻辑）
├── CMakeLists.txt     # OpenJPH构建配置
└── build_scripts/     # 辅助构建脚本
```

#### 静态编译配置
- **Linux**: gcc/clang + static linking
- **Windows**: MSVC + static CRT
- **macOS**: clang + static linking

#### CI/CD Pipeline
- GitHub Actions多平台构建
- cibuildwheel自动化wheel生成
- 自动发布到PyPI

## 实施计划

### 阶段1: 基础架构重构
1. 重构pyproject.toml配置
2. 创建CMake构建脚本
3. 修改C++扩展构建逻辑

### 阶段2: 静态编译实现
1. 实现OpenJPH静态编译
2. 修改扩展链接方式
3. 本地测试验证

### 阶段3: 跨平台支持
1. 配置CI/CD构建
2. 多平台wheel生成
3. 自动化测试

### 阶段4: 验证和优化
1. 各平台安装测试
2. 性能优化
3. 文档更新

## 预期效果

### 对普通用户
- ✅ 一行命令安装：`pip install dicube`
- ✅ 无需额外依赖
- ✅ 跨平台兼容
- ✅ 快速安装

### 对开发者
- ✅ 源码可修改
- ✅ 本地调试方便
- ✅ 完整构建控制
- ✅ 开发环境一致

## 风险评估

### 技术风险
- **跨平台编译复杂性**：不同平台的编译器差异
- **静态链接问题**：可能的符号冲突
- **构建时间**：CI/CD构建时间可能较长

### 缓解措施
- 分阶段实施，逐步验证
- 充分测试各平台兼容性
- 使用成熟的构建工具链

## 成功指标

1. **普通用户**：`pip install dicube` 在3大平台均可成功安装并使用
2. **开发者**：源码安装编译成功，可正常开发调试
3. **性能**：静态编译版本性能不低于动态链接版本
4. **维护性**：构建系统清晰，易于维护和扩展 