# LiYing 依赖项安装指南

## 概述

LiYing 项目依赖于以下主要组件：
- **AGPicCompress**：图像压缩功能
- **mozjpeg**：JPEG 图像优化
- **pngquant**：PNG 图像压缩
- **mozjpeg-lossless-optimization**：无损 JPEG 优化

## 依赖项安装步骤

### 1. Python 依赖项安装

首先安装 Python 包依赖：

```bash
pip install -r requirements.txt
```

这将安装以下包：
- `click`
- `cffi`
- `colorama`
- `mozjpeg_lossless_optimization`
- `onnxruntime>=1.14.0`
- `orjson>=3.10.7`
- `gradio>=4.44.1`
- `Pillow`
- `opencv-python`
- `numpy`
- `piexif`

### 2. pngquant 安装

**pngquant 是必需的外部依赖项**，需要手动安装。

#### Windows 系统

1. **方法一：下载预编译版本**
   - 访问 [pngquant 官方网站](https://pngquant.org/)
   - 下载 Windows 版本的 `pngquant.exe`
   - 将 `pngquant.exe` 放置在以下任一位置：
     - 系统环境变量 PATH 中的任一目录（推荐）
     - `LiYing/src/tool/` 目录下
     - `LiYing/src/tool/ext/` 目录下

2. **方法二：使用包管理器**
   ```bash
   # 使用 Chocolatey
   choco install pngquant
   
   # 使用 Scoop
   scoop install pngquant
   ```

#### macOS 系统

```bash
# 使用 Homebrew
brew install pngquant

# 使用 MacPorts
sudo port install pngquant
```

#### Linux 系统

```bash
# Ubuntu/Debian
sudo apt-get install pngquant

# CentOS/RHEL/Fedora
sudo yum install pngquant
# 或者对于较新版本
sudo dnf install pngquant

# Arch Linux
sudo pacman -S pngquant
```

### 3. pngquant 配置位置优先级

程序会按以下优先级查找 pngquant：

1. **环境变量 PATH**（推荐）
   - 系统会自动在 PATH 中查找 `pngquant` 命令
   
2. **LiYing/src/tool/ 目录**
   - 将 `pngquant` 可执行文件直接放在此目录
   
3. **LiYing/src/tool/ext/ 目录**
   - 将 `pngquant` 可执行文件放在 ext 子目录中

### 4. 验证安装

创建测试脚本验证依赖项是否正确安装：

```python
# 测试脚本
import subprocess
import sys
from pathlib import Path

def test_pngquant():
    """测试 pngquant 是否可用"""
    try:
        # 尝试从环境变量查找
        result = subprocess.run(['pngquant', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ pngquant 在环境变量中找到")
            return True
    except FileNotFoundError:
        pass
    
    # 尝试从项目目录查找
    search_paths = [
        Path(__file__).parent / 'src' / 'tool',
        Path(__file__).parent / 'src' / 'tool' / 'ext'
    ]
    
    for path in search_paths:
        pngquant_path = path / ('pngquant.exe' if sys.platform == 'win32' else 'pngquant')
        if pngquant_path.exists():
            print(f"✓ pngquant 在 {path} 中找到")
            return True
    
    print("✗ pngquant 未找到")
    return False

def test_mozjpeg():
    """测试 mozjpeg-lossless-optimization 是否可用"""
    try:
        import mozjpeg_lossless_optimization
        print("✓ mozjpeg-lossless-optimization 已安装")
        return True
    except ImportError:
        print("✗ mozjpeg-lossless-optimization 未安装")
        return False

if __name__ == "__main__":
    print("LiYing 依赖项检查")
    print("=" * 30)
    
    pngquant_ok = test_pngquant()
    mozjpeg_ok = test_mozjpeg()
    
    if pngquant_ok and mozjpeg_ok:
        print("\n✓ 所有依赖项已正确安装")
    else:
        print("\n✗ 部分依赖项缺失，请按照指南安装")
```

### 5. 系统要求

- **Windows**：
  - 最低版本：Windows 7 SP1 及以上
  - 需要安装 [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)
  - 对于 Windows 7：需要 `onnxruntime==1.14.0, orjson==3.10.7, gradio==4.44.1`

- **macOS**：
  - 支持主流版本

- **Linux**：
  - 支持主流发行版

### 6. 常见问题解决

#### 问题：pngquant 未找到
**解决方案**：
1. 确认 pngquant 已正确安装
2. 检查环境变量 PATH 设置
3. 将 pngquant 可执行文件复制到项目指定目录

#### 问题：mozjpeg_lossless_optimization 安装失败
**解决方案**：
1. 确保 Python 版本兼容
2. 更新 pip：`pip install --upgrade pip`
3. 尝试使用预编译版本：`pip install --only-binary=all mozjpeg_lossless_optimization`

#### 问题：Windows 7 兼容性
**解决方案**：
使用指定版本：
```bash
pip install onnxruntime==1.14.0 orjson==3.10.7 gradio==4.44.1
```

### 7. 完整安装示例

```bash
# 1. 克隆项目
git clone https://github.com/aoguai/LiYing
cd LiYing

# 2. 安装 Python 依赖
pip install -r requirements.txt

# 3. 安装 pngquant（以 Ubuntu 为例）
sudo apt-get install pngquant

# 4. 验证安装
python -c "
import subprocess
try:
    subprocess.run(['pngquant', '--version'], check=True)
    print('pngquant 安装成功')
except:
    print('pngquant 安装失败')

try:
    import mozjpeg_lossless_optimization
    print('mozjpeg_lossless_optimization 安装成功')
except:
    print('mozjpeg_lossless_optimization 安装失败')
"

# 5. 下载模型文件到 src/model 目录
# （根据项目文档下载所需模型）
```

## 总结

完成以上步骤后，LiYing 项目的所有依赖项应该已正确安装。如果遇到问题，请检查：

1. Python 环境是否正确
2. pngquant 是否在正确位置
3. 系统要求是否满足
4. 网络连接是否正常（用于下载依赖包）

更多信息请参考：
- [pngquant 官方文档](https://pngquant.org/)
- [AGPicCompress 项目](https://github.com/aoguai/AGPicCompress)
- [mozjpeg 项目](https://github.com/mozilla/mozjpeg)