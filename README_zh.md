# ChromSeek

[English](README.md)

ChromSeek 包含一系列组学预测任务的流水线，涵盖了从 1D（多组学信号预测）、2D（TAD、Loop 预测）到 3D（Hi-C 增强、ChIA-PET 预测）等不同的基因组序列和结构特征预测。

## 环境依赖安装

请确保您的 Python 版本 >= 3.8。通过以下命令安装所需的依赖库：

```bash
pip install torch numpy scipy scikit-learn matplotlib cooler jupyter
```
*(如果您有可用的 GPU，建议前往 [PyTorch 官网](https://pytorch.org/) 安装支持 CUDA 的 PyTorch 版本以提升运行速度。)*

## 运行方式

### 1. 下载预训练模型
在此仓库运行相应任务前，请前往项目的 [GitHub Releases 页面](https://github.com/zzzlnb/chromSeek/releases) 下载对应的模型权重文件（`.pth`），并将它们分别放入对应的文件夹下：
- `chromSeek_hic_enhancement.pth` ➡️ 放入 `hic_enhancement/` 目录中。
- `chromSeek_tad_prediction.pth` ➡️ 放入 `tad_prediction/` 目录中。
- `chromSeek_loop_prediction.pth` ➡️ 放入 `loop_prediction/` 目录中。
- `chromSeek_chiapet_prediction.pth` ➡️ 放入 `chiapet_prediction/` 目录中。
- `transfer_multiomics_best.pth` ➡️ 放入 `multiomics_prediction/` 目录中。

### 2. 执行测试样例
各个模块均独立组织。您可以在 `chromSeek` 根目录下，直接使用 Python 运行入门教程脚本进行测试：
```bash
python hic_enhancement/tutorial.py
```

## 各个任务教程的路径
- **Hi-C 增强**: `hic_enhancement/tutorial.py`
- **TAD 预测**: `tad_prediction/tutorial.py`
- **Loop 预测**: `loop_prediction/tutorial.py`
- **ChIA-PET 预测**: `chiapet_prediction/tutorial.py`
- **多组学预测**: `multiomics_prediction/tutorial.py`

### 3. 交互式单细胞分析工作流 (Jupyter App) 🌟
项目特别提供了一个用户友好的端到端 Jupyter Notebook 工具，适用于**单细胞级别 Micro-C / Hi-C 的分辨率增强及 TAD 结构域预测**。在这个交互应用中，您可以：
- 随时指定 `mcool` 稀疏单细胞矩阵或截取好的 `numpy` 阵列。
- 支持自由设定需要预测的染色体（例如 `chr1`）和起始区域坐标（以 `10kb` 表示的相对 Bin ID）。
- 自动提取本地存储的序列特征予以对齐计算（包含自动利用本地人类全基因组 `hg38` 缓存）。
- 调用深度框架自动完成重构，并在此基础上利用动态百分位自适应机制（95th percentile）定位高置信度 TAD，进行可视化高亮对比。

**快速上手入口**：使用 VS Code 或者官方 Jupyter 服务打开并运行根目录下的 `chromSeek_enhancement_app.ipynb`。
