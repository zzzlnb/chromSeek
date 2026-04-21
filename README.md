# ChromSeek

ChromSeek 包含一系列组学预测任务的流水线，涵盖了从 1D（多组学信号预测）、2D（TAD、Loop 预测）到 3D（Hi-C 增强、ChIA-PET 预测）等不同的基因组序列和结构特征预测。

## 运行方式 (How to run)

### 1. 下载预训练模型 (Download Pre-trained Models)
由于预训练模型体积较大，请在运行对应任务的教程前，先前往项目的 [GitHub Releases 页面](https://github.com/zzzlnb/chromSeek/releases) 下载对应的模型权重文件（`.pth`），并将它们分别放置到其对应的子文件夹下：
- `chromSeek_hic_enhancement.pth` ➡️ 放入 `hic_enhancement/` 目录中。
- `chromSeek_tad_prediction.pth` ➡️ 放入 `tad_prediction/` 目录中。
- `chromSeek_loop_prediction.pth` ➡️ 放入 `loop_prediction/` 目录中。
- `chromSeek_chiapet_prediction.pth` ➡️ 放入 `chiapet_prediction/` 目录中。
- `transfer_multiomics_best.pth` ➡️ 放入 `multiomics_prediction/` 目录中。

### 2. 执行测试样例 (Run the Tutorials)
项目被组织成了各自独立的模块，每个子预测模块下都有一个对应的 `tutorial.py` 或特定的预处理/预测脚本作为入口点。
您可以在 `chromSeek` 根目录下，直接使用 Python 运行相应的教程脚本，例如：
```bash
python hic_enhancement/tutorial.py
```

## 各个任务教程的路径 (Tutorial Paths)

以下是每个子任务对应的官方运行教程路径：

- **Hi-C 增强 (Hi-C Enhancement)**: `hic_enhancement/tutorial.py`
- **TAD 预测 (TAD Prediction)**: `tad_prediction/tutorial.py`
- **Loop 预测 (Loop Prediction)**: `loop_prediction/tutorial.py`
- **ChIA-PET 预测 (ChIA-PET Prediction)**: `chiapet_prediction/tutorial.py`
- **多组学预测 (Multi-omics Prediction)**: `multiomics_prediction/tutorial.py`

### 3. 交互式单细胞分析工作流 (Interactive Jupyter App) 🌟
项目特别提供了一个用户友好的端到端 Jupyter Notebook 工具，适用于**单细胞级别 Micro-C / Hi-C 的分辨率极大化增强及 TAD 结构域预测**。在这个交互应用中，您可以：
- 随时指定 `mcool` 稀疏单细胞矩阵或截取好的 `numpy` 片段阵列。
- 支持自由设定需要预测的染色体（例如 `chr1`）和起始区域坐标（以 `10kb` 表示的相对 Bin ID）。
- 自动提取本地存储的序列特征予以对齐计算（包含支持人类全基因组 `hg38` 高速缓存）。
- 一键调用深度框架自动完成图像级重构，并在此基础上利用动态百分位自适应机制（95th percentile）精准定位高置信度 TAD，进行可视化高亮对比！

**快速上手入口**：使用 VS Code 或者官方 Jupyter 服务打开根目录下的 `chromSeek_enhancement_app.ipynb` 依次运行计算块即可。
