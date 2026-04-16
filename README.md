# ChromSeek

ChromSeek 包含一系列组学预测任务的流水线，涵盖了从 1D（多组学信号预测）、2D（TAD、Loop 预测）到 3D（Hi-C 增强、ChIA-PET 预测）等不同的基因组序列和结构特征预测。

## 运行方式 (How to run)

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
