# ChromSeek

[中文文档](README_zh.md)

ChromSeek contains a pipeline for various omics prediction tasks, spanning from 1D (multi-omics signal prediction), 2D (TAD and Loop prediction), to 3D (Hi-C enhancement and ChIA-PET prediction) based on genomic sequence and structural features.

## Installation

Ensure you have Python 3.8+ installed. You can install the required dependencies using pip:

```bash
pip install torch numpy scipy scikit-learn matplotlib cooler jupyter
```
*(If you have a GPU, please visit [PyTorch](https://pytorch.org/) to install the appropriate GPU version of torch.)*

## How to Run

### 1. Download Pre-trained Models
Before running the tutorials, please download the corresponding model weight files (`.pth`) from the [GitHub Releases page](https://github.com/zzzlnb/chromSeek/releases) and place them in their respective subdirectories:
- `chromSeek_hic_enhancement.pth` ➡️ Place in `hic_enhancement/`
- `chromSeek_tad_prediction.pth` ➡️ Place in `tad_prediction/`
- `chromSeek_loop_prediction.pth` ➡️ Place in `loop_prediction/`
- `chromSeek_chiapet_prediction.pth` ➡️ Place in `chiapet_prediction/`
- `transfer_multiomics_best.pth` ➡️ Place in `multiomics_prediction/`

### 2. Run the Tutorials
The project is organized into modular directories. You can run the tutorial scripts using Python directly from the `chromSeek` root directory. For example:
```bash
python hic_enhancement/tutorial.py
```

## Tutorial Paths
- **Hi-C Enhancement**: `hic_enhancement/tutorial.py`
- **TAD Prediction**: `tad_prediction/tutorial.py`
- **Loop Prediction**: `loop_prediction/tutorial.py`
- **ChIA-PET Prediction**: `chiapet_prediction/tutorial.py`
- **Multi-omics Prediction**: `multiomics_prediction/tutorial.py`

### 3. Interactive Single-Cell Analysis Workflow (Jupyter App) 🌟
We provide a user-friendly, end-to-end Jupyter Notebook for **single-cell Micro-C / Hi-C resolution enhancement and TAD structure prediction**. In this interactive application, you can:
- Specify an `mcool` sparse single-cell matrix or a cropped `numpy` array.
- Freely define the target chromosome (e.g., `chr1`) and start region coordinates (using 10kb resolution Bin IDs).
- Automatically extract and align local genomic sequence features (includes local cache support for the `hg38` human genome).
- Call the deep learning framework to perform image reconstruction, and precisely identify high-confidence TAD domains using a dynamic percentile-adaptive mechanism (95th percentile) with visualized comparisons.

**Quick Start**: Access the app by opening `chromSeek_enhancement_app.ipynb` in the root directory using VS Code or a Jupyter server, and run the cells sequentially.
