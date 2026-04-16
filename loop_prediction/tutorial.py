import os
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import LoopPredictionModel
import scipy.ndimage as ndimage
def extract_peaks_pred(prob_map, threshold=0.5):
    """Thresholds and clusters predicted probabilities into distinct 2D loop peak coordinates."""
    mask = (prob_map >= threshold).astype(int)
    labeled_array, num_features = ndimage.label(mask)
    peaks = []
    for region_idx in range(1, num_features + 1):
        coords = np.argwhere(labeled_array == region_idx)
        upper_coords = [tuple(c) for c in coords if c[1] > c[0]]
        if not upper_coords:
            continue
        best_coord = max(upper_coords, key=lambda c: prob_map[c[0], c[1]])
        peaks.append(best_coord)
    return peaks
def extract_peaks_gt(binary_map):
    """Extracts centroid coordinates from binary Ground Truth loop annotations."""
    labeled_array, num_features = ndimage.label(binary_map)
    peaks = []
    for region_idx in range(1, num_features + 1):
        coords = np.argwhere(labeled_array == region_idx)
        upper_coords = [tuple(c) for c in coords if c[1] > c[0]]
        if not upper_coords:
            continue
        center_c = upper_coords[len(upper_coords) // 2]
        peaks.append(center_c)
    return peaks
def plot_loops_on_hic(hic_matrix, loop_peaks, ax, title, marker_color):
    """Render the Hi-C map and overlay 2D scatter points for loops."""
    disp_hic = np.log1p(hic_matrix)
    val_80 = np.percentile(disp_hic[disp_hic > 0], 80) if np.sum(disp_hic > 0) > 0 else 1.0
    ax.imshow(disp_hic, cmap='Reds', vmin=0, vmax=val_80)
    if loop_peaks:
        pr_x, pr_y = zip(*[(c[1], c[0]) for c in loop_peaks])
        ax.scatter(pr_x, pr_y, facecolors='none', edgecolors=marker_color, marker='o', s=100, linewidths=2.5)
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.set_xlabel("Hi-C bins (10kb)", fontweight='bold')
    ax.set_ylabel("Hi-C bins (10kb)", fontweight='bold')
def main():
    print("==============================================")
    print(" chromSeek Tutorial: Loop Prediction")
    print("==============================================\n")
    dataset_path = 'sample_data.pt'
    ckpt_path = 'chromSeek_loop_prediction.pth'
    print(f"[Step 1] Loading sample data from {dataset_path}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_list = torch.load(dataset_path, weights_only=False)
    sample = data_list[0]
    t_seq = sample['inputs']['seq'].to(device)
    if t_seq.dim() == 2:
        t_seq = t_seq.unsqueeze(0)
    t_hic = sample['inputs']['hic'].to(device)
    if t_hic.dim() == 3:
        t_hic = t_hic.unsqueeze(0)
    gt_loop = sample['targets']['loop'].squeeze().cpu().numpy()
    raw_hic = sample['info']['raw_hic']
    print(f"  --> DNA Sequence Shape      : {t_seq.shape}")
    print(f"  --> Sparse Hi-C Shape       : {t_hic.shape}")
    print(f"  --> Ground Truth Loop Mask  : {gt_loop.shape} (Sum loops: {int(np.sum(gt_loop))})\n")
    print(f"[Step 2] Initializing model and loading weights from {ckpt_path}...")
    model = LoopPredictionModel(pretrained_path=None).to(device)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    print("[Step 3] Running Forward Pass (Inference)...")
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            out = model(t_seq, t_hic)
        pred_prob = torch.softmax(out, dim=1)[:, 1, :, :].squeeze().cpu().numpy()
    print("[Step 4] Computing Metrics...")
    triu_indices = np.triu_indices_from(gt_loop, k=2)
    gt_flat = gt_loop[triu_indices].astype(int)
    pred_flat = pred_prob[triu_indices]
    if np.sum(gt_flat) > 0:
        auc = roc_auc_score(gt_flat, pred_flat)
        auprc = average_precision_score(gt_flat, pred_flat)
    else:
        auc, auprc = 0.0, 0.0
    print(f"  --> Loop AUROC   : {auc:.4f}")
    print(f"  --> Loop AUPRC   : {auprc:.4f}\n")
    print("[Step 5] Generating Visualizations...")
    true_peaks = extract_peaks_gt(gt_loop)
    pred_peaks = extract_peaks_pred(pred_prob, threshold=0.5)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    plot_loops_on_hic(raw_hic, true_peaks, axes[0], 
                      title=f"Ground Truth Annotated Loops", 
                      marker_color='blue')
    plot_loops_on_hic(raw_hic, pred_peaks, axes[1], 
                      title=f"chromSeek Predicted Anchors", 
                      marker_color='red')
    plt.tight_layout()
    out_path = "chromSeek_tutorial_loop_prediction.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Successfully saved visualization to '{out_path}'.")
if __name__ == "__main__":
    main()
