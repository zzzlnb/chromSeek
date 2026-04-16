import os
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from model import DnaHicTadPredictor
def cluster_and_nms(pred_prob, threshold=0.5, max_gap=20):
    """Applies Non-Maximum Suppression (NMS) to convert peak probabilities into definite TAD boundary indices."""
    pred_classes = np.zeros_like(pred_prob, dtype=int)
    valid_idx = np.where(pred_prob > threshold)[0]
    if len(valid_idx) == 0:
        return pred_classes
    clusters = []
    current_cluster = [valid_idx[0]]
    for idx in valid_idx[1:]:
        if idx - current_cluster[-1] <= max_gap:
            current_cluster.append(idx)
        else:
            clusters.append(current_cluster)
            current_cluster = [idx]
    clusters.append(current_cluster)
    for cluster in clusters:
        best_idx_in_cluster = cluster[np.argmax(pred_prob[cluster])]
        pred_classes[best_idx_in_cluster] = 1
    return pred_classes
def format_tads_from_boundaries(boundaries_1kb):
    """ Converts 1D 1kb boundaries to 10kb HiC matrix coordinates (start, end) pairs """
    b = set(boundaries_1kb)
    if 0 not in b: b.add(0)
    if 2239 not in b: b.add(2239)
    b_sorted = np.sort(list(b))
    tads = []
    for i in range(len(b_sorted) - 1):
        if b_sorted[i+1] - b_sorted[i] >= 10:
            tads.append((b_sorted[i] / 10.0, b_sorted[i+1] / 10.0))
    return tads
def plot_tads_on_hic_correctly(hic_matrix, tads, ax, title, box_color):
    """Renders the true underlying High-Depth Hi-C map and overlay TAD boundary triangles."""
    disp_hic = np.log1p(hic_matrix)
    val_80 = np.percentile(disp_hic[disp_hic > 0], 80) if np.sum(disp_hic > 0) > 0 else 1.0
    ax.imshow(disp_hic, cmap='Reds', vmin=0, vmax=val_80)
    for (start, end) in tads:
        width = end - start
        if width > 0:
            x_coords = [start, end, end]
            y_coords = [start, start, end]
            ax.plot(x_coords, y_coords, color=box_color, linewidth=3.0)
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.set_xlabel("Hi-C bins (10kb)", fontweight='bold')
    ax.set_ylabel("Hi-C bins (10kb)", fontweight='bold')
def main():
    print("==============================================")
    print(" chromSeek Tutorial: TAD Prediction")
    print("==============================================\n")
    dataset_path = 'sample_data.pt'
    ckpt_path = 'chromSeek_tad_prediction.pth'
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
    gt_tad = sample['targets']['tad'].squeeze().cpu().numpy()
    raw_hic = sample['info']['raw_hic']
    print(f"  --> DNA Sequence Shape      : {t_seq.shape}")
    print(f"  --> Background Hi-C Shape   : {t_hic.shape}")
    print(f"  --> Ground Truth Boundary   : {gt_tad.shape} (Sum: {int(np.sum(gt_tad))})\n")
    print(f"[Step 2] Initializing model and loading weights from {ckpt_path}...")
    model = DnaHicTadPredictor().to(device)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    print("[Step 3] Running Forward Pass (Inference)...")
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            out = model(t_seq, t_hic)
        pred_prob = F.softmax(out, dim=1)[:, 1, :].squeeze().cpu().numpy()
    pred_nms = cluster_and_nms(pred_prob, threshold=0.5, max_gap=50)
    print("[Step 4] Computing Metrics...")
    precision = precision_score(gt_tad, pred_nms, zero_division=0)
    recall = recall_score(gt_tad, pred_nms, zero_division=0)
    f1 = f1_score(gt_tad, pred_nms, zero_division=0)
    print(f"  --> Strict Precision : {precision:.4f}")
    print(f"  --> Strict Recall    : {recall:.4f}")
    print(f"  --> F1-Score         : {f1:.4f}\n")
    print("[Step 5] Generating Visualizations...")
    true_boundaries = np.where(gt_tad == 1)[0]
    true_tads = format_tads_from_boundaries(true_boundaries)
    pred_boundaries = np.where(pred_nms == 1)[0]
    pred_tads = format_tads_from_boundaries(pred_boundaries)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    plot_tads_on_hic_correctly(raw_hic, true_tads, axes[0], 
                               title=f"Ground Truth Annotations", 
                               box_color='lime')
    plot_tads_on_hic_correctly(raw_hic, pred_tads, axes[1], 
                               title=f"chromSeek Predictions (F1: {f1:.2f})", 
                               box_color='orange')
    plt.tight_layout()
    out_path = "chromSeek_tutorial_tad_prediction.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Successfully saved visualization to '{out_path}'.")
if __name__ == "__main__":
    main()
