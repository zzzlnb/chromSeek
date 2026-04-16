import os
import torch
import numpy as np
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import DNAToHic_BridgeModel
def compute_metrics(pred, gt):
    """Compute SSIM and Pearson Correlation Coefficient."""
    pcc, _ = pearsonr(pred.flatten(), gt.flatten())
    data_range = max(pred.max(), gt.max()) - min(pred.min(), gt.min())
    ssim_val = ssim(pred, gt, data_range=data_range)
    return pcc, ssim_val
def main():
    print("==============================================")
    print(" chromSeek Tutorial: Hi-C Enhancement")
    print("==============================================\n")
    dataset_path = 'sample_data.pt'
    ckpt_path = 'chromSeek_hic_enhancement.pth'
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
    gt_hic = sample['targets']['enhanced_hic'].squeeze().cpu().numpy()
    raw_hic = t_hic.squeeze().cpu().numpy()
    print(f"  --> DNA Sequence Shape  : {t_seq.shape}")
    print(f"  --> Sparse Hi-C Shape   : {t_hic.shape}")
    print(f"  --> Target Hi-C Shape   : {gt_hic.shape}\n")
    print(f"[Step 2] Initializing model and loading weights from {ckpt_path}...")
    model = DNAToHic_BridgeModel().to(device)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print("[Step 3] Running Forward Pass (Inference)...")
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            out = model(t_seq, t_hic)
        pred_hic = out.squeeze().cpu().numpy()
        pred_hic = np.clip(pred_hic, 0, None)                         
    print("[Step 4] Computing Metrics...")
    pcc, ssim_val = compute_metrics(pred_hic, gt_hic)
    print(f"  --> Pearson Correlation (PCC)     : {pcc:.4f}")
    print(f"  --> Structural Similarity (SSIM)  : {ssim_val:.4f}\n")
    print("[Step 5] Generating Visualizations...")
    def process_for_vis(mat):
        log_mat = np.log1p(mat)
        p80 = np.percentile(log_mat[log_mat > 0], 80) if np.sum(log_mat > 0) > 0 else 1.0
        return log_mat, p80
    vis_raw, p80_raw = process_for_vis(raw_hic)
    vis_pred, p80_pred = process_for_vis(pred_hic)
    vis_gt, p80_gt = process_for_vis(gt_hic)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(vis_raw, cmap='Reds', vmin=0, vmax=p80_raw)
    axes[0].set_title("Input Sparse Hi-C", fontsize=14)
    axes[0].axis('off')
    axes[1].imshow(vis_pred, cmap='Reds', vmin=0, vmax=p80_pred)
    axes[1].set_title(f"chromSeek Enhanced (SSIM: {ssim_val:.2f})", fontsize=14)
    axes[1].axis('off')
    axes[2].imshow(vis_gt, cmap='Reds', vmin=0, vmax=p80_gt)
    axes[2].set_title("High-Depth Ground Truth", fontsize=14)
    axes[2].axis('off')
    plt.tight_layout()
    out_path = "chromSeek_tutorial_hic_enhancement.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Successfully saved visualization to '{out_path}'.")
if __name__ == "__main__":
    main()
