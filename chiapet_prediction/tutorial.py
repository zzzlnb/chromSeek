import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score
from model import ChromSeekChiapetModel
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("Loading model...")
    model = ChromSeekChiapetModel(pretrained_path=None).to(device)
    ckpt_path = "chromSeek_chiapet_prediction.pth"
    if not os.path.exists(ckpt_path):
        print(f"Error: View weights file not found at {ckpt_path}")
        return
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("Loading sample data...")
    data_path = "sample_data.pt"
    if not os.path.exists(data_path):
        print(f"Error: Sample data not found at {data_path}")
        return
    data = torch.load(data_path, weights_only=False)
    seq_tensor = data['seq'].unsqueeze(0).to(device)                   
    hic_tensor = data['hic'].unsqueeze(0).to(device)                    
    true_tensor = data['gt_chiapet']
    info = data['info']
    print("Running inference...")
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", enabled=True):
            _, pred_2d = model(seq_tensor, hic_tensor)
        pred_prob = torch.sigmoid(pred_2d).squeeze().cpu().numpy()
    true_binary = true_tensor.numpy()
    hic_raw = data.get('hic_raw', hic_tensor.squeeze().cpu().numpy())
    mask = np.triu(np.ones_like(true_binary), k=1).astype(bool)
    flat_true = true_binary[mask].astype(int)
    flat_pred = pred_prob[mask]
    auc_val = 0.0
    f1_val = 0.0
    if flat_true.sum() > 0 and flat_true.sum() < flat_true.size:
        try:
            auc_val = roc_auc_score(flat_true, flat_pred)
            f1_val = f1_score(flat_true, (flat_pred > 0.5).astype(int))
        except ValueError:
            pass
    print(f"Metrics - AUC: {auc_val:.3f}, F1: {f1_val:.3f}")
    print("Generating visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    title_str = f"ChromSeek ChIA-PET Prediction | {info.get('cell', 'Unknown')} | {info.get('chrom', 'Unknown')} | start: {info.get('start', 'Unknown')}"
    fig.suptitle(title_str, fontsize=14)
    vmin, vmax = 0, 1.0
    if isinstance(hic_raw, np.ndarray):
        hic_disp = np.log1p(hic_raw)
        vmax = max(1.0, np.percentile(hic_disp, 80))
    else:
        hic_disp = hic_tensor.squeeze().cpu().numpy()
    axes[0].imshow(hic_disp, cmap="RdBu_r", vmin=0, vmax=vmax)
    threshold = 0.5
    pred_prob_disp = np.where(pred_prob >= threshold, pred_prob, 0)
    true_binary_disp = np.where(true_binary >= threshold, true_binary, 0)
    axes[1].imshow(pred_prob_disp, cmap="Blues", vmin=0, vmax=1.0)
    axes[1].set_title(f"Predicted Prob (>0.5): AUC={auc_val:.3f}\nF1={f1_val:.3f}")
    axes[1].axis("off")
    axes[2].imshow(true_binary_disp, cmap="Blues", vmin=0, vmax=1.0)
    axes[2].set_title("GT ChIA-PET (Filtered)")
    axes[2].axis("off")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_img = "chromSeek_tutorial_chiapet_prediction.png"
    plt.savefig(out_img, dpi=150)
    plt.close()
    print(f"Success! Visualization saved to {out_img}")
if __name__ == "__main__":
    main()
