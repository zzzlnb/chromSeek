import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from multitask_model import MultiOmicsPredictor
def main():
    model_path = "./transfer_multiomics_best.pth"
    data_path = "./sample_data.pt"
    if not os.path.exists(model_path) or not os.path.exists(data_path):
        print("Required files are missing. Please make sure transfer_multiomics_best.pth and sample_data.pt exist.")
        return
    sample = torch.load(data_path, map_location="cpu", weights_only=False)
    seq = sample["seq"].unsqueeze(0)                                                                      
    hic = sample["hic"]                                                                                                       
    if hic.dim() == 3:
        hic = hic.unsqueeze(0)
    target_z = sample["target_z"]                                            
    target_log = sample["target_log"]                                    
    chrom = sample["chrom"]
    start_bp = sample["start_bp"]
    cell = sample["cell"]
    track_order = sample["track_order"]
    model = MultiOmicsPredictor(
        track_names=track_order,
        cells=["GM12878", "K562"],
        pretrained_path=None
    )
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_k = k[7:] if k.startswith("module.") else k
        cleaned_state_dict[new_k] = v
    model.load_state_dict(cleaned_state_dict, strict=False)
    model.eval()
    with torch.no_grad():
        pred_z = model(seq, hic, [cell])
    pred = pred_z.squeeze(0).cpu().numpy()
    target = target_z.numpy()
    tracks_to_plot = ["CTCF", "ATAC", "MYC", "H3K27ac", "H3K4me3"]
    tracks_to_plot = [t for t in tracks_to_plot if t in track_order]
    fig, axes = plt.subplots(nrows=len(tracks_to_plot), ncols=1, figsize=(10, 2*len(tracks_to_plot)), sharex=True)
    if len(tracks_to_plot) == 1:
        axes = [axes]
    color_map = {
        "CTCF": "#1f77b4", "ATAC": "#ff7f0e", "MYC": "#2ca02c",
        "H3K27ac": "#d62728", "H3K4me3": "#9467bd", "WGBS": "#8c564b"
    }
    for ax, t_name in zip(axes, tracks_to_plot):
        t_idx = track_order.index(t_name)
        color = color_map.get(t_name, "#333333")
        t_pred = pred[t_idx]
        t_true = target[t_idx]
        smooth_window = 10
        t_pred = np.convolve(t_pred, np.ones(smooth_window)/smooth_window, mode='same')
        t_true = np.convolve(t_true, np.ones(smooth_window)/smooth_window, mode='same')
        clip_val = max(np.percentile(t_true, 99), np.percentile(t_pred, 99))
        if clip_val <= 0: clip_val = 1.0
        t_pred = np.clip(t_pred, -1, clip_val)
        t_true = np.clip(t_true, -1, clip_val)
        x = np.arange(len(t_pred))
        ax.fill_between(x, t_true, color="gray", alpha=0.3, label="Ground Truth (Z)")
        ax.plot(x, t_pred, color=color, linewidth=2, label=f"Predicted (Z)")
        ax.set_ylabel(t_name, rotation=0, labelpad=50, fontdict={'fontweight': 'bold', 'fontsize': 14}, va='center')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])
        ax.legend(loc="upper right")
    plt.suptitle(f"Multi-Omics Prediction ({cell} {chrom}:{start_bp})", fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("multiomics_prediction_example.png", dpi=150)
    print("Saved prediction example to multiomics_prediction_example.png")
if __name__ == "__main__":
    main()
