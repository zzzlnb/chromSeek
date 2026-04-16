import os
import sys
import torch
import numpy as np
sys.path.append("/mnt/nfs/jyzhu/proj/ChromSeek/DNA_ChromSeek/ChromSeekFinal/gemini_dna_hic_to_loop_2class")
from dataset import DNA2LoopDataset
def main():
    ds = DNA2LoopDataset(
        hic_root="/mnt/nfs/jyzhu/dataset/lets_fk_hic/hic_patches_10kb_multiratio/",
        seq_root="/mnt/nfs/jyzhu/proj/ChromSeek/DNA_ChromSeek/DNA_only_10kb/genome_cache",
        loop_paths=["/mnt/nfs/jyzhu/dataset/lets_fk_hic/ENCODE_Data/Gm12878_Loop.bedpe.gz"],
        chroms=["chr18"],
        mode="val",
        rc_prob=0.0,
        max_samples=None, 
        downsample_hic=False 
    )
    idx_to_save = -1
    for i in range(len(ds)):
        _, raw_hic, target_loop, _ = ds.load_raw_data(i, jitter=0)
        num_loops = np.sum(target_loop)
        if 2 <= num_loops <= 10:
            idx_to_save = i
            break
    if idx_to_save != -1:
        res = ds[idx_to_save]; t_seq, t_hic_in, t_loop_target = res[0], res[1], res[2]
        _, raw_hic, _, _ = ds.load_raw_data(idx_to_save, jitter=0)
        sample_out = {
            'inputs': {'seq': t_seq, 'hic': t_hic_in},
            'targets': {'loop': t_loop_target},
            'info': {'raw_hic': raw_hic}
        }
        torch.save([sample_out], 'sample_data.pt')
        print(f"Extracted real Loop sample at index {idx_to_save} with {np.sum(target_loop)} loop anchors.")
    else:
        print("Failed to find a sample.")
if __name__ == "__main__":
    main()
