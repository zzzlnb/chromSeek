import os
import gzip
import random
import sys
from typing import Dict, List, Optional, Sequence
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.sparse import coo_matrix
def dense2tag(matrix):
    matrix = np.triu(matrix)
    matrix = np.maximum(matrix, 0)
    tag_len = np.sum(matrix).astype(np.int64) 
    if tag_len == 0:
        return np.zeros((0, 2), dtype=np.int32), 0
    tag_mat = np.zeros((tag_len, 2), dtype=np.int32)
    coo_mat = coo_matrix(matrix)
    row, col, data = coo_mat.row, coo_mat.col, coo_mat.data.astype(np.int64)
    start_idx = 0
    for i in range(len(row)):
        end_idx = start_idx + data[i]
        tag_mat[start_idx:end_idx, :] = (row[i], col[i])
        start_idx = end_idx
    return tag_mat, tag_len
def tag2dense(tag, nsize):
    if tag.shape[0] == 0:
        return np.zeros((nsize, nsize), dtype=np.int32)
    coo_data, data = np.unique(tag, axis=0, return_counts=True)
    row, col = coo_data[:, 0], coo_data[:, 1]
    dense_mat = coo_matrix((data, (row, col)), shape=(nsize, nsize)).toarray()
    dense_mat = dense_mat + np.triu(dense_mat, k=1).T
    return dense_mat
def downsampling_deephic(matrix, down_ratio):
    matrix_int = matrix.astype(np.int32)
    tag_mat, tag_len = dense2tag(matrix_int)
    if tag_len == 0 or down_ratio <= 1:
        return matrix_int.copy()
    n_samples = max(1, int(tag_len / down_ratio)) 
    sample_idx = np.random.choice(tag_len, n_samples, replace=False)
    sample_tag = tag_mat[sample_idx]
    down_mat = tag2dense(sample_tag, matrix.shape[0])
    return down_mat
sys.path.append("/mnt/nfs/jyzhu/proj/ChromSeek/DNA_ChromSeek/ChromSeekFinal")
from DNA_loader import build_one_hot_table, get_encoded_segment, load_chr                
from utils import (
    BIN_SIZE,
    DNA_BP_WINDOW,
    NUM_1KB_BINS,
    HIC_RESOLUTION,
)
TAD_BEDPE_MAP = {
    'A673_Hic': "/mnt/nfs/jyzhu/dataset/lets_fk_hic/ENCODE_Data/A673_Tad.bedpe.gz",
    'Caco2_Hic': "/mnt/nfs/jyzhu/dataset/lets_fk_hic/ENCODE_Data/Caco2_Tad.bedpe.gz",
    'Calu3_Hic': "/mnt/nfs/jyzhu/dataset/lets_fk_hic/ENCODE_Data/Calu3_Tad.bedpe.gz",
    'CH12LX_Hic': "/mnt/nfs/jyzhu/dataset/lets_fk_hic/ENCODE_Data/CH12LX_Tad.bedpe.gz",
    'GM12878_Hic': "/mnt/nfs/jyzhu/dataset/lets_fk_hic/ENCODE_Data/Gm12878_Tad.bedpe.gz",
    'GM23248_Hic': "/mnt/nfs/jyzhu/dataset/lets_fk_hic/ENCODE_Data/GM23248_Tad.bedpe.gz",
    'HepG2_Hic': "/mnt/nfs/jyzhu/dataset/lets_fk_hic/ENCODE_Data/HepG2_Tad.bedpe.gz",
    'IMR90_Hic': "/mnt/nfs/jyzhu/dataset/lets_fk_hic/ENCODE_Data/IMR90_Tad.bedpe.gz",
    'MCF10A_Hic': "/mnt/nfs/jyzhu/dataset/lets_fk_hic/ENCODE_Data/MCF10A_Tad.bedpe.gz",
    'Mcf7_Hic': "/mnt/nfs/jyzhu/dataset/lets_fk_hic/ENCODE_Data/Mcf7_Tad.bedpe.gz",
    'OCILY7_Hic': "/mnt/nfs/jyzhu/dataset/lets_fk_hic/ENCODE_Data/OCILY7_Tad.bedpe.gz",
    'Panc1_Hic': "/mnt/nfs/jyzhu/dataset/lets_fk_hic/ENCODE_Data/Panc1_Tad.bedpe.gz",
    'PC3_Hic': "/mnt/nfs/jyzhu/dataset/lets_fk_hic/ENCODE_Data/PC3_Tad.bedpe.gz",
    'PC9_Hic': "/mnt/nfs/jyzhu/dataset/lets_fk_hic/ENCODE_Data/PC9_Tad.bedpe.gz",
    'T47D_Hic': "/mnt/nfs/jyzhu/dataset/lets_fk_hic/ENCODE_Data/T47D_Tad.bedpe.gz",
}
class DNA2TadDataset(Dataset):
    def __init__(
        self,
        hic_root: str,
        seq_root: str,
        cell_lines_list: Sequence[str],
        chroms: Sequence[str],
        mode: str = "train",
        rc_prob: float = 0.5,
        max_samples: Optional[int] = None,
        sample_stride: int = 1,
        use_downsample: bool = False,
    ) -> None:
        super().__init__()
        self.hic_root = hic_root
        self.seq_root = seq_root
        self.chroms = list(chroms)
        self.mode = mode
        self.rc_prob = rc_prob
        self.max_samples = max_samples
        self.sample_stride = max(1, sample_stride)
        self.use_downsample = use_downsample
        self.one_hot_table = build_one_hot_table()
        self.cell_lines_list = list(cell_lines_list)
        self.tad_labels: Dict[str, Dict[str, np.ndarray]] = self._load_tad_labels()
        self.chr_seqs = {chrom: load_chr(chrom, seq_root, mmap=True) for chrom in self.chroms}
        self.samples = self._collect_samples()
        if self.max_samples is not None:
            self.samples = self.samples[: self.max_samples]
    def _load_tad_labels(self) -> Dict[str, Dict[str, np.ndarray]]:
        all_tad_arrays = {}
        for cell in self.cell_lines_list:
            if cell not in TAD_BEDPE_MAP:
                continue
            bedpe_path = TAD_BEDPE_MAP[cell]
            tad_arrays = {chrom: np.zeros(300000, dtype=np.int64) for chrom in self.chroms}
            if os.path.exists(bedpe_path):
                with gzip.open(bedpe_path, 'rt') as f:
                    for line in f:
                        if line.startswith('chr'):
                            parts = line.strip().split('\t')
                            chrom = parts[0]
                            if chrom in tad_arrays:
                                start = int(parts[1])
                                end = int(parts[2])
                                start_bin = start // BIN_SIZE
                                end_bin = end // BIN_SIZE
                                if start_bin < len(tad_arrays[chrom]):
                                    tad_arrays[chrom][start_bin] = 1
                                if end_bin < len(tad_arrays[chrom]):
                                    tad_arrays[chrom][end_bin] = 1
            all_tad_arrays[cell] = tad_arrays
        return all_tad_arrays
    def _collect_samples(self) -> List[Dict]:
        samples: List[Dict] = []
        for cell in self.cell_lines_list:
            cell_dir = os.path.join(self.hic_root, cell)
            if not os.path.isdir(cell_dir):
                continue
            for chrom in self.chroms:
                chrom_dir = os.path.join(cell_dir, chrom)
                meta_path = os.path.join(chrom_dir, "meta.txt")
                if not os.path.exists(meta_path):
                    continue
                with open(meta_path, "r", encoding="utf-8") as handle:
                    starts = [int(line.strip()) for line in handle if line.strip()]
                for idx, start_idx in enumerate(starts):
                    if idx % self.sample_stride != 0:
                        continue
                    start_bp = start_idx * HIC_RESOLUTION
                    end_bp = start_bp + DNA_BP_WINDOW
                    start_bin = start_bp // BIN_SIZE
                    end_bin = start_bin + NUM_1KB_BINS
                    if chrom not in self.chr_seqs:
                        continue
                    if end_bp > len(self.chr_seqs[chrom]):
                        continue
                    pt_path = os.path.join(chrom_dir, f"{start_idx}.pt")
                    if not os.path.exists(pt_path):
                        continue
                    samples.append(
                        {
                            "cell": cell,
                            "chrom": chrom,
                        "start_idx": start_idx,
                        "start_bp": start_bp,
                        "pt_path": pt_path,
                    }
                )
        return samples
    def __len__(self) -> int:
        return len(self.samples)
    def _load_hic_high_depth(self, pt_path: str) -> np.ndarray:
        hic_data = torch.load(pt_path, map_location="cpu", weights_only=False)
        if isinstance(hic_data, dict) and "target" in hic_data:
            hic_target = hic_data["target"]
        else:
            hic_target = hic_data
        if hasattr(hic_target, "numpy"):
            hic_target = hic_target.numpy()
        hic_target = np.asarray(hic_target, dtype=np.float32)
        hic_target = np.nan_to_num(hic_target, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        hic_target = np.clip(hic_target, a_min=0.0, a_max=None)
        return hic_target
    def _load_sequence(self, chrom: str, start_bp: int, end_bp: int) -> np.ndarray:
        chr_arr = self.chr_seqs[chrom]
        seq = get_encoded_segment(chr_arr, start_bp, end_bp, self.one_hot_table)
        return seq.astype(np.float32, copy=False)
    def _load_targets(self, cell: str, chrom: str, start_bp: int) -> np.ndarray:
        start_bin = start_bp // BIN_SIZE
        end_bin = start_bin + NUM_1KB_BINS
        if cell in self.tad_labels and chrom in self.tad_labels[cell]:
            labels = self.tad_labels[cell][chrom][start_bin:end_bin].copy()
        else:
            labels = np.zeros(NUM_1KB_BINS, dtype=np.int64)
        return labels
    def __getitem__(self, index: int):
        sample = self.samples[index]
        hic_target = self._load_hic_high_depth(sample["pt_path"])
        seq = self._load_sequence(sample["chrom"], sample["start_bp"], sample["start_bp"] + DNA_BP_WINDOW)
        tad_label = self._load_targets(sample["cell"], sample["chrom"], sample["start_bp"])
        if self.use_downsample and self.mode == "train":
            ratio = random.randint(1, 10) ** 2
            hic_target = downsampling_deephic(hic_target, ratio)
        if self.mode == "train" and random.random() < self.rc_prob:
            seq = np.flip(seq, axis=0).copy()
            seq = seq[:, [3, 2, 1, 0]]
            hic_target = np.flip(hic_target, axis=(0, 1)).copy()
            tad_label = np.flip(tad_label, axis=0).copy()
        seq_tensor = torch.from_numpy(seq).float().transpose(0, 1)
        hic_tensor = torch.from_numpy(np.log1p(hic_target)).float()
        hic_max = hic_tensor.max()
        if hic_max > 0:
            hic_tensor = hic_tensor / hic_max
        hic_tensor = hic_tensor.unsqueeze(0)
        tad_label_tensor = torch.from_numpy(tad_label).long()
        return (
            seq_tensor,
            hic_tensor,
            tad_label_tensor,
            sample["chrom"],
            int(sample["start_bp"]),
        )
