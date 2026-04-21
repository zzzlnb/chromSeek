import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
from typing import List
from cc6_200bp import ConvBlock, ResidualBlock, PositionalEncoding
import os
class MultiOmicsModel_2Mb(nn.Module):
    """
    Designed for 2,240,000 bp input and 2kb (2000 bp) output resolution.
    Total Downsampling: 2000x
    Outputs: 4 tasks
    """
    def __init__(self, 
                 num_tasks: int = 4,
                 body_dim: int = 384,
                 pretrained_path: str = None):
        super().__init__()
        self.num_tasks = num_tasks
        self.stem = ConvBlock(4, 64, kernel_size=15, stride=2) 
        self.down_blocks = nn.ModuleList()
        self.down_blocks.append(ConvBlock(64, 128, kernel_size=5, stride=5))
        self.down_blocks.append(ConvBlock(128, 256, kernel_size=5, stride=5))
        self.down_blocks.append(ConvBlock(256, 384, kernel_size=5, stride=4))
        self.extra_down_blocks = nn.ModuleList()
        self.extra_down_blocks.append(ConvBlock(384, 384, kernel_size=5, stride=5))
        self.extra_down_blocks.append(ConvBlock(384, 384, kernel_size=5, stride=2))
        self.body_proj = ConvBlock(384, body_dim, kernel_size=1)
        self.body_res = nn.Sequential(
            ResidualBlock(body_dim, dilation=1),
            ResidualBlock(body_dim, dilation=2),
            ResidualBlock(body_dim, dilation=4)
        )
        self.pos_encoder = PositionalEncoding(body_dim, max_len=10000)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=body_dim,
            nhead=8,
            dim_feedforward=body_dim * 4,
            dropout=0.2,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.decoder = nn.Sequential(
            ConvBlock(body_dim, 256, kernel_size=5, dropout=0.1),
            ConvBlock(256, 256, kernel_size=5, dropout=0.1),
            ConvBlock(256, 128, kernel_size=5, dropout=0.1),
            ConvBlock(128, 64, kernel_size=5, dropout=0.1)
        )
        self.final_head = nn.Sequential(
            nn.Conv1d(64, num_tasks, kernel_size=1),
            nn.Softplus()
        )
        if pretrained_path:
            self._load_pretrained(pretrained_path)
    def _load_pretrained(self, pretrained_path):
        print(f"[Model] Loading pretrained weights from {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location='cpu')
        my_state = self.state_dict()
        new_state = {}
        loaded_keys = []
        for k, v in state_dict.items():
            new_k = k.replace('module.', '')
            if new_k in my_state:
                if my_state[new_k].shape == v.shape:
                    new_state[new_k] = v
                    loaded_keys.append(new_k)
        self.load_state_dict(new_state, strict=False)
        print(f"[Model] Successfully injected {len(loaded_keys)} matching weight matrices into the model.")
    def forward(self, dna_seq):
        x = self.stem(dna_seq)
        for block in self.down_blocks:
            x = block(x)
        for block in self.extra_down_blocks:
            x = block(x)
        x = self.body_proj(x)
        x = self.body_res(x)
        x = x.permute(0, 2, 1) 
        x = self.pos_encoder(x)
        x = self.transformer(x) 
        x = x.permute(0, 2, 1) 
        x = self.decoder(x)
        out = self.final_head(x)                        
        return out
if __name__ == "__main__":
    pretrained_path_default = os.path.join(os.path.dirname(__file__), 'best_model_448k_200bp.pth')
    model = MultiOmicsModel_2Mb(num_tasks=4, pretrained_path=pretrained_path_default)
    dummy = torch.randn(2, 4, 2240000)
    out = model(dummy)
    print("Output shape:", out.shape)
