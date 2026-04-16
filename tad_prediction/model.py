import math
import sys
import torch
import torch.nn as nn
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from model_2kb_1d import MultiOmicsModel_2Mb                
class ConvBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
class ResBlock2D(nn.Module):
    def __init__(self, channels: int, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.bn1(x))
        out = self.conv1(out)
        out = self.act(self.bn2(out))
        out = self.conv2(out)
        out = self.dropout(out)
        return residual + out
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.pe.size(0):
            x = x + self.pe[:seq_len, :].to(x.device)
        else:
            x = x + self.pe[:seq_len, :]
        return self.dropout(x)
class DNAEncoderNew(nn.Module):
    def __init__(self, out_dim=384, pretrained_path=None):
        super().__init__()
        self.encoder_model = MultiOmicsModel_2Mb(num_tasks=4, pretrained_path=pretrained_path)
        self.proj = nn.Conv1d(64, out_dim, kernel_size=1)
    def forward(self, dna_seq):
        x = self.encoder_model.stem(dna_seq)
        for block in self.encoder_model.down_blocks:      x = block(x)
        for block in self.encoder_model.extra_down_blocks: x = block(x)
        x = self.encoder_model.body_proj(x)
        x = self.encoder_model.body_res(x)
        x = x.permute(0, 2, 1) 
        x = self.encoder_model.pos_encoder(x)
        x = self.encoder_model.transformer(x) 
        x = x.permute(0, 2, 1) 
        x = self.encoder_model.decoder(x)                 
        x = self.proj(x)                      
        return x
class HicTopologyEncoder2D(nn.Module):
    def __init__(self, base_filters=64):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Conv2d(1, base_filters // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters // 2), nn.GELU(),
            nn.Conv2d(base_filters // 2, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters), nn.GELU()
        )
        self.body = nn.Sequential(
            ResBlock2D(base_filters, dilation=1),
            ResBlock2D(base_filters, dilation=2),
            ResBlock2D(base_filters, dilation=4), 
            ResBlock2D(base_filters, dilation=2),
            ResBlock2D(base_filters, dilation=1)
        )
    def forward(self, x):
        return self.body(self.entry(x))
class HicToDnaBridge_New(nn.Module):
    def __init__(self, hic_dim=64, dna_target_dim=384):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Conv1d(hic_dim * 2, hic_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hic_dim * 2),
            nn.GELU()
        )
        self.upsampler = nn.Sequential(
            nn.ConvTranspose1d(hic_dim * 2, hic_dim, kernel_size=5, stride=5),
            nn.BatchNorm1d(hic_dim),
            nn.GELU()
        )
        self.out_conv = nn.Conv1d(hic_dim, dna_target_dim, kernel_size=1)
    def forward(self, x_2d):
        row_mean = torch.mean(x_2d, dim=3)
        row_max, _ = torch.max(x_2d, dim=3)
        x_1d = torch.cat([row_mean, row_max], dim=1)                 
        x_1d = self.projector(x_1d)
        x_up = self.upsampler(x_1d)                 
        out = self.out_conv(x_up)                  
        return out
class TadDecoder(nn.Module):
    def __init__(self, in_dim: int = 384, hidden_dim: int = 128):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose1d(in_dim, hidden_dim * 2, kernel_size=2, stride=2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.GELU(),
        )
        self.body = nn.Sequential(
            ConvBlock1D(hidden_dim * 2, hidden_dim, kernel_size=5, dropout=0.1),
            ConvBlock1D(hidden_dim, 64, kernel_size=5, dropout=0.1),
        )
        self.head = nn.Conv1d(64, 2, kernel_size=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)                     
        x = self.body(x)                    
        return self.head(x)                   
class DnaHicTadPredictor(nn.Module):
    def __init__(
        self,
        dna_embed_dim: int = 384,
        hic_embed_dim: int = 64,
        transformer_dim: int = 384,
        pretrained_path: str = None,
    ) -> None:
        super().__init__()
        self.dna_encoder = DNAEncoderNew(out_dim=dna_embed_dim, pretrained_path=pretrained_path)
        self.hic_encoder = HicTopologyEncoder2D(base_filters=hic_embed_dim)
        self.hic_bridge = HicToDnaBridge_New(hic_dim=hic_embed_dim, dna_target_dim=dna_embed_dim)
        self.fusion_gate = nn.Sequential(
            nn.Conv1d(dna_embed_dim * 2, transformer_dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.fusion_proj = nn.Conv1d(dna_embed_dim * 2, transformer_dim, kernel_size=1)
        self.pos_encoder = PositionalEncoding(transformer_dim, max_len=5000)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim, nhead=8, dim_feedforward=transformer_dim * 4,
            dropout=0.2, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.tad_decoder = TadDecoder(in_dim=transformer_dim, hidden_dim=128)
    def forward(self, dna_seq: torch.Tensor, hic_input: torch.Tensor) -> torch.Tensor:
        x_dna = self.dna_encoder(dna_seq)                                                    
        x_hic_2d = self.hic_encoder(hic_input)                                             
        x_hic = self.hic_bridge(x_hic_2d)                                                    
        combined = torch.cat([x_dna, x_hic], dim=1)                                
        gate = self.fusion_gate(combined)
        feat = self.fusion_proj(combined)
        x = feat * gate                                                                                        
        x = x.permute(0, 2, 1)                                                                          
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.permute(0, 2, 1)                                                                          
        final_out = self.tad_decoder(x)                                      
        return final_out
if __name__ == "__main__":
    model = DnaHicTadPredictor(pretrained_path=None)
    dna = torch.randn(1, 4, 2240000)
    hic = torch.randn(1, 1, 224, 224)
    out = model(dna, hic)
    print("Output shape:", out.shape)
    assert out.shape == (1, 2, 2240)
    print("Success.")
