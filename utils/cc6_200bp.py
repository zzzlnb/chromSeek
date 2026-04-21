import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List

# ================================================================
# 1. 基础组件
# ================================================================

class ConvBlock(nn.Module):
    """
    标准卷积块: Conv1d -> BatchNorm -> GELU
    支持 stride 下采样
    """
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, 
                 dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        # 计算 padding: 
        # 如果 stride=1, 这是 same padding。
        # 如果 stride>1, 这能保证整除情况下的维度正确缩放 (L_out = L_in / stride)
        padding = (dilation * (kernel_size - 1)) // 2
        
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, 
                      padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.block(x)

class ResidualBlock(nn.Module):
    """
    预激活残差块: 保持维度不变，增加深度
    """
    def __init__(self, channels: int, kernel_size: int = 3, 
                 dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = (dilation * (kernel_size - 1)) // 2
        
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, 
                               padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, 
                               padding=padding, dilation=dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = F.gelu(self.bn1(x))
        out = self.conv1(out)
        out = F.gelu(self.bn2(out))
        out = self.conv2(out)
        out = self.dropout(out)
        return residual + out

class PositionalEncoding(nn.Module):
    """
    正弦位置编码
    修改：max_len 默认设为 10000，足够覆盖 2240 (448k/200) 的长度
    """
    def __init__(self, d_model, max_len=10000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Dim)
        # 动态截取对应长度的 PE
        seq_len = x.size(1)
        if seq_len > self.pe.size(0):
            raise ValueError(f"Sequence length {seq_len} exceeds PositionalEncoding max_len {self.pe.size(0)}")
        
        x = x + self.pe[:seq_len, :]
        return self.dropout(x)

# ================================================================
# 2. 核心机制: Cell-Cross-Attention
# ================================================================

class CellAttention(nn.Module):
    """
    使用细胞 Embedding 对基因组特征进行门控调制
    """
    def __init__(self, hidden_dim, cell_dim):
        super().__init__()
        self.to_q = nn.Linear(hidden_dim, hidden_dim)
        self.to_k = nn.Linear(cell_dim, hidden_dim)
        self.to_v = nn.Linear(cell_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, cell_embed):
        # x: (B, L, C)
        # cell_embed: (B, cell_dim)
        B, L, C = x.shape
        
        q = self.to_q(x)                    # (B, L, C)
        k = self.to_k(cell_embed).view(B, 1, C) # (B, 1, C)
        v = self.to_v(cell_embed).view(B, 1, C) # (B, 1, C)

        # Attention Score & Sigmoid Gating
        dots = (q @ k.transpose(-2, -1)) * self.scale
        attn = dots.sigmoid() 
        
        out = attn * v 
        out = self.out_proj(out)
        
        return x + out

# ================================================================
# 3. 模型主类: CellSpecificOmicsModel_448k
# ================================================================

class CellSpecificOmicsModel_448k(nn.Module):
    """
    专为 448kb 输入 (224*2000) 和 200bp 输出分辨率设计的版本
    Total Downsampling: 200x
    Transformer Depth: 6 layers
    """
    def __init__(self, 
                 num_cells: int,
                 embed_dim: int = 64,
                 num_tasks: int = 1,
                 # 输入长度 448,000
                 seq_len: int = 448000, 
                 encoder_channels: List[int] = [64, 128, 256, 384],
                 body_dim: int = 384
                 ):
        super().__init__()
        
        self.num_tasks = num_tasks
        
        # --- 1. Embedding ---
        self.cell_embedder = nn.Embedding(num_cells, embed_dim)

        # --- 2. Downsampler Tower (Total 200x) ---
        # 目标: 448,000 -> 2,240 (Scale factor 200)
        # 组合: 2 (Stem) * 5 * 5 * 4 = 200
        
        # Stem: Stride 2 (448k -> 224k)
        self.stem = ConvBlock(4, encoder_channels[0], kernel_size=15, stride=2) 
        
        self.down_blocks = nn.ModuleList()
        in_c = encoder_channels[0]
        
        # 修改点：Strides 设置为 [5, 5, 4]
        factors = [5, 5, 4] 
        
        # 动态构建下采样层
        # Layer 1: stride 5 (224k -> 44800)
        # Layer 2: stride 5 (44800 -> 8960)
        # Layer 3: stride 4 (8960 -> 2240) -> 进入 Body
        
        # 确保 encoder_channels 长度够用，这里复用 channels 定义
        # encoder_channels[1:] 对应 128, 256, 384
        for i, (out_c, pool) in enumerate(zip(encoder_channels[1:], factors)):
            self.down_blocks.append(
                ConvBlock(in_c, out_c, kernel_size=5, stride=pool)
            )
            in_c = out_c
            
        # --- 3. Body (Bottleneck L=2240) ---
        self.body_proj = ConvBlock(encoder_channels[-1], body_dim, kernel_size=1)
        
        # A. 局部特征提取 (ResBlocks)
        self.body_res = nn.Sequential(
            ResidualBlock(body_dim, dilation=1),
            ResidualBlock(body_dim, dilation=2),
            ResidualBlock(body_dim, dilation=4)
        )
        
        # B. 全局上下文 (Transformer)
        # 修改点：max_len=10000 (安全覆盖 2240)
        self.pos_encoder = PositionalEncoding(body_dim, max_len=10000)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=body_dim,
            nhead=8,
            dim_feedforward=body_dim * 4,
            dropout=0.2,
            activation='gelu',
            batch_first=True # (B, L, C)
        )
        
        # 修改点：深度增加到 6 层
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # C. 细胞特异性调制
        self.cell_attn1 = CellAttention(hidden_dim=body_dim, cell_dim=embed_dim)
        self.cell_attn2 = CellAttention(hidden_dim=body_dim, cell_dim=embed_dim)
        
        # --- 4. Decoder ---
        # 保持分辨率不变 (L=2240)，只做特征提炼
        self.decoder = nn.Sequential(
            ConvBlock(body_dim, 256, kernel_size=5, dropout=0.1),
            ConvBlock(256, 256, kernel_size=5, dropout=0.1),
            ConvBlock(256, 128, kernel_size=5, dropout=0.1),
            ConvBlock(128, 64, kernel_size=5, dropout=0.1)
        )
        
        # --- 5. Final Head ---
        self.final_head = nn.Sequential(
            nn.Conv1d(64, num_tasks, kernel_size=1)
        )

    def forward(self, dna_seq, cell_id):
        # dna_seq: (B, 4, 448000)
        
        cell_emb = self.cell_embedder(cell_id) 
        
        # --- Encoder ---
        x = self.stem(dna_seq)
        for block in self.down_blocks:
            x = block(x)
            
        # --- Body ---
        x = self.body_proj(x)
        x = self.body_res(x)
        
        # --- Transformer (B, L, C) ---
        x = x.permute(0, 2, 1) 
        x = self.pos_encoder(x)
        x = self.transformer(x) 
        
        # --- Cell Attention ---
        x = self.cell_attn1(x, cell_emb)
        x = self.cell_attn2(x, cell_emb)
        
        # --- Decoder (B, C, L) ---
        x = x.permute(0, 2, 1) 
        x = self.decoder(x)
        
        # Output
        out = self.final_head(x) # (B, num_tasks, 2240)
        
        return out

# ================================================================
# 维度与逻辑检查
# ================================================================
if __name__ == "__main__":
    # 模拟参数
    BATCH_SIZE = 2
    SEQ_LEN = 448000    # 224 * 2000 bp
    NUM_TASKS = 10      # 对应输出的通道数
    
    model = CellSpecificOmicsModel_448k(num_cells=6, embed_dim=64, num_tasks=NUM_TASKS)
    
    # 1. 创建 Dummy Input
    dummy_dna = torch.randn(BATCH_SIZE, 4, SEQ_LEN)
    dummy_cell = torch.randint(0, 6, (BATCH_SIZE,))
    
    print(f"Input Shape: {dummy_dna.shape}")
    
    # 2. Forward Pass
    output = model(dummy_dna, dummy_cell)
    
    print(f"Output Shape: {output.shape}")
    
    # 3. 验证维度
    # 预期长度: 448000 / 200 = 2240
    expected_len = SEQ_LEN // 200
    assert output.shape == (BATCH_SIZE, NUM_TASKS, expected_len), \
        f"Mismatch! Expected last dim {expected_len}, got {output.shape[-1]}"
        
    print(f"✅ Test Passed: Output correctly downsampled to {output.shape[-1]} bins (200bp resolution).")