import torch
import torch.nn as nn
from model import DNAEncoderNew, HicTopologyEncoder2D, HicToDnaBridge_New, PositionalEncoding, TrackDecoder
class MultiOmicsPredictor(nn.Module):
    def __init__(self, track_names, cells=["GM12878", "K562"], dna_embed_dim=384, hic_embed_dim=64, transformer_dim=384, pretrained_path=None):
        super().__init__()
        self.track_names = track_names
        self.cells = cells
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
        self.cell_decoders = nn.ModuleDict({
            cell: nn.ModuleDict({
                track: TrackDecoder(in_dim=transformer_dim, hidden_dim=128)
                for track in track_names
            }) for cell in cells
        })
    def forward(self, dna_seq, hic_input, cell_names):
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
        outputs = []
        for i, cell in enumerate(cell_names):
            decoder_dict = self.cell_decoders[cell] if cell in self.cell_decoders else self.cell_decoders[self.cells[0]]
            cell_out = []
            for track in self.track_names:
                out = decoder_dict[track](x[i:i+1])                                      
                cell_out.append(out)
            final_out = torch.cat(cell_out, dim=1)                                         
            outputs.append(final_out)
        return torch.cat(outputs, dim=0)                         
