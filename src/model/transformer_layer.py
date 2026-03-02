# Motivation : 
# -- single GT layer (attn + FFN + LN + residual)
import torch.nn as nn
import torch.nn.functional as F

from attention import GraphTransformerAttention

class GraphTransformerLayer(nn.Module):
    # In one Graph Transformer layer, we have 2 sublayers
        # Sublayer 1
        # LayerNorm -> Attention -> Dropout -> Residual 
        # Sublayer 2
        # LayerNorm -> FFN1 -> GeLU -> Dropout -> FFN2 -> Dropout -> Residual
    def __init__(self, d_model, dropout, num_heads, max_dist, d_ff):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = GraphTransformerAttention(d_model, dropout, num_heads, max_dist)
        
        self.dropout = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_ff, bias = True) # (d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model, bias = True) # (d_ff, d_model)
    
    def forward(self, x, spd_matrix):

        # For sublayer1
        x = x + self.dropout(self.attention(self.norm1(x), spd_matrix))

        # For sublayer2
        x = x + self.dropout(self.ff2(self.dropout(F.gelu(self.ff1(self.norm2(x))))))

        return x

# if __name__ == "__main__":
#     import torch
#     layer = GraphTransformerLayer(64, 0.4, 4, 10, 256)
#     x = torch.randn(2708, 64)
#     spd_matrix = torch.randint(0, 12, (2708, 2708))
#     out = layer(x, spd_matrix)
#     print(out.shape)   # torch.Size([2708, 64])