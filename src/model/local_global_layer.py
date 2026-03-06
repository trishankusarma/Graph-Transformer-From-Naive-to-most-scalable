# Motivation
# one layer combining GAT + GT + gate
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gat_layer import GAT_Layer
from .transformer_layer import GraphTransformerLayer

class Local_Global_Transformer_Layer(nn.Module):
    def __init__(self, config, gate_init = 0.0):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

        self.GAT_layer = GAT_Layer(config.d_model, config.leaky_relu_slope, config.dropout)
        self.Transformer_Layer = GraphTransformerLayer(config)

        self.gate = nn.Parameter(torch.tensor(gate_init))   # sigmoid(2.0)=0.88 → local dominant
        self.dropout = nn.Dropout(config.dropout)

        self.ff1 = nn.Linear(config.d_model, config.d_ff, bias = True)
        self.ff2 = nn.Linear(config.d_ff, config.d_model, bias = True)

    def forward(self, x, edge_list, spd_matrix):
        # x(num_nodes, d_model) 
        # Step 1: Layer Normalization
        x_norm = self.norm1(x)
        # Step 2: Apply GAT and GT parallely
        x1 = self.GAT_layer(x_norm, edge_list)
        x2 = self.Transformer_Layer(x_norm, spd_matrix)
        # Step 3: Combine via x = alpha*x1 + (1-alpha)*x2
        alpha = torch.sigmoid(self.gate)
        x_combined = alpha * x1 + (1-alpha) * x2
        # Step 4: apply Dropout and apply residual connection
        x = x + self.dropout(x_combined)
        # Step 5: LayerNormalization
        x_norm = self.norm2(x)
        # Step 6: Apply FFNN -> Dropout -> apply residual connection
        x_ff1_forward = self.dropout(F.gelu(self.ff1(x_norm)))
        x =  x + self.dropout(self.ff2(x_ff1_forward))

        return x