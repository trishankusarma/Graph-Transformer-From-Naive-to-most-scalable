import torch
import torch.nn as nn
import torch.nn.functional as F

from .gcn_layer import GCN_Layer
from .transformer_layer import GraphTransformerLayer

class GCN_GT_Hybrid_Transformer(nn.Module):
    # this class has following sub-parts layers
    # 1. Concat feature dimension and k-eigen vectors
    # 2. Inital projection to the embed layer
    # 3. 2 Layers of GCN
    # 4. 2 Layers of Graph Transformer
    # 5. Final Layer Normalization & Dropout
    # 6. LogSoftmax Classification
    def __init__(self, input_feature_dim, config):
        super().__init__()
        # Step 1
        self.input_projection = nn.Linear(input_feature_dim + config.k_lap_pe, config.d_model, bias = False)
        # Step 3
        self.gcn_layers = nn.ModuleList([GCN_Layer(
                config.d_model, config.dropout
            ) for _ in range(config.num_gcn_layers)])
        # Step 4
        self.transformer_layers = nn.ModuleList([GraphTransformerLayer(
             config.d_model, config.dropout, config.num_heads, config.max_dist, config.d_ff
        ) for _ in range(config.num_layers)])
        # Step 5
        self.final_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        # Step 6
        self.classification = nn.Linear(config.d_model, config.d_classes, bias = False)
    
    def forward(self, x, k_eigen_vectors_pe, Adj_matrix, spd_matrix, config, **kwargs):
        
        # Step 1
        x = torch.cat([x, k_eigen_vectors_pe], dim = 1) #(num_nodes, feature_dim + k_lap_pe)
        x = F.dropout(x, p=config.dropout_input, training=self.training)
        # Step 2
        x = self.input_projection(x) #(num_nodes, d_model )
        # Step 3
        for layer in self.gcn_layers:
            x = layer(Adj_matrix, x)
        # Step 4
        for layer in self.transformer_layers:
            x = layer(x, spd_matrix)
        # Step 5 
        x = self.dropout(self.final_norm(x))
        # Step 6
        x = F.log_softmax(self.classification(x), dim = -1)

        return x

# if __name__ == '__main__':
#     import torch
#     import sys
#     sys.path.append('.')
#     from src.config import Config
#     config = Config()

#     model = GCN_GT_Hybrid_Transformer(1433, config)

#     x          = torch.randn(2708, 1433)
#     lap_pe     = torch.randn(2708, config.k_lap_pe)
#     adj_matrix = torch.zeros(2708, 2708)
#     spd_matrix = torch.randint(0, 12, (2708, 2708))

#     out = model(x, lap_pe, adj_matrix, spd_matrix, config)
#     print(out.shape)            # torch.Size([2708, 7])
#     print(out[0].exp().sum())   # tensor(1.0000)