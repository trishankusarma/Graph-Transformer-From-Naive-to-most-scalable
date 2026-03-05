# Motivation
# stack + classifier
import torch
import torch.nn as nn
import torch.nn.functional as F

from .local_global_layer import Local_Global_Transformer_Layer

class LocalGlobalTransformer(nn.Module):
    def __init__(self, input_feature_dim, config):
        super().__init__()

        # Step 1
        self.dropout = nn.Dropout(config.dropout_input)
        self.input_projection = nn.Linear(input_feature_dim + config.k_lap_pe, config.d_model, bias = False)
        # Step 2
        self.layers = nn.ModuleList([
            Local_Global_Transformer_Layer(config.d_model, config.leaky_relu_slope, config.dropout, config.num_heads, config.max_dist, config.d_ff)
        for _ in range(config.num_lg_layers)])
        # Step 3
        self.final_norm = nn.LayerNorm(config.d_model)
        # Step 4
        self.classification = nn.Linear(config.d_model, config.d_classes, bias = False)

    def forward(self, x, k_eigen_vectors_pe, spd_matrix, edge_list, **kwargs):
        # Step 1 : Concat and input dropout for generalization -> project
        combined_x = torch.cat([x, k_eigen_vectors_pe], dim = 1)
        x = self.dropout( self.input_projection(combined_x))

        # Step 2: Layering
        for layer in self.layers:
            x = layer(x, edge_list, spd_matrix)
        
        # Step 3 and 4
        x = self.classification(self.final_norm(x))

        return F.log_softmax(x, dim = -1)
        
# if __name__ == "__main__":
#     import torch
#     num_nodes = 20
#     input_feature_dim = 1433
#     x = torch.randn(num_nodes, input_feature_dim)
    
#     edge_list = []
#     p = 0.4
#     for i in range(num_nodes):
#         for j in range(i+1, num_nodes):
#             import random
#             if random.random() < p:
#                 edge_list.append([i, j])
#                 edge_list.append([j, i])

#     spd_matrix = torch.randint(0, 10, (num_nodes, num_nodes))
#     k_eigen_vectors_pe = torch.randn(num_nodes, 32)

#     from src.config import Config
#     config = Config()

#     model = LocalGlobalTransformer(input_feature_dim, config)
#     out = model(x, k_eigen_vectors_pe, spd_matrix, edge_list)

#     print(out.shape)