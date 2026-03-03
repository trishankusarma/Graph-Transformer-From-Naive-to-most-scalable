# Motivation :
# -- full model (embedding + stack of layers + classifier)
import torch.nn as nn
import torch
import torch.nn.functional as F

from .transformer_layer import GraphTransformerLayer

class GraphTransformer(nn.Module):
    # this class does 4 things
    # 1. Transform the input(num_nodes, features+k_lap_pe) to (num_nodes, d_model)
    # 2. Pass on the transformed_input over L layers
    # 3. Layer normalization over the final layer
    # 4. Softmax classification layer 
    def __init__(self, input_feature_dim, config):
        super().__init__()

        # Step 1.1
        self.input_projection = nn.Linear(input_feature_dim + config.k_lap_pe, config.d_model)

        # Step 1.2
        self.layers = nn.ModuleList([GraphTransformerLayer(
            config.d_model, config.dropout, config.num_heads, config.max_dist, config.d_ff
        ) for _ in range(config.num_layers)])

        # Step 1.3
        self.final_norm = nn.LayerNorm(config.d_model)

        # Step 1.4
        self.classifier = nn.Linear(config.d_model, config.d_classes)
    
    def forward(self, x, k_eigen_vectors_pe, spd_matrix, config, **kwargs):
        # Step 2.1 :: concat the node features with the positional encoding
        x = torch.cat([x, k_eigen_vectors_pe], dim = 1) # (num_nodes, feature_dim + k_lap_pe)
        # This is something I add to control over-fitting
        x = F.dropout(x, p=config.dropout_input, training=self.training) # adding drop out for more generelizability

        # Step 2.2 :: project to d_model
        x = self.input_projection(x) # (num_nodes, d_model)

        # Step 3 :: pass among num_layers of attention + ffn
        for layer in self.layers:
            x = layer(x, spd_matrix) # (num_nodes, d_model)
        
        # Step 4 :: layer normalize to protect the arbitary scaling
        x = self.final_norm(x)

        # Step 5 :: log-Softmax classification
        x = F.log_softmax(self.classifier(x), dim = -1) # (num_nodes, d_classes) -> applying softmax to last layer over all classes

        return x
    
# if __name__ == "__main__":
#     import torch
#     from src.config import Config
#     config = Config()

#     x = torch.randn(2708, 1433)
#     k_eigen_vectors_pe = torch.randn(2708, config.k_lap_pe)
#     spd_matrix = torch.randint(0, config.max_dist, (2708, 2708))

#     model = GraphTransformer(x.shape[1], config)
#     out = model(x, k_eigen_vectors_pe, spd_matrix)
#     print(out.shape)