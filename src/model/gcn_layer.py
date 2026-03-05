import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN_Layer(nn.Module):
    # Step 1: Add self loops
    # Step 2: Normalize
    # Step 3: Aggregation and Transformation
    # Step 4: Apply ReLU activation function, apply dropout and return the output
    def __init__(self, d_model, dropout):
        super().__init__()

        self.W = nn.Linear(d_model, d_model, bias = False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Adj_matrix, x):
        # Step 1
        num_nodes = x.shape[0]
        I = torch.eye(num_nodes) #(num_nodes, num_nodes)

        Adj_matrix = Adj_matrix +  I #(num_nodes, num_nodes)
        degree = Adj_matrix.sum(dim=1) #(num_nodes)
        
        # Step 2
        degree_inv_root = torch.pow(degree, -0.5) #(num_nodes)

        # Using boardcast to find Adj_matrix_norm
        # (num_nodes, 1)*(num_nodes, num_nodes)*(1, num_nodes)
        Adj_matrix_norm = degree_inv_root.unsqueeze(1) * Adj_matrix * degree_inv_root.unsqueeze(0)

        # Step 3
        # (num_nodes, num_nodes) @ (num_nodes, d_model) @ (d_model, d_model)
        Z = self.W(Adj_matrix_norm @ x)

        # Step 4
        Z = F.relu(Z)
        Z = self.dropout(Z)
        return Z
