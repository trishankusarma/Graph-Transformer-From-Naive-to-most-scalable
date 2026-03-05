# Motivation
# GAT attention over neighbors
import torch
import torch.nn as nn
import torch.nn.functional as F

class GAT_Layer(nn.Module):
    # Steps
    # 1. Transform x -> x @ W.T
    # 2. For each edge -> e_ij = dot product of learned matrix a_first_half and W_x_i + dot product of a_second_half and W_x_j  
    # 3. Apply leaky relu on it 
    # 4. softmax(e_ij, dim = -1) over the neighbors and get the scores
    # 5. x[node_number] = sum(score[j] * x[j]) over all neighbors
    def __init__(self, d_model, leaky_relu_slope, dropout):
        super().__init__()
        self.W = nn.Linear(d_model, d_model, bias = False)
        self.a = nn.Parameter(torch.randn(2*d_model))

        self.leaky_relu_slope = leaky_relu_slope
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_list, **kwargs):
        # x(num_nodes, d_model)
        num_nodes, d_model = x.shape[0], x.shape[1]
        # Step 1
        x = self.W(x)
        
        a_src = self.a[:d_model] # (d_model,)
        a_dst = self.a[d_model:] # (d_model,)

        scores_src = x @ a_src # stack wise dot product gives (n_nodes) -> (num_nodes,)
        scores_dst = x @ a_dst # (num_nodes,)

        # let's separate source and destination 
        # Step 2
        src, dest = map(torch.tensor, zip(*edge_list)) # (num_edges,)
        # Adding self loops
        src_self_loop_nodes = torch.arange(num_nodes)
        src = torch.cat([src, src_self_loop_nodes])
        dest = torch.cat([dest, src_self_loop_nodes])

        scores = scores_src[src] + scores_dst[dest] # (num_edges)

        # Step 3
        scores = F.leaky_relu(scores, negative_slope = self.leaky_relu_slope) # (num_edges)

        # Step 4 :: using softmax per node over its neighbors 
        # For numerical stability -- subtracting the max value
        score_max = torch.zeros(num_nodes).scatter_reduce(
            0, src, scores, reduce = "amax", include_self = True
        )
        e_exp = torch.exp(scores - score_max[src]) # (num_edges)
        e_sum = torch.zeros(num_nodes).scatter_add(0, src, e_exp) # (num_nodes)

        alpha = e_exp/(e_sum[src] + 1e-9) # (num_edges)
        alpha = self.dropout(alpha) # (num_edges)

        # Step 5
        new_x = torch.zeros_like(x) # (num_nodes, d_model)
        new_x.scatter_add_(
            0,
            src.unsqueeze(1).expand(-1, d_model), # (num_edges, 1)
            alpha.unsqueeze(1) * x[dest] # (num_edges, 1) * (num_edges, d_model) = (num_edges, d_model) 
        )
        return new_x