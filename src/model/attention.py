# Motivation :
# -- multi-head global attention + SPD bias
import torch.nn as nn
import torch.nn.functional as F
import torch

class GraphTransformerAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        d_model, dropout, num_heads, max_dist, max_degree = config.d_model, config.dropout, config.num_heads, config.max_dist, config.max_degree
        # Step 1
        # all parameters goes here
        # Weights corresponding to Q, K, V, O
        self.W_Q = nn.Linear(d_model, d_model, bias = False)
        self.W_K = nn.Linear(d_model, d_model, bias = False)
        self.W_V = nn.Linear(d_model, d_model, bias = False)
        
        # Once we do have the attention weight and multiply with Output
        self.W_O = nn.Linear(d_model, d_model, bias = False)
        
        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads

        # Lookup table for SPD Matrix 
        self.spd_bias = nn.Embedding(max_dist+2, num_heads)
        # max_dist+2 buckets: 0,1,...,max_dist and max_dist+1 (unreachable)
        self.d_k = d_model // num_heads # dimensions per head

        # adding degree encoding for approach = 4 :: a look up table for degree encoding
        self.max_degree = max_degree
        self.degree_encoding_src = nn.Embedding(max_degree, num_heads)
        self.degree_encoding_dst = nn.Embedding(max_degree, num_heads)
    
    def forward(self, x, spd_matrix):
        num_nodes, d_model = x.shape[0], x.shape[1]
        
        # Step 2 : compute Q, K, V here
        Q = self.W_Q(x) # x@W_Q.T #(n_nodes, d_model)
        K = self.W_K(x) # x@W_K.T #(n_nodes, d_model)
        V = self.W_V(x) # x@W_V.T #(n_nodes, d_model)

        # Step 2.1 : Reshape to num_heads
        Q = Q.view(num_nodes, self.num_heads, self.d_k).transpose(0,1) # (num_heads, num_nodes, d_k)
        K = K.view(num_nodes, self.num_heads, self.d_k).transpose(0,1) # (num_heads, num_nodes, d_k)
        V = V.view(num_nodes, self.num_heads, self.d_k).transpose(0,1) # (num_heads, num_nodes, d_k)

        # Step 3 :: Calculate the logit scores based on node features
        scores = Q @ K.transpose(-2, -1) # (num_heads, num_nodes, d_k) @ (num_heads, d_k, num_nodes)
        # scores = scores / (self.d_k ** 0.5*T) # (num_heads, num_nodes, num_nodes)
        scores = scores / (self.d_k ** 0.5) # (√d_k)
        # (num_heads, num_nodes, num_nodes) :: scores scaled by 1/(d_k^0.5 * 0.1) → 10x sharper

        # Step 4 :: Distance Bias calculation
        bias = self.spd_bias(spd_matrix).permute(2, 0, 1) # (num_nodes, num_nodes, num_heads) -> (num_heads, num_nodes, num_nodes)
        scores = scores + bias

        # Add degree encoding + spatial encoding directly into attention
        # Compute degree from SPD matrix — nodes at distance 1 are neighbors
        degree = (spd_matrix == 1).sum(dim=-1).clamp(0, self.max_degree - 1)  # (num_nodes) long tensor
        # degree_bias
        degree_src_encoding = self.degree_encoding_src(degree) # (num_nodes, num_heads)
        degree_dst_encoding = self.degree_encoding_dst(degree) # (num_nodes, num_heads)

        degree_bias = degree_src_encoding.T.unsqueeze(2) + degree_dst_encoding.T.unsqueeze(1)
                    # (num_heads, num_nodes, 1)          (num_heads, 1, num_nodes)
        scores = scores + degree_bias #(num_heads, num_nodes, num_nodes)
        
        # Step 5 :: Calculating the softmax over all the keys for each query
        attention_weight = F.softmax(scores, dim = -1) # (num_heads, num_nodes, num_nodes)
        # Step 5.1 :: Apply dropout
        attention_weight = self.dropout(attention_weight)

        # Step 6 :: Multiply with V matrix to finally find the attention
        attention = attention_weight @ V # (num_heads, num_nodes, num_nodes) * (num_heads, num_nodes, d_k) 
                                         # = (num_heads, num_nodes, d_k)

        # Step 7 :: Reshape to (num_nodes, d_model) 
        attention = attention.transpose(0,1) # (num_nodes, num_heads, d_k)
        attention = attention.contiguous().view(num_nodes, d_model) # (num_nodes, d_model)

        # Step 8 :: Finally multiply with W_O
        attention = self.W_O(attention) # attention @ W_O.T ||| (num_nodes, d_model) @ (d_model, d_model)

        return attention