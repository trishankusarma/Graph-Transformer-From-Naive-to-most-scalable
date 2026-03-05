# Motivation
# one layer combining GAT + GT + gate


# if __name__ == "__main__":
#     import torch
#     num_nodes = 20
#     embed_dim = 32
#     x = torch.randn(num_nodes, embed_dim)
    
#     edge_list = []
#     p = 0.4
#     for i in range(num_nodes):
#         for j in range(i+1, num_nodes):
#             import random
#             if random.random() < p:
#                 edge_list.append([i, j])
#                 edge_list.append([j, i])

#     d_model = embed_dim
#     gat_layer = GAT_Layer(d_model, 5, 0.3)
#     out = gat_layer(x, edge_list)

#     print(out.shape)