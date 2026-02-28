# Motivation for this file :
# 1. Load Cora using torch_geometric or raw files
# 2. Compute LapPE — normalized laplacian → eigenvectors → take k smallest non-trivial
# 3. Compute SPD(shortest path distance) matrix
#  — BFS from every node → store as [N, N] integer tensor, cap at max_dist, disconnected nodes get max_dist + 1

# Returns everything the model needs: features, labels, adj, lap_pe, spd_matrix, train_mask, val_mask, test_mask

from torch_geometric.datasets import Planetoid
import torch
from collections import deque
from config import Config
config = Config()

def compute_normalized_laplacian_pe(adj_matrix, degree, num_nodes):#  = I - D^(-0.5)AD^(-0.5) 
    I = torch.eye(num_nodes)
    
    degree_root_inv = torch.pow(degree, -0.5)
    # using boardcasting instead of using the regular diagonal matrix
    adj_matrix_norm = degree_root_inv.unsqueeze(1) * adj_matrix * degree_root_inv.unsqueeze(0)
    
    return I - adj_matrix_norm

def spectral_decomposition(L, k=None): 
    # Compute all eigenvalues and eigenvectors
    eigenvalues, eigenvectors = torch.linalg.eigh(L)

    if k is not None:
        # Take k smallest eigenvectors
        eigenvectors = eigenvectors[:, 1:(k+1)]
        eigenvalues = eigenvalues[1:(k+1)]

    return eigenvalues, eigenvectors

def compute_smallest_distance_matrix(adj_list, num_nodes, start_node):
    # regular bfs code for covering the distance from the start_node
    distance = [-1] * num_nodes
    distance[start_node] = 0

    queue = deque([start_node])

    while queue:
        node = queue.popleft()
        
        for neighbor in adj_list[node]:
            if distance[neighbor] == -1:
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)
                
    return distance

def load_data(train_dir):
    # step 1
    print(f"Loading data from {train_dir}")
    dataset = Planetoid(root=train_dir, name='Cora')
    data = dataset[0]

    # a brief on the data
    num_nodes = data.num_nodes
    num_edges = data.num_edges
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {num_edges}")
    print(f"Node feature shape: {data.x.shape}")
    print(f"Labels shape: {data.y.shape}")
    print(f"Edge_index shape {data.edge_index.shape}")

    # step 2
    print("Computing the positional encoding of the data")
    # step 2.1 :: compute the adjacency matrix
    src, dst = data.edge_index[0], data.edge_index[1]
    adj_matrix = torch.zeros(num_nodes, num_nodes)
    adj_matrix[src, dst] = 1.0

    print(f"Adjacency matrix shape: {adj_matrix.shape}")
    print(f"Non zero entries: {adj_matrix.sum().int()}")

    # step 2.2 :: compute the adjacency list
    adj_list = {i : [] for i in range(num_nodes)}
    for s, d in zip(src.tolist(), dst.tolist()):
        adj_list[s].append(d)

    # step 2.3 :: compute the degree matrix
    degree = adj_matrix.sum(dim=1).clamp(min=1e-10) # [num_nodes] row wise sum and clamping to avoid division by 0
    print(f"Degree stats → min: {degree.min()}, max: {degree.max()}, mean: {degree.mean():.2f}")

    # step 2.4 :: compute the normalized laplacian PE
    L = compute_normalized_laplacian_pe(adj_matrix, degree, num_nodes)
    
    # step 2.5 :: compute its eigen vectors and find the top k smallest non-trivial vectors as the positional embedding
    eigenvalues, eigenvectors = spectral_decomposition(L, k = config.k_lap_pe)

    # step 3
    print("Computing the Smallest Distance Matrix")
    # step 3.1 :: store as [N, N] integer tensor
    distance_matrix = []
    for start_node in range(num_nodes):
        distance_matrix.append(compute_smallest_distance_matrix(adj_list, num_nodes, start_node))

    distance_matrix = torch.tensor(distance_matrix, dtype=torch.long)
    distance_matrix[distance_matrix == -1] = config.max_dist + 1  # unreachable nodes
    distance_matrix = distance_matrix.clamp(0, config.max_dist)

    print(f"LapPE shape: {eigenvectors.shape}")        # should be [2708, 32]
    print(f"SPD matrix shape: {distance_matrix.shape}") # should be [2708, 2708]
    print(f"SPD unique values: {distance_matrix.unique()}")  # should be 0,1,2,...,10,11

    # add this
    return {
        'x'          : data.x,
        'y'          : data.y,
        'lap_pe'     : eigenvectors,
        'spd_matrix' : distance_matrix,
        'train_mask' : data.train_mask,
        'val_mask'   : data.val_mask,
        'test_mask'  : data.test_mask,
    }
    