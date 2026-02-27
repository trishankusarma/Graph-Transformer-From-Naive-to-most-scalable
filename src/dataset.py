# Motivation for this file :
# 1. Load Cora using torch_geometric or raw files
# 2. Compute LapPE — normalized laplacian → eigenvectors → take k smallest non-trivial
# 3. Compute SPD(shortest path distance) matrix
#  — BFS from every node → store as [N, N] integer tensor, cap at max_dist, disconnected nodes get max_dist + 1

# Returns everything the model needs: features, labels, adj, lap_pe, spd_matrix, train_mask, val_mask, test_mask
def load_data(train_dir):
    print(f"Loading data from {train_dir}")

    from torch_geometric.datasets import Planetoid

    dataset = Planetoid(root=train_dir, name='Cora')
    data = dataset[0]
