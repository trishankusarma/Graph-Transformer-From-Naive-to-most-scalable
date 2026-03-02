# Motivation :
#  -- training loop
import sys
import torch
import torch.nn.functional as F

from .dataset import load_data_and_evaluate_pe
from model.graph_transformer import GraphTransformer
from .config import Config
config = Config()

def evaluate(model, x, y, k_eigen_vectors_pe, spd_matrix, test_mask):
    model.eval()
    
    with torch.no_grad():
        out = model(x = x, k_eigen_vectors_pe = k_eigen_vectors_pe, spd_matrix = spd_matrix) # (num_nodes, num_classes)
        preds = out.argmax(dim = 1) # (num_nodes, 1)

        correct_pred = (preds[test_mask] == y[test_mask]).sum().item()
        total = test_mask.sum().item()

        accuracy = correct_pred/total

    return accuracy

if __name__ == "__main__":

    input_data = sys.argv[1]
    print("Step 1: Loading training dataset from Cora :: building up the positional encodings and concat")
    dataset_info = load_data_and_evaluate_pe(input_data)

    x, y, lap_pe, spd_matrix, train_mask, test_mask = (
        dataset_info['x'], 
        dataset_info['y'], 
        dataset_info['lap_pe'], 
        dataset_info['spd_matrix'],
        dataset_info['train_mask'], 
        dataset_info['test_mask']
    )

    print("Step 2: Initiating the Graph transformer model")
    model = GraphTransformer(input_feature_dim = x.shape[1], config = config)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    print("Step 3: Starting the training phase")
    for epoch in range(config.epochs):
        
        model.train()
        optimizer.zero_grad()

        # Forward pass
        out = model(x = x, k_eigen_vectors_pe = lap_pe, spd_matrix = spd_matrix)
        # loss
        loss = F.nll_loss(out[train_mask], y[train_mask])
        # backward
        loss.backward()
        # step
        optimizer.step()

        if (epoch+1) % 10 == 0:
            acc = evaluate(
                model = model,
                x = x, 
                y = y,
                k_eigen_vectors_pe = lap_pe, 
                spd_matrix = spd_matrix,
                test_mask = test_mask
            )
            print(f"Epoch {epoch+1}/{config.epochs} :: Loss = {loss.item():.4f} :: test acc = {acc:.4f}")