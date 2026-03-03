# Motivation :
#  -- training loop
import sys
import torch
import torch.nn.functional as F
import os

from .dataset import load_data_and_evaluate_pe
from src.model.graph_transformer import GraphTransformer
from .config import Config
from src.utils.plot_utils import plot_curves
config = Config()

torch.set_num_threads(config.no_of_cpu_threads)
torch.set_num_interop_threads(config.no_of_cpu_threads)

PLOT_DIR = './models'

torch.manual_seed(config.seed)
os.makedirs(PLOT_DIR, exist_ok=True)

def evaluate(model, x, y, k_eigen_vectors_pe, spd_matrix, mask):
    model.eval()
    
    with torch.no_grad():
        out = model(x = x, k_eigen_vectors_pe = k_eigen_vectors_pe, spd_matrix = spd_matrix, config = config) # (num_nodes, num_classes)
        preds = out.argmax(dim = -1) # (num_nodes, 1)

        correct_pred = (preds[mask] == y[mask]).sum().item()
        total = mask.sum().item()

        accuracy = correct_pred/total

    return accuracy

if __name__ == "__main__":

    input_data = sys.argv[1]
    print("Step 1: Loading training dataset from Cora :: building up the positional encodings and concat")
    dataset_info = load_data_and_evaluate_pe(input_data)

    x, y, lap_pe, spd_matrix, train_mask, val_mask, test_mask = (
        dataset_info['x'], 
        dataset_info['y'], 
        dataset_info['lap_pe'], 
        dataset_info['spd_matrix'],
        dataset_info['train_mask'],
        dataset_info['val_mask'],
        dataset_info['test_mask']
    )

    print("Step 2: Initiating the Graph transformer model")
    model = GraphTransformer(input_feature_dim = x.shape[1], config = config)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.epochs,
        eta_min=config.eta_min
    )

    training_loss = []
    val_accuracy = []
    test_accuracy = []

    best_val_acc  = 0
    best_test_acc = 0
    patience_counter = 0

    print("Step 3: Starting the training phase")
    for epoch in range(config.epochs):
        
        model.train()
        optimizer.zero_grad()

        # Forward pass
        out = model(x = x, k_eigen_vectors_pe = lap_pe, spd_matrix = spd_matrix, config = config)
        # loss
        loss = F.nll_loss(out[train_mask], y[train_mask])
        # backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_gradient_clip)
        # step
        optimizer.step()
        scheduler.step()

        training_loss.append(loss.item())
        if (epoch+1) % config.skip_accuracy_freq == 0:
            val_acc = evaluate(
                model = model,
                x = x, 
                y = y,
                k_eigen_vectors_pe = lap_pe, 
                spd_matrix = spd_matrix,
                mask = val_mask
            )
            test_acc = evaluate(
                model = model,
                x = x, 
                y = y,
                k_eigen_vectors_pe = lap_pe, 
                spd_matrix = spd_matrix,
                mask = test_mask
            )
            print(f"Epoch {epoch+1}/{config.epochs} :: Loss = {loss.item():.4f} :: val acc = {val_acc:.4f} :: test acc = {test_acc:.4f}")
            val_accuracy.append(val_acc)
            test_accuracy.append(test_acc)

            if val_acc > best_val_acc:
                best_val_acc  = val_acc
                best_test_acc = test_acc
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(PLOT_DIR, './best_model.pt'))
                print("Saved the best model")
            else:
                patience_counter += 1

            if patience_counter >= config.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    plot_curves(
        training_loss = training_loss, 
        val_accuracy = val_accuracy, 
        test_accuracy = test_accuracy, 
        epochs = config.epochs,
        skip_accuracy_freq = config.skip_accuracy_freq
    )
