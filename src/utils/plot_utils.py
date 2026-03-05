import matplotlib.pyplot as plt
import os

def plot_curves(training_loss = None, val_accuracy = None, test_accuracy = None, epochs = 200, skip_accuracy_freq = 10, approach = 1, plot_dir = './plots'):

    plot_dir = f'{plot_dir}/approach{approach}'
    # Create directory if not exists
    os.makedirs(plot_dir, exist_ok=True)

    # Plot 1: Training Loss
    if training_loss is not None and len(training_loss) > 0:
        epochs_list = range(1, len(training_loss) + 1)
        
        plt.figure()
        plt.plot(epochs_list, training_loss)
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss")
        plt.title("Training Loss vs Epochs")
        plt.savefig(os.path.join(plot_dir, "training_loss.png"))
        plt.close()

    # Plot 2: Val and Test Accuracy
    if( val_accuracy is not None and len(val_accuracy) > 0 and
        test_accuracy is not None and len(test_accuracy) > 0):

        acc_epochs = [skip_accuracy_freq * (i + 1) for i in range(len(val_accuracy))]

        plt.figure()
        plt.plot(acc_epochs, val_accuracy)
        plt.plot(acc_epochs, test_accuracy)
        plt.xlabel("Epoch")
        plt.ylabel("Validation and Test Accuracy")
        plt.legend(["Validation", "Test"])
        plt.title("Test Accuracy vs Epochs")
        plt.savefig(os.path.join(plot_dir, "val_test_accuracy.png"))
        plt.close()
    
    print(f"Plots saved in directory: {plot_dir}")
