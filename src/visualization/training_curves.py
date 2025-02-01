import matplotlib.pyplot as plt

def plot_training_curves(train_losses, valid_losses):
    """
    Plot training and validation loss curves over time.
    """
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(valid_losses, label="Validation Loss")
    plt.xlabel("Iterations / Steps")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss Curves")
    plt.legend()
    plt.show()
