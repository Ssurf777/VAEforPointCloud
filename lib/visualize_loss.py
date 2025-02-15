import matplotlib.pyplot as plt

def visualize_loss(rec_error, reg_error, total_error):
    """
    Visualizes the loss over training epochs.
    Args:
        rec_error (list): List of reconstruction losses.
        reg_error (list): List of regularization losses.
        total_error (list): List of total losses.
    """
    plt.plot(range(1, len(rec_error) + 1), rec_error, label="Rec")
    plt.plot(range(1, len(reg_error) + 1), reg_error, label="Reg")
    plt.plot(range(1, len(total_error) + 1), total_error, label="Total")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.ylim(0, 1000)
    plt.show()
