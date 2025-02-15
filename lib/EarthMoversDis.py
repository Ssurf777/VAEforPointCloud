import torch
import ot  # Python Optimal Transport library

# Earth Mover's Distance (EMD) calculation using POT
def emd_distance(x, y):
    """
    Computes the Earth Mover's Distance (EMD) between two point clouds.
    Args:
        x (Tensor): Original point cloud of shape (N, 3)
        y (Tensor): Reconstructed point cloud of shape (M, 3)
    Returns:
        float: The Earth Mover's Distance between the two point clouds.
    """
    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()

    # Compute cost matrix using squared Euclidean distance
    cost_matrix = ot.dist(x_np, y_np, metric='euclidean')

    # Assume uniform weights for samples
    a = torch.ones(x.shape[0]) / x.shape[0]
    b = torch.ones(y.shape[0]) / y.shape[0]

    # Compute EMD using POT
    emd = ot.emd2(a.numpy(), b.numpy(), cost_matrix)
    return emd