import torch

# Chamfer Distance calculation
def chamfer_distance(x, y):
    """
    Custom function to compute Chamfer Distance.
    Args:
        x (Tensor): Original point cloud of shape (B, N, 3)
        y (Tensor): Reconstructed point cloud of shape (B, M, 3)
    Returns:
        Tensor: Batch mean Chamfer Distance.
    """
    x = x.unsqueeze(1)  # (B, 1, N, 3)
    y = y.unsqueeze(2)  # (B, M, 1, 3)

    # Compute pairwise distances
    dist = torch.norm(x - y, dim=-1)  # (B, M, N)

    # Compute minimum distances from x to y and y to x, then average
    min_dist_x_to_y = torch.min(dist, dim=1)[0].mean(dim=1)  # (B,)
    min_dist_y_to_x = torch.min(dist, dim=2)[0].mean(dim=1)  # (B,)

    chamfer_dist = min_dist_x_to_y + min_dist_y_to_x  # Sum bidirectional distances
    return chamfer_dist.mean()  # Return batch mean Chamfer distance
