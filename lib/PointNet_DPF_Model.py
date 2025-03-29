import torch
import torch.nn as nn
import torch.nn.functional as F

# PointNet Encoder (Conv1d + MaxPool)
class PointNetEncoder(nn.Module):
    def __init__(self, input_dim=3, latent_dim=128):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, latent_dim, 1)

    def forward(self, x):  # x: (B, 3, N)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = torch.max(x, dim=2)[0]  # Global max pooling
        return x  # (B, latent_dim)

# DPF Block (1 Flow Layer)
class DPFBlock(nn.Module):
    def __init__(self, latent_dim, num_points=1024, num_bins=64):
        super().__init__()
        self.num_points = num_points
        self.num_bins = num_bins
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_points * 3 * num_bins)
        )

    def forward(self, z):
        B = z.size(0)
        logits = self.linear(z).view(B, self.num_points, 3, self.num_bins)
        probs = F.softmax(logits, dim=-1)
        bins = torch.linspace(-1, 1, self.num_bins, device=z.device)
        coords = torch.sum(probs * bins[None, None, None, :], dim=-1)
        return coords  # (B, num_points, 3)

# Stacked DPF Decoder (7 Flow Layers)
class StackedDPFDecoder(nn.Module):
    def __init__(self, latent_dim=128, num_flows=7, num_points=1024, num_bins=64):
        super().__init__()
        self.flow_blocks = nn.ModuleList([
            DPFBlock(latent_dim, num_points, num_bins) for _ in range(num_flows)
        ])

    def forward(self, z):
        # 同じ潜在変数 z をすべての Flow に渡す
        x = None
        for flow in self.flow_blocks:
            x = flow(z)
        return x



# Chamfer Distance
def chamfer_distance(p1, p2):
    """
    p1, p2: (B, N, 3)
    """
    dist = torch.cdist(p1, p2, p=2)
    min1 = torch.min(dist, dim=2)[0]
    min2 = torch.min(dist, dim=1)[0]
    return torch.mean(min1) + torch.mean(min2)

# 全体統合モデル
class PointNet_DPF_Model(nn.Module):
    def __init__(self, input_dim=3, latent_dim=128, num_points=1024, num_bins=64, num_flows=7):
        super().__init__()
        self.encoder = PointNetEncoder(input_dim=input_dim, latent_dim=latent_dim)
        self.decoder = StackedDPFDecoder(latent_dim=latent_dim, num_flows=num_flows,
                                         num_points=num_points, num_bins=num_bins)

    def forward(self, x):  # x: (B, 3, N)
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

    def compute_loss(self, x_input, x_target):
        """
        x_input: (B, 3, N) - 入力点群
        x_target: (B, M, 3) - 教師信号点群
        """
        x_recon = self.forward(x_input)   # (B, M, 3)
        loss = chamfer_distance(x_recon, x_target)
        return loss, x_recon
        
    def decode(self, z):
        return self.decoder(z)
