import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ChamferDis import chamfer_distance  # External module for Chamfer distance calculation

class MoGVAE(nn.Module):
    def __init__(self, num_points=5000, n_z=3, n_components=2):
        """
        version 2025.02.24 (B x Np x 3) input
        Mixture of Gaussians Variational Autoencoder (MoGVAE).
        Args:
            num_points (int): Number of points in the input/output point cloud
            n_z (int): Dimensionality of the latent variable
            n_components (int): Number of components in the Gaussian mixture
        """
        super(MoGVAE, self).__init__()
        self.n_components = n_components
        self.num_points = num_points  # Store num_points for later use in decode

        # PointNet Encoder
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 9)
        self.enc_mu = nn.Linear(9, n_z * n_components)      # Dimension for μ multiplied by number of components
        self.enc_logvar = nn.Linear(9, n_z * n_components)   # Dimension for logvar similarly multiplied
        self.enc_pi = nn.Linear(9, n_components)             # Mixture coefficients

        # Decoder
        self.dec1 = nn.Linear(n_z, 1024)
        self.dec2 = nn.Linear(1024, 512)
        # For upsampling with ConvTranspose1d, the following layer takes input as (B, 512, 1)
        self.dconv1 = nn.ConvTranspose1d(512, 1024, kernel_size=1)
        self.dconv2 = nn.ConvTranspose1d(1024, 2048, kernel_size=1)
        # Finally, convert to 5000 points x 3 coordinates via fully connected layer
        self.dec3 = nn.Linear(2048, num_points * 3)

        self._init_weights()

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        y = self.decode(z)
        return y, z

    def encode(self, x):
        device = x.device
        # Reshape input x from (B, num_points, 3) to (B, 3, num_points)
        x = x.permute(0, 2, 1).to(device)
        x = F.leaky_relu(self.conv1(x), negative_slope=0.001)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.001)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.001)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.linear1(x), negative_slope=0.001)
        x = F.leaky_relu(self.linear2(x), negative_slope=0.001)
        x = self.linear3(x)

        # Mixture of Gaussians prior (with batch support + looping over each component)
        # pi: (B, n_components)
        pi = F.softmax(self.enc_pi(x), dim=-1)
        # mu, logvar: (B, n_components, n_z)
        mu = self.enc_mu(x).view(x.size(0), self.n_components, -1).to(device)
        logvar = self.enc_logvar(x).view(x.size(0), self.n_components, -1).to(device)
        std = torch.exp(0.5 * logvar)

        z_samples = []
        # Loop over each Gaussian component
        for i in range(self.n_components):
            # eps: (B, n_z)
            eps = torch.randn_like(std[:, i, :]).to(device)
            # Sample for each mixture component: (B, n_z)
            z_sample = mu[:, i, :] + std[:, i, :] * eps
            z_samples.append(z_sample)
        # z_samples: (B, n_components, n_z)
        z_samples = torch.stack(z_samples, dim=1)
        # Weighted average with mixture coefficients -> (B, n_z)
        z = torch.sum(pi.unsqueeze(-1) * z_samples, dim=1)

        self.mu = mu
        self.logvar = logvar
        return z, mu, logvar

    def decode(self, z):
        device = z.device
        # Assume the shape of z is (B, n_z)
        x = F.leaky_relu(self.dec1(z), negative_slope=0.001)
        x = F.leaky_relu(self.dec2(x), negative_slope=0.001)
        # Reshape to (B, 512, 1) to suit ConvTranspose1d
        x = x.unsqueeze(2)
        x = F.leaky_relu(self.dconv1(x), negative_slope=0.001)
        x = F.leaky_relu(self.dconv2(x), negative_slope=0.001)
        # The output shape is (B, 2048, 1), so adjust dimensions accordingly
        x = x.squeeze(2)
        x = self.dec3(x)
        x = x.view(-1, self.num_points, 3)  # Reshape to (B, num_points, 3)
        return x

    def loss(self, y, x):
        rec_loss = F.mse_loss(y, x, reduction="sum")
        reg_loss = 0.5 * torch.sum(self.mu ** 2 + torch.exp(self.logvar) - self.logvar - 1)
        return rec_loss, reg_loss

    @staticmethod
    def chamfer_distance(x, y):
        # If the shapes of x, y are (N, 3), add a batch dimension
        if x.dim() == 2:
            x = x.view(1, x.size(0), x.size(1))
            y = y.view(1, y.size(0), y.size(1))
        x = x.unsqueeze(1)  # (B, 1, N, 3)
        y = y.unsqueeze(2)  # (B, M, 1, 3)
        dist = torch.norm(x - y, dim=-1)
        min_dist_x_to_y = torch.min(dist, dim=1)[0].mean(dim=1)
        min_dist_y_to_x = torch.min(dist, dim=2)[0].mean(dim=1)
        chamfer_dist = min_dist_x_to_y + min_dist_y_to_x
        return chamfer_dist.mean()

    def loss2(self, y, x):
        batch_size = x.size(0)
        mse_loss = F.mse_loss(y, x, reduction="sum")
        # Reshape for Chamfer distance calculation to (N, 3)
        x_reshaped = x.view(-1, 3)
        y_reshaped = y.view(-1, 3)
        rec_loss = self.chamfer_distance(x_reshaped, y_reshaped)
        reg_loss = 0.5 * torch.sum(self.mu ** 2 + torch.exp(self.logvar) - self.logvar - 1) / batch_size
        return mse_loss, rec_loss, reg_loss

    def _init_weights(self):
        # Perform He initialization to match ReLU-based activation (leaky ReLU)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
