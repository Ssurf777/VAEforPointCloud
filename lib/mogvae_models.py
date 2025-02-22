import torch
import torch.nn as nn
import torch.nn.functional as F
from chamfer_distance import ChamferDis

class MoGVAE(nn.Module):
    def __init__(self, n_in_out, n_z, n_components=2):
        """
        Mixture of Gaussians Variational Autoencoder (MoGVAE).
        Args:
            n_in_out (int): The dimension of input/output.
            n_z (int): The dimension of the latent variable.
            n_components (int): Number of Gaussian components in the mixture.
        """
        super(MoGVAE, self).__init__()
        self.n_components = n_components

        # PointNet Encoder
        self.conv1 = nn.Conv1d(n_in_out, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 9)
        self.enc_mu = nn.Linear(9, n_z * n_components)  # mu's dimension is multiplied by the number of components
        self.enc_logvar = nn.Linear(9, n_z * n_components)  # logvar's dimension is also multiplied
        self.enc_pi = nn.Linear(9, n_components)  # Mixture coefficients

        # Decoder
        self.dec1 = nn.Linear(n_z, 1024)
        self.dec2 = nn.Linear(1024, 512)
        self.dconv1 = nn.ConvTranspose1d(512, 1024, kernel_size=1)
        self.dconv2 = nn.ConvTranspose1d(1024, 2048, kernel_size=1)
        self.dec_out = nn.Linear(2048, n_in_out)

        # Weight initialization
        self._init_weights()

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        y = self.decode(z)
        return y, z

    def encode(self, x):
        device = x.device
        x = x.reshape(1, -1, 1).to(device)
        x = F.leaky_relu(self.conv1(x), negative_slope=0.001)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.001)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.001)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.linear1(x), negative_slope=0.001)
        x = F.leaky_relu(self.linear2(x), negative_slope=0.001)
        x = self.linear3(x)

        # Mixture of Gaussians prior
        pi = F.softmax(self.enc_pi(x), dim=-1)  # Softmax for mixture coefficients
        mu = self.enc_mu(x).view(self.n_components, -1).to(device)
        logvar = self.enc_logvar(x).view(self.n_components, -1).to(device)
        std = torch.exp(0.5 * logvar)

        z_samples = []
        for i in range(self.n_components):
            eps = torch.randn_like(std[i]).to(device)
            z_sample = mu[i] + std[i] * eps
            z_samples.append(z_sample)

        z_samples = torch.stack(z_samples).to(device)
        z = torch.sum(pi.view(self.n_components, 1) * z_samples, dim=0)
        self.mu = mu
        self.logvar = logvar
        return z, mu, logvar

    def decode(self, z):
        device = z.device
        z = z.squeeze(0)
        z = torch.cat([z], dim=-1)
        x = F.leaky_relu(self.dec1(z), negative_slope=0.001)
        x = F.leaky_relu(self.dec2(x), negative_slope=0.001)
        x = x.unsqueeze(0).unsqueeze(2)
        x = F.leaky_relu(self.dconv1(x), negative_slope=0.001)
        x = F.leaky_relu(self.dconv2(x), negative_slope=0.001)
        x = x.squeeze(0).squeeze(-1)
        x = self.dec_out(x)
        return x

    def loss(self, y, x):
        device = y.device
        x_reordered = y.view(3, -1).transpose(0, 1)
        rec_loss = F.mse_loss(y, x, reduction="sum")
        reg_loss = 0.5 * torch.sum(self.mu ** 2 + torch.exp(self.logvar) - self.logvar - 1)
        return rec_loss, reg_loss
        
    def loss2(self, y, x):
        """
        Compute loss2: MSE, Chamfer Distance, and KL Divergence.
        Args:
          y (Tensor): Reconstructed output (B, N, 3)
          x (Tensor): Original input (B, N, 3)
        Returns:
          Tuple: mse_loss, chamfer_loss, kl_loss
        """
        device = y.device
        # MSE Loss
        mse_loss = F.mse_loss(y, x, reduction="mean")
        # Chamfer Distance Loss
        chamfer_loss = chamfer_distance(y, x)
        # KL Divergence Loss 
        kl_div =  0.5 * torch.sum(self.mu ** 2 + torch.exp(self.logvar) - self.logvar - 1)
        # Average over batch
        kl_loss = kl_div.mean()
        return mse_loss, chamfer_loss, kl_loss

    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
