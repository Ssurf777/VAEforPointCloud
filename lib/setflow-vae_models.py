import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads=4, num_inducing_points=16):
        """
        Induced Set Attention Block (ISAB)
        """
        super(ISAB, self).__init__()
        self.inducing_points = nn.Parameter(torch.randn(num_inducing_points, dim_out))
        self.multihead_attn1 = nn.MultiheadAttention(embed_dim=dim_out, num_heads=num_heads)
        self.multihead_attn2 = nn.MultiheadAttention(embed_dim=dim_out, num_heads=num_heads)
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, X):
        X = self.linear(X)
        H, _ = self.multihead_attn1(self.inducing_points.unsqueeze(1), X, X)
        Y, _ = self.multihead_attn2(X, H, H)
        return Y

class PlanarFlow(nn.Module):
    def __init__(self, dim):
        super(PlanarFlow, self).__init__()
        self.u = nn.Parameter(torch.randn(dim))
        self.w = nn.Parameter(torch.randn(dim))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, z):
        activation = torch.tanh(z @ self.w + self.b)
        return z + self.u * activation

class SetVAE(nn.Module):
    def __init__(self, n_in, n_z, num_heads=4, num_inducing_points=16, num_flows=2):
        """
        Set-based Variational Autoencoder (Set-VAE) with ISAB and Normalizing Flow.
        """
        super(SetVAE, self).__init__()
        self.encoder = ISAB(n_in, 128, num_heads, num_inducing_points)
        self.fc_mu = nn.Linear(128, n_z)
        self.fc_logvar = nn.Linear(128, n_z)
        
        # Normalizing Flow layers
        self.flows = nn.ModuleList([PlanarFlow(n_z) for _ in range(num_flows)])
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(n_z, 128),
            nn.ReLU(),
            nn.Linear(128, n_in)
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.mean(dim=1)  # Pooling step to get global representation
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        for flow in self.flows:
            z = flow(z)
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss, kl_div
