import torch
import torch.nn as nn
import torch.nn.functional as F
from ISAB import ISAB  # 提供いただいたISABモジュールを使用

class SetVAE(nn.Module):
    def __init__(self, num_points=5000, n_z=3, dim_input=3, dim_hidden=128, num_heads=4, num_inds=32):
        super(SetVAE, self).__init__()
        self.num_points = num_points
        self.dim_input = dim_input
        
        # ISAB encoder
        self.isab = ISAB(dim_input, dim_hidden, num_heads, num_inds)

        # VAE latent variables
        self.linear_mu = nn.Linear(dim_hidden, n_z)
        self.linear_logvar = nn.Linear(dim_hidden, n_z)

        # Decoder
        self.dec_linear1 = nn.Linear(n_z, 1024)
        self.dec_linear2 = nn.Linear(1024, num_points * dim_input)

    def encode(self, x):
        # x: (B, num_points, dim_input)
        device = x.device
        x = self.isab(x)
        x = x.mean(dim=1)  # 集合の特徴を平均で集約

        mu = self.linear_mu(x)
        logvar = self.linear_logvar(x)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(device)
        z = mu + eps * std

        return z, mu, logvar

    def decode(self, z):
        x = F.leaky_relu(self.dec_linear1(z), negative_slope=0.001)
        x = self.dec_linear2(x)
        x = x.view(-1, self.num_points, self.dim_input)
        return x

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
