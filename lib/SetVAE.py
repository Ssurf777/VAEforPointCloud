import torch
import torch.nn as nn
import torch.nn.functional as F
from ISAB import ISAB

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

        # 重みの初期化
        self._init_weights()

    def encode(self, x):
        x = self.isab(x)
        x = x.mean(dim=1)

        mu = self.linear_mu(x)
        logvar = self.linear_logvar(x)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return z, mu, logvar

    def decode(self, z):
        x = F.leaky_relu(self.dec_linear1(z), negative_slope=0.001)
        x = self.dec_linear2(x)
        x = x.view(-1, self.num_points, self.dim_input)
        return x

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        y = self.decode(z)
        return y, z, mu, logvar

    def loss(self, y, x, mu, logvar):
        rec_loss = F.mse_loss(y, x, reduction="sum")
        reg_loss = 0.5 * torch.sum(mu**2 + torch.exp(logvar) - logvar - 1)
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

    def loss2(self, y, x, mu, logvar):
        batch_size = x.size(0)
        mse_loss = F.mse_loss(y, x, reduction="sum")
        # Reshape for Chamfer distance calculation to (N, 3)
        x_reshaped = x.view(-1, 3)
        y_reshaped = y.view(-1, 3)
        rec_loss = self.chamfer_distance(x_reshaped, y_reshaped)
        reg_loss = 0.5 * torch.sum(self.mu ** 2 + torch.exp(self.logvar) - self.logvar - 1) / batch_size
        return mse_loss, rec_loss, reg_loss
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
