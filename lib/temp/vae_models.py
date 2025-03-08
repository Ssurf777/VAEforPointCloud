import torch
import torch.nn as nn
import torch.nn.functional as F

class standVAE(nn.Module):
    def __init__(self, n_in_out, n_z):
        super(standVAE, self).__init__()

        # PointNet Encoder
        self.conv1 = nn.Conv1d(n_in_out, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 9)
        self.enc_mu = nn.Linear(9, n_z)         # 平均 (mu)
        self.enc_logvar = nn.Linear(9, n_z)    # 対数分散 (logvar)

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
        return y, z, mu, logvar

    def encode(self, x):
        device = x.device
        x = x.reshape(1, -1, 1).to(device)  # Reshape to match the input dimension
        x = F.leaky_relu(self.conv1(x), negative_slope=0.001)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.001)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.001)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)  # Flatten for linear layers
        x = F.leaky_relu(self.linear1(x), negative_slope=0.001)
        x = F.leaky_relu(self.linear2(x), negative_slope=0.001)
        x = self.linear3(x)

        mu = self.enc_mu(x)              # 平均 (mu) の計算
        logvar = self.enc_logvar(x)      # 対数分散 (logvar) の計算
        std = torch.exp(0.5 * logvar)    # 標準偏差 (std)

        # Reparameterization trick for sampling z
        eps = torch.randn_like(std).to(device)
        z = mu + std * eps

        return z, mu, logvar

    def decode(self, z):
        device = z.device  # デバイスを取得
        z = z.squeeze(0)
        z = torch.cat([z], dim=-1)
        x = F.leaky_relu(self.dec1(z), negative_slope=0.001)
        x = F.leaky_relu(self.dec2(x), negative_slope=0.001)
        x = x.unsqueeze(0).unsqueeze(2)
        x = F.leaky_relu(self.dconv1(x), negative_slope=0.001)
        x = F.leaky_relu(self.dconv2(x), negative_slope=0.001)
        x = x.squeeze(0).squeeze(-1)
        x = self.dec_out(x)  # 最終的な出力層
        return x

    def loss(self, y, x, mu, logvar):
        device = y.device  # デバイスを取得
        x_reordered = y.view(3, -1).transpose(0, 1)
        rec_loss = F.mse_loss(y, x, reduction="sum")  # 再構成誤差をMSEで計算
        reg_loss = 0.5 * torch.sum(mu**2 + torch.exp(logvar) - logvar - 1)  # 正則化項
        return rec_loss, reg_loss

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
