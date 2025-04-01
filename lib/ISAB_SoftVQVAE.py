import torch
import torch.nn as nn
import torch.nn.functional as F
from ISAB import ISAB

# --------- Soft Vector Quantizer ---------
class SoftVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, temperature=1.0):
        super(SoftVectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.temperature = temperature

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z):
        z_flattened = z.view(-1, self.embedding_dim)  # (B, D)

        # 距離計算 (B, N)
        distances = (
            (z_flattened ** 2).sum(dim=1, keepdim=True)
            + (self.embedding.weight ** 2).sum(dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )

        # Softmaxで重み計算 (温度あり)
        weights = F.softmax(-distances / self.temperature, dim=1)  # (B, N)

        # 埋め込みの重み付き平均 (B, D)
        z_soft = torch.matmul(weights, self.embedding.weight)

        # 損失：zとsoft embeddingの距離
        commitment_loss = F.mse_loss(z_soft, z.detach())

        return z_soft, commitment_loss


# --------- ISAB + SoftVQ-VAE ---------
class ISAB_SoftVQVAE(nn.Module):
    def __init__(self, num_points=5000, dim_input=3, dim_hidden=128, num_heads=4, num_inds=32,
                 embedding_dim=3, num_embeddings=1024, temperature=1.0):
        super(ISAB_SoftVQVAE, self).__init__()
        self.num_points = num_points
        self.dim_input = dim_input
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        # ISAB Encoder
        self.isab = ISAB(dim_input, dim_hidden, num_heads, num_inds)
        self.fc_enc = nn.Linear(dim_hidden, embedding_dim)

        # Soft Vector Quantization
        self.quantizer = SoftVectorQuantizer(num_embeddings, embedding_dim, temperature)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(embedding_dim, 128, kernel_size=1),
            nn.LeakyReLU(0.001),
            nn.ConvTranspose1d(128, 64, kernel_size=1),
            nn.LeakyReLU(0.001),
            nn.ConvTranspose1d(64, num_points * dim_input, kernel_size=1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x):
        x = x.permute(0, 2, 1)  # (B, N, 3)
        x = self.isab(x)
        x = x.mean(dim=1)  # (B, H)
        z = self.fc_enc(x)  # (B, D)
        z_soft, quantization_loss = self.quantizer(z)
        return z, z_soft, quantization_loss

    def decode(self, z):
        z = z.unsqueeze(-1)  # (B, D, 1)
        x_recon = self.decoder(z)
        x_recon = x_recon.squeeze(-1).view(-1, self.num_points, self.dim_input)
        return x_recon

    def forward(self, x):
        z, z_soft, quantization_loss = self.encode(x)
        x_recon = self.decode(z_soft)
        return x_recon, quantization_loss, z, self.embedding_dim
