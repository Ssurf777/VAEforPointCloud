import torch
import torch.nn as nn
import torch.nn.functional as F
from ISAB import ISAB

class ISAB_VQVAE(nn.Module):
    def __init__(self, num_points=5000, dim_input=3, dim_hidden=128, num_heads=4, num_inds=32,
                 embedding_dim=3, num_embeddings=1024):
        super(ISAB_VQVAE, self).__init__()
        self.num_points = num_points
        self.dim_input = dim_input
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        # ISAB Encoder
        self.isab = ISAB(dim_input, dim_hidden, num_heads, num_inds)
        self.fc_enc = nn.Linear(dim_hidden, embedding_dim)
        
        # Vector Quantization
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim)

        # Decoder (Transposed Convolutions)
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
        # print(f"Input x shape: {x.shape}")  # 確認
        # print("permute")
        x = x.permute(0, 2, 1) #(batch_size, num_points, dim_input)
        # print(f"Permuted x shape: {x.shape}") 
        x = self.isab(x)
        # print(f"ISAB output shape: {x.shape}")  # ここで形状を確認
        x = x.mean(dim=1)  # Global feature representation
        # print(f"After mean shape: {x.shape}")  # ここで形状を確認
        z = self.fc_enc(x)  # Project to latent space
        # print(f"fc_enc output shape: {z.shape}")  # ここで形状を確認
        z_quantized, quantization_loss = self.quantizer(z)
        return z, z_quantized, quantization_loss

    def decode(self, z):
        z = z.unsqueeze(-1)  # Reshape for transposed convs
        x_recon = self.decoder(z)
        x_recon = x_recon.squeeze(-1).view(-1, self.num_points, self.dim_input)
        return x_recon

    def forward(self, x):
        z, z_quantized, quantization_loss = self.encode(x)
        x_recon = self.decode(z_quantized)
        return x_recon, quantization_loss, z, self.embedding_dim

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z):
        z_flattened = z.view(-1, self.embedding_dim)
        distances = (
            (z_flattened ** 2).sum(dim=1, keepdim=True)
            + (self.embedding.weight ** 2).sum(dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        z_quantized = self.embedding(encoding_indices).view_as(z)
        
        quantization_loss = F.mse_loss(z_quantized.detach(), z) + self.beta * F.mse_loss(z_quantized, z.detach())
        z_quantized = z + (z_quantized - z).detach()
        
        return z_quantized, quantization_loss
