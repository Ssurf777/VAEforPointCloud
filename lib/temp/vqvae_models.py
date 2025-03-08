import torch
import torch.nn as nn
import torch.nn.functional as F

class VQVAE(nn.Module):
    def __init__(self, n_in_out, embedding_dim=3, num_embeddings=1024):
        """
        Vector Quantized Variational Autoencoder (VQVAE).
        Args:
            n_in_out (int): Dimension of input/output.
            embedding_dim (int): Dimension of the embedding space.
            num_embeddings (int): Number of embeddings in the codebook.
        """
        super(VQVAE, self).__init__()
        self.embedding_dim = embedding_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(n_in_out, 64, kernel_size=1),
            nn.LeakyReLU(0.001),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.LeakyReLU(0.001),
            nn.Conv1d(128, embedding_dim, kernel_size=1),
            nn.AdaptiveMaxPool1d(1)
        )

        # Quantizer
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(embedding_dim, 128, kernel_size=1),
            nn.LeakyReLU(0.001),
            nn.ConvTranspose1d(128, 64, kernel_size=1),
            nn.LeakyReLU(0.001),
            nn.ConvTranspose1d(64, n_in_out, kernel_size=1)
        )

    def forward(self, x):
        z = self.encoder(x).squeeze(-1)  # Encoder output
        z_quantized, quantization_loss = self.quantizer(z)  # Quantization
        x_recon = self.decoder(z_quantized.unsqueeze(-1))  # Decoder output
        return x_recon, quantization_loss, z, self.embedding_dim

    def decode(self, z):
        """
        Decodes the latent variable z to reconstruct the original data.
        Args:
            z (Tensor): Latent variable.
        Returns:
            Tensor: Reconstructed data.
        """
        if z.dim() == 2:  # Add dimension if necessary
            z = z.unsqueeze(-1)
        x_recon = self.decoder(z)
        return x_recon

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25):
        """
        Vector Quantizer for VQVAE.
        Args:
            num_embeddings (int): Number of embeddings in the codebook.
            embedding_dim (int): Dimension of each embedding.
            beta (float): Commitment loss parameter.
        """
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z):
        z_flattened = z.view(-1, self.embedding_dim)

        # Compute distances to each embedding vector
        distances = (z_flattened ** 2).sum(dim=1, keepdim=True) + \
                    (self.embedding.weight ** 2).sum(dim=1) - \
                    2 * torch.matmul(z_flattened, self.embedding.weight.t())

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        z_quantized = self.embedding(encoding_indices).view_as(z)

        # Compute quantization loss
        quantization_loss = F.mse_loss(z_quantized.detach(), z) + self.beta * F.mse_loss(z_quantized, z.detach())

        # Pass gradients through quantized representation
        z_quantized = z + (z_quantized - z).detach()

        return z_quantized, quantization_loss

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
