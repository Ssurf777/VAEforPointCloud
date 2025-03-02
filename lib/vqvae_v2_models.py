import torch
import torch.nn as nn
import torch.nn.functional as F

class VQVAE(nn.Module):
    def __init__(self, n_dim=3, num_points=5000, embedding_dim=3, num_embeddings=1024):
        """
        Vector Quantized Variational Autoencoder (VQVAE).

        Args:
            n_dim (int): The dimensionality of the input features. For instance,
                if input is a point cloud with xyz, n_dim would be 3.
            num_points (int): The number of points in each input sample.
            embedding_dim (int): The dimensionality of the latent embedding space.
            num_embeddings (int): The number of embedding vectors in the codebook.
        """
        super(VQVAE, self).__init__()
        self._init_weights()
        self.n_dim = n_dim
        self.num_points = num_points
        self.embedding_dim = embedding_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(n_dim, 64, kernel_size=1),
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
            nn.ConvTranspose1d(64, num_points * 3, kernel_size=1)
        )

    def forward(self, x):
        """
        Forward pass through the VQVAE.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, n_dim, num_points].

        Returns:
            x_recon (torch.Tensor): Reconstructed output.
            quantization_loss (torch.Tensor): Quantization loss from the VectorQuantizer.
            z (torch.Tensor): Latent representation before quantization.
            embedding_dim (int): The embedding dimensionality (for reference).
        """
        z = self.encoder(x).squeeze(-1)  # Encoder output
        z_quantized, quantization_loss = self.quantizer(z)  # Quantization
        x_recon = self.decoder(z_quantized.unsqueeze(-1))  # Decoder output
        return x_recon, quantization_loss, z, self.embedding_dim

    def decode(self, z):
        """
        Decodes the latent variable z to reconstruct the original data.

        Args:
            z (torch.Tensor): Latent variable of shape [batch_size, embedding_dim] or
                [batch_size, embedding_dim, 1].

        Returns:
            torch.Tensor: The reconstructed output.
        """
        if z.dim() == 2:  # Add a channel dimension if necessary
            z = z.unsqueeze(-1)
        x_recon = self.decoder(z)  # [batch_size, num_points*3, 1]
        x_recon = x_recon.squeeze(-1).view(-1, self.num_points, 3)  # [batch_size, num_points, 3]

        return x_recon

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25):
        """
        Vector Quantizer for VQVAE.

        Args:
            num_embeddings (int): The number of embeddings in the codebook.
            embedding_dim (int): The dimensionality of each embedding vector.
            beta (float): The commitment loss parameter that controls the
                importance of matching the encoder output to the chosen
                embedding vector (codebook entry).
        """
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z):
        """
        Forward pass through the vector quantizer.

        Args:
            z (torch.Tensor): The latent representation (encoder output)
                of shape [batch_size, embedding_dim].

        Returns:
            z_quantized (torch.Tensor): The quantized latent representation,
                shaped like the input z.
            quantization_loss (torch.Tensor): MSE-based loss for the commitment
                and codebook distance.
        """
        z_flattened = z.view(-1, self.embedding_dim)

        # Compute distances to each embedding vector
        distances = (
            (z_flattened ** 2).sum(dim=1, keepdim=True)
            + (self.embedding.weight ** 2).sum(dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        z_quantized = self.embedding(encoding_indices).view_as(z)

        # Compute quantization loss
        quantization_loss = F.mse_loss(z_quantized.detach(), z) + self.beta * F.mse_loss(z_quantized, z.detach())

        # Pass gradients through quantized representation
        z_quantized = z + (z_quantized - z).detach()

        return z_quantized, quantization_loss

    def _init_weights(self):
        """
        Applies He initialization (Kaiming initialization) to all linear,
        convolution, and transpose convolution layers in this module.
        """
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
