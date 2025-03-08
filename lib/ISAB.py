import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiheadAttentionBlock(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads):
        super(MultiheadAttentionBlock, self).__init__()
        self.mab = nn.MultiheadAttention(embed_dim=dim_V, num_heads=num_heads, batch_first=True)
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)

    def forward(self, Q, K):
        Q_proj = self.fc_q(Q)
        K_proj = self.fc_k(K)
        output, _ = self.mab(Q_proj, K_proj, K_proj)
        return output

class ISAB(nn.Module):
    def __init__(self, dim_input, dim_output, num_heads, num_inds):
        super(ISAB, self).__init__()
        self.inducing_points = nn.Parameter(torch.randn(1, num_inds, dim_output))
        self.mab1 = MultiheadAttentionBlock(dim_output, dim_input, dim_output, num_heads)
        self.mab2 = MultiheadAttentionBlock(dim_input, dim_output, dim_output, num_heads)

    def forward(self, X):
        H = self.mab1(self.inducing_points.repeat(X.size(0), 1, 1), X)
        return self.mab2(X, H)
