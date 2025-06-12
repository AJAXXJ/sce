import torch
from torch import nn


class PatchEmbedding(nn.Module):
    def __init__(self, gene_dim, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.gene_dim = gene_dim
        self.embed_dim = embed_dim
        self.proj = nn.Linear(patch_size, embed_dim)
    
    def forward(self, x):
        # x: (N, G)
        N, G = x.shape
        n_patch = (G + self.patch_size - 1) // self.patch_size
        pad_len = n_patch * self.patch_size - G
        if pad_len > 0:
            x = torch.cat([x, torch.zeros(N, pad_len, device=x.device)], dim=1)
        x = x.view(N, n_patch, self.patch_size)
        x = self.proj(x)  # (N, T, D)
        # Add RoPE positional encoding
        # x = apply_rope(x)
        # Add absolute positional encoding
        x = add_absolute_positional_encoding(x)
        return x, n_patch, pad_len

def apply_rope(x):
    """
    Apply Rotary Positional Encoding to input tensor x.
    x: (N, T, D)
    """
    N, T, D = x.shape
    # Only even D is supported for simplicity
    if D % 2 != 0:
        raise ValueError("Embedding dimension must be even for RoPE.")
    device = x.device
    # Build position ids
    pos = torch.arange(T, device=device, dtype=torch.float32)  # (T,)
    dim = torch.arange(0, D // 2, device=device, dtype=torch.float32)  # (D//2,)
    theta = 10000 ** (-dim / (D // 2))
    # (T, D//2)
    freqs = torch.einsum('t,d->td', pos, theta)
    # (T, D)
    sin, cos = freqs.sin(), freqs.cos()
    sin = torch.stack([sin, sin], dim=-1).reshape(T, D)
    cos = torch.stack([cos, cos], dim=-1).reshape(T, D)
    # x: (N, T, D)
    x1, x2 = x[..., ::2], x[..., 1::2]  # (N, T, D//2)
    x_rope_even = x1 * cos[:, ::2] - x2 * sin[:, ::2]
    x_rope_odd  = x1 * sin[:, ::2] + x2 * cos[:, ::2]
    x_rope = torch.stack([x_rope_even, x_rope_odd], dim=-1).reshape(N, T, D)
    return x_rope


def add_absolute_positional_encoding(x):
    """
    Add absolute sinusoidal positional encoding to input tensor x.
    x: (N, T, D)
    """
    N, T, D = x.shape
    if D % 2 != 0:
        raise ValueError("Embedding dimension must be even for sinusoidal encoding.")

    device = x.device
    pos = torch.arange(T, device=device).unsqueeze(1)  # (T, 1)
    dim = torch.arange(D // 2, device=device).unsqueeze(0)  # (1, D//2)
    angle_rates = 1 / torch.pow(10000, (2 * dim) / D)  # (1, D//2)
    angle_rads = pos * angle_rates  # (T, D//2)

    pe = torch.zeros(T, D, device=device)
    pe[:, 0::2] = torch.sin(angle_rads)
    pe[:, 1::2] = torch.cos(angle_rads)

    x = x + pe.unsqueeze(0)  # shape: (1, T, D) broadcast to (N, T, D)
    return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, depth=4, heads=8, mlp_ratio=4.):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=heads, dim_feedforward=int(embed_dim*mlp_ratio),
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        return self.encoder(x)

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, depth=4, heads=8, mlp_ratio=4.):
        super().__init__()
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=heads, dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=depth)

    def forward(self, x):
        return self.decoder(x)