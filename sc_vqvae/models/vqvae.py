import torch
import torch.nn as nn
from torch import optim
from .transformer import PatchEmbedding, TransformerEncoder, TransformerDecoder
import lightning as L

from vector_quantize_pytorch import VectorQuantize

from sc_vqvae.config import VQVAETransformerConfig


class VQEncoder(nn.Module):
    def __init__(self, gene_dim, patch_size=100, embed_dim=128, encoder_layers=4, heads=8, n_codebook=512):
        super().__init__()
        self.patch_embed = PatchEmbedding(gene_dim, patch_size, embed_dim)
        self.encoder = TransformerEncoder(embed_dim, encoder_layers, heads)
        self.vq = VectorQuantize(
            dim = embed_dim,
            codebook_size = n_codebook,
            commitment_weight = 0.25,
            use_cosine_sim = False    # 通常用于连续空间，Euclidean距离更常见
        )

    def forward(self, x):
        # x: (N, G)
        x_patch, n_patch, pad_len = self.patch_embed(x)

        # x_patch: (N, T, D)
        z = self.encoder(x_patch)
        
        # Vector Quantization
        z_q, codes, vq_loss = self.vq(z)

        return z_q, codes, vq_loss
    
    def get_codebook(self):
        # Get the codebook from the VQ layer
        return self.vq.get_codebook()

class VQDecoder(nn.Module):
    """ VQ Decoder: decode the quantized latent vectors back to the original space, with remask """
    def __init__(self, gene_dim, patch_size=100, embed_dim=128, decoder_layers=4, heads=8):
        super().__init__()
        self.decoder = TransformerDecoder(embed_dim, decoder_layers, heads)
        self.projector = nn.Linear(embed_dim, patch_size)
        self.gene_dim = gene_dim

    def forward(self, z_q):
        # z_q: (N, T, D)
        y_patch = self.decoder(z_q)
        y_patch = self.projector(y_patch)  # (N, T, P)
        y = y_patch.view(z_q.size(0), -1)[:, :self.gene_dim]  # 去除padding，恢复原始维度
        return y

class VQVAETransformer(nn.Module):
    def __init__(self, gene_dim, patch_size=16, embed_dim=128, n_codebook=512,
                 encoder_layers=4, decoder_layers=4, heads=8):
        super().__init__()
        self.gene_dim = gene_dim
        self.encoder = VQEncoder(gene_dim, patch_size, embed_dim, encoder_layers, heads, n_codebook)
        self.decoder = VQDecoder(gene_dim, patch_size, embed_dim, decoder_layers, heads)


    def forward(self, x):
        # x: (N, G)
        z_q, codes, vq_loss = self.encoder(x)
        y = self.decoder(z_q)
        return y, vq_loss
    
    def encode(self, x):
        # Encode the input x to the quantized latent space
        z_q, codes, vq_loss = self.encoder(x)
        return z_q, codes
    
    def decode(self, z_q):
        # Decode the quantized latent vectors back to the original space
        y = self.decoder(z_q)
        return y
    
    def get_codebook(self):
        # Get the codebook from the VQ layer
        return self.encoder.get_codebook()

class VQVAETransformerLit(L.LightningModule):

    def __init__(self, config: VQVAETransformerConfig):
        super().__init__()
        self.save_hyperparameters(vars(config))
        self.config = config
        self.model = VQVAETransformer(
            gene_dim=config.gene_dim,
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
            n_codebook=config.n_codebook,
            encoder_layers=config.encoder_layers,
            decoder_layers=config.decoder_layers,
            heads=config.num_heads
        )
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        recon, vq_loss = self(x)
        recon_loss = self.criterion(recon, x)
        loss = recon_loss + vq_loss
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_recon", recon_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_vq", vq_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        recon, vq_loss = self(x)
        recon_loss = self.criterion(recon, x)
        loss = recon_loss + vq_loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_recon", recon_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_vq", vq_loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.config.lr)


