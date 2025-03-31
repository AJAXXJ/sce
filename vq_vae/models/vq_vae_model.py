# ---encoding:utf-8---
# @Time    : 2025/3/21 23:55
# @Author  : AJAXXJ
# @Email   : 1751353695@qq.com
# @Project : sce
# @Software: PyCharm
import torch
import torch.nn.functional as F
import lightning as L

from vq_vae.config import VQVAEModelConfig
from vq_vae.models.prototype_model import VQVAE


class VQVAELightningModel(L.LightningModule):
    def __init__(self, config: VQVAEModelConfig):
        super().__init__()
        # 使用针对基因表达的 VQ-VAE 模型
        self.model = VQVAE(config.num_genes, config.hidden_dim,config.latent_dim,config.dropout, config.K)
        self.lr = config.lr
        self.beta = config.beta

    def forward(self, x):
        # x: 基因表达矩阵，形状 (B, num_genes)
        return self.model(x)


    def calculation_process(self,batch, batch_idx):
        x = batch[0]  # (B, num_genes)
        labels = batch[1]  # (B,)
        # 计算
        recon_x, z_e_x, z_q_x = self.model(x)
        # 计算重构损失
        loss_recons = F.mse_loss(recon_x, x)
        # 计算向量量化损失
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        loss_commit = F.mse_loss(z_e_x, z_q_x.detach())
        # 计算总损失
        loss = loss_recons + loss_vq + self.beta * loss_commit

        return loss, loss_recons, loss_vq, loss_commit


    def training_step(self, batch, batch_idx):
        loss, loss_recons, loss_vq, loss_commit = self.calculation_process(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_loss_recons", loss_recons, prog_bar=True)
        self.log("train_loss_vq", loss_vq, prog_bar=True)
        self.log("train_loss_commit", loss_commit, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_recons, loss_vq, loss_commit = self.calculation_process(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_loss_recons", loss_recons, prog_bar=True)
        self.log("val_loss_vq", loss_vq, prog_bar=True)
        self.log("val_loss_commit", loss_commit, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, loss_recons, loss_vq, loss_commit = self.calculation_process(batch, batch_idx)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_loss_recons", loss_recons, prog_bar=True)
        self.log("test_loss_vq", loss_vq, prog_bar=True)
        self.log("test_loss_commit", loss_commit, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)