# ---encoding:utf-8---
# @Time    : 2025/3/25 21:45
# @Author  : AJAXXJ
# @Email   : 1751353695@qq.com
# @Project : sce
# @Software: PyCharm
from collections import OrderedDict
from copy import deepcopy
import torch
import lightning as L
from fast_dit.config import FastDiTModelConfig
from fast_dit.diffusion import create_diffusion
from fast_dit.models.prototype_model import SCEDiT


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    通过指数加权平均 (EMA) 模型更新模型的权重
    ema_model: 目标 EMA 模型
    model: 当前训练中的模型
    decay: 控制 EMA 更新速率的衰减因子（默认为 0.9999）
    """
    # 获取 ema_model 和 model 的所有参数，并转换为有序字典形式
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    # 遍历当前模型的所有参数
    for name, param in model_params.items():
        # name = name.replace("module.", "") 非分布式
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    设置模型中所有参数的 requires_grad 标志
    model: 目标模型。
    flag: 如果为 True，则设置所有参数的 requires_grad 为 True，表示需要计算梯度；如果为 False，则禁用梯度计算
    """
    for p in model.parameters():
        p.requires_grad = flag


class FastDiTLightningModel(L.LightningModule):

    def __init__(self, config: FastDiTModelConfig):
        super().__init__()
        self.lr = config.lr
        self.model = SCEDiT(latent_dim=config.vq_vae_config.latent_dim,
                            hidden_dim=config.hidden_dim,
                            depth=config.depth, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio,
                            dropout=config.dropout, num_classes=config.vq_vae_config.cell_types,
                            learn_sigma=config.learn_sigma)
        self.ema = deepcopy(self.model).to(self.device)
        requires_grad(self.ema, False)
        self.diffusion = create_diffusion(timestep_respacing="")  # 默认1000步 线性噪声调度
        update_ema(self.ema, self.model, decay=0)  # 确保使用同步权重初始化EMA
        self.ema.eval()

    def forward(self, x):
        return self.model(x)

    def calculation_process(self, batch, batch_idx):
        samples, cell_type = batch
        # 对批次中每个样本都生成随机的时间步数
        t = torch.randint(0, self.diffusion.num_timesteps, (samples.shape[0],), device=self.device)
        model_kwargs = dict(y=cell_type)  # 将标签传递
        # 计算损失
        loss_dict = self.diffusion.training_losses(self.model, samples, t, model_kwargs)
        loss = loss_dict["loss"].mean()
        # 更新 EMA
        update_ema(self.ema, self.model)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.calculation_process(batch, batch_idx)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.calculation_process(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.calculation_process(batch, batch_idx)
        self.log('test_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0)
