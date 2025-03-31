# ---encoding:utf-8---
# @Time    : 2025/3/25 16:29
# @Author  : AJAXXJ
# @Email   : 1751353695@qq.com
# @Project : sce
# @Software: PyCharm
import math
import torch
import numpy as np
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.vision_transformer import Attention, Mlp



def modulate(x, shift, scale):
    # 对 x 进行广播
    # shift 和 scale 通过 unsqueeze 扩展到 (N, T, hidden_size)
    shift = shift.unsqueeze(1)  # 变成 (N, 1, hidden_size)
    scale = scale.unsqueeze(1)  # 变成 (N, 1, hidden_size)

    return x * (1 + scale) + shift

def get_1d_sin_cos_pos_embed(hidden_dim, pos):
    """
    :param hidden_dim: 隐藏层维度
    :param pos: (M,) 位置编码
    :return: out: (M, D)
    """
    assert hidden_dim % 2 == 0
    omega = np.arange(hidden_dim // 2, dtype=np.float64)
    omega /= hidden_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class TimestepEmbedder(nn.Module):
    """
    将标量时间步长嵌入到向量表示中 用于扩散模型，以提供时间信息
    """

    def __init__(self, hidden_dim, frequency_embedding_size=256):
        """
        :param hidden_dim: 隐藏层维度
        :param frequency_embedding_size: 频率嵌入的维度，决定正弦-余弦编码的大小
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        生成时间步的正弦-余弦嵌入
        :param t: (N,) 形状的时间步向量，每个样本对应一个时间步
        :param dim: 目标嵌入维度。
        :param max_period: 控制最小频率（值越大，周期越长）
        :return: (N, dim) 形状的时间步嵌入
        """
        # 计算一半维度的频率编码
        half = dim // 2
        # 计算不同频率的指数衰减
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        # 计算 cos 和 sin 编码
        args = t[:, None].float() * freqs[None] # (N, half)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1) # (N, H)
        # 如果 `dim` 是奇数，则补 0 保证维度一致
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding # (N, H)

    def forward(self, t):
        """
        前向传播：将时间步 t 通过 MLP 映射到 hidden_dim。
        :param t: (N,) 形状的时间步输入
        :return: (N, H) 形状的时间步嵌入
        """
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class LabelEmbedder(nn.Module):
    """
    将类别标签（labels）嵌入到向量表示中，并支持类别丢弃（用于 Classifier-Free Guidance）
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        """
        :param num_classes: 总类别数
        :param hidden_size: 嵌入向量的维度
        :param dropout_prob: 标签丢弃概率（用于 Classifier-Free Guidance）
        """
        super().__init__()
        # 如果 dropout_prob > 0，则需要额外的无类别 ID (num_classes)
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        进行标签丢弃（用于 Classifier-Free Guidance）

        :param labels: (N,) 形状的类别标签
        :param force_drop_ids: (N,) 形状的布尔值张量，如果提供，则强制指定哪些标签被丢弃。
        :return: (N,) 形状的丢弃后类别标签，其中部分标签可能被替换为 `num_classes`（代表无类别）
        """
        if force_drop_ids is None:
            # 以 dropout_prob 的概率随机丢弃类别
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            # 如果提供 force_drop_ids，则按照给定的布尔值进行丢弃
            drop_ids = force_drop_ids == 1
        # 将被丢弃的标签替换为 `num_classes`（代表无类别）
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        """
        前向传播：将类别标签映射到嵌入空间，并进行可选的类别丢弃
        :param labels: (N,) 形状的类别标签。
        :param train: 是否处于训练模式（决定是否启用类别丢弃）。
        :param force_drop_ids: (N,) 形状的布尔值张量，强制指定哪些标签被丢弃。
        :return: (N, hidden_size) 形状的类别嵌入向量。
        """
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_dim, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, out_dim, bias=True)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class SCEDiT(nn.Module):
    """
    以Transformer为骨干的Diffusion 针对输入做修改
    """

    def __init__(self, latent_dim, hidden_dim, depth, num_heads, mlp_ratio, dropout, num_classes, learn_sigma):
        """
        :param latent_dim: 潜在特征维度
        :param hidden_dim: 隐藏层维度
        :param depth: DiTBlock 模块数量
        :param num_heads: 多头注意力的头数
        :param mlp_ratio: MLP 扩展 hidden_dim 比例
        :param dropout: LabelEmbedder 中标签丢弃率
        :param num_classes: 总类别数量
        :param learn_sigma: 是否学习噪声尺度 True 输出通道数会翻倍 hidden_dim * 2 通常一部分用于预测噪声本身 另一部分用于预测 sigma
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.learn_sigma = learn_sigma
        self.out_dim = latent_dim * 2 if learn_sigma else latent_dim
        # 线性层
        self.x_embedder = nn.Linear(latent_dim, hidden_dim)
        # 时间步嵌入模块，将扩散过程中的时间信息嵌入到与 hidden_dim 相同的向量中
        self.t_embedder = TimestepEmbedder(hidden_dim)
        # 标签嵌入模块，将类别标签转换为与 hidden_dim 相同的向量，并支持随机丢弃标签信息
        self.y_embedder = LabelEmbedder(num_classes, hidden_dim, dropout)
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, hidden_dim))
        # 构建 Transformer 模块，由 depth 个 DiTBlock 组成，每个 block 内部实现了自注意力和 MLP 机制
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        # 输出层
        self.final_layer = FinalLayer(hidden_dim, self.out_dim)  # 关键修改

        self.initialize_weights()


    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (N, D) tensor of latents
        t: (N) tensor of diffusion timesteps
        y: (N) tensor of class labels
        """
        # 固定位置编码
        x = self.x_embedder(x) + self.pos_embed
        # 时间步嵌入
        t = self.t_embedder(t)
        # 类别标签嵌入
        y = self.y_embedder(y, self.training)
        # 计算调制信息 c，用于控制 Transformer
        c = t + y
        # 适配transformer输入
        x = x.unsqueeze(1)
        for block in self.blocks:
            x = checkpoint.checkpoint(self.ckpt_wrapper(block), x, c)
        x = self.final_layer(x, c)
        x = x.squeeze(1)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        分类器指导前向传播 推理阶段
        :param x: (N, D) 潜在特征
        :param t: (N,) diffusion timesteps 时间步
        :param y: (N,) class labels 类别标签
        :param cfg_scale: 控制分类器自由引导强度的系数
        :return: 经过 CFG 调整后的预测结果
        """
        # 分为 无条件 和 有条件 两部分
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        eps, rest = model_out[:, :self.latent_dim], model_out[:, self.latent_dim:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    def initialize_weights(self):

        # 基本的线性层初始化
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # 初始化1D正弦余弦位置编码
        pos_embed = get_1d_sin_cos_pos_embed(self.pos_embed.shape[-1], np.array([0]))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

        # 初始化标签和时间步嵌入相关参数
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers in final_layer:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def ckpt_wrapper(self, module):
        """
        梯度检查点存储 通过丢弃部分前向计算结果 在反向传播时重新计算来节省显存
        :param module:
        :return:
        """
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward