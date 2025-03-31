# ---encoding:utf-8---
# @Time    : 2025/3/21 23:30
# @Author  : AJAXXJ
# @Email   : 1751353695@qq.com
# @Project : sce
# @Software: PyCharm
from torch import nn

from vq_vae.models.vector_quantization import vq, vq_st


# 针对 Linear 层的 Xavier 初始化函数
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight.data)  # 使用 Xavier 均匀初始化
        if m.bias is not None:
            m.bias.data.fill_(0)


# 修改后的 VQEmbedding，不再对张量维度进行转置，适用于一维输入
class VQEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.embedding = nn.Embedding(K, D)  # 创建 K x D 的嵌入矩阵
        self.embedding.weight.data.uniform_(-1. / K, 1. / K)  # 初始化嵌入权重

    def forward(self, z_e_x):
        # z_e_x 形状为 (B, D)，直接进行向量量化
        indices = vq(z_e_x, self.embedding.weight)  # 此处 vq 函数需自行实现或导入
        return indices

    def straight_through(self, z_e_x):
        # 采用直通估计器进行量化，返回量化后的结果及根据索引重新构造的码字
        z_q_x, indices = vq_st(z_e_x, self.embedding.weight.detach())  # vq_st 同样需自行实现或导入
        # 直接利用嵌入层获取对应码字
        z_q_x_bar = self.embedding(indices)
        return z_q_x, z_q_x_bar


# 修改后的 VectorQuantizedVAE，适用于基因表达矩阵（输入为一维向量）
class VQVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout=0.5, K=512):
        super().__init__()
        # 编码器
        self.encoder = nn.ModuleList()

        for i in range(len(hidden_dim)):
            if i == 0:  # 输入层
                self.encoder.append(
                    nn.Sequential(
                        nn.Dropout(p=dropout),
                        nn.Linear(input_dim, hidden_dim[i]),
                        nn.BatchNorm1d(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
            else:  # 隐藏层
                self.encoder.append(
                    nn.Sequential(
                        nn.Dropout(p=dropout),
                        nn.Linear(hidden_dim[i - 1], hidden_dim[i]),
                        nn.BatchNorm1d(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )

        self.encoder.append(nn.Linear(hidden_dim[-1], latent_dim))
        # 向量量化嵌入层
        self.codebook = VQEmbedding(K, latent_dim)
        # 解码器
        self.decoder = nn.ModuleList()
        for i in range(len(hidden_dim)):
            if i == 0:  # 第一层
                self.decoder.append(
                    nn.Sequential(
                        nn.Linear(latent_dim, hidden_dim[i]),
                        nn.BatchNorm1d(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
            else:  # 其他隐藏层
                self.decoder.append(
                    nn.Sequential(
                        nn.Dropout(p=dropout),
                        nn.Linear(hidden_dim[i - 1], hidden_dim[i]),
                        nn.BatchNorm1d(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
        self.decoder.append(nn.Linear(hidden_dim[-1], input_dim))

        # 对所有 Linear 层进行 Xavier 初始化
        self.apply(weights_init)

    def encode(self, x):
        for layer in self.encoder:
            x = layer(x)  # 逐层通过 encoder
        scatter_latents = self.codebook(x)  # 利用向量量化层得到离散编码
        continuous_latents = self.codebook.embedding(scatter_latents) # 根据量化后的离散索引，通过嵌入层恢复对应向量，再解码回原始维度
        return continuous_latents

    def decode(self, latents):
        z_q_x = latents
        for layer in self.decoder:
            z_q_x = layer(z_q_x)  # 逐层通过 decoder
        gene_reconstructions = nn.ReLU()(z_q_x) # only relu when inference
        return gene_reconstructions

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        z_e_x = x
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        x_tilde = z_q_x_st
        for layer in self.decoder:
            x_tilde = layer(x_tilde)
        return x_tilde, z_e_x, z_q_x # 返回重建结果、编码器输出和量化结果