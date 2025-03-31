# ---encoding:utf-8---
# @Time    : 2025/3/26 10:19
# @Author  : AJAXXJ
# @Email   : 1751353695@qq.com
# @Project : sce
# @Software: PyCharm
import numpy as np
import torch
from torch.utils.data import DataLoader
from fast_dit.config import FastDiTModelConfig
from vq_vae.data.data_module import H5ADDataset
from vq_vae.models.vq_vae_model import VQVAELightningModel


def extract_features(config:FastDiTModelConfig):
    config.set_seed()
    torch.set_grad_enabled(False)

    # VQVAE前数据加载
    full_dataset = H5ADDataset(config.vq_vae_config.file_path)
    train_loader = DataLoader(
        full_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=getattr(config, "num_workers", 0),
        pin_memory=True,
        drop_last=True
    )
    # VQVAE模型加载
    vq_vae = VQVAELightningModel.load_from_checkpoint(config.vq_vae_config.checkpoint, config=config.vq_vae_config)
    vq_vae = vq_vae.to(config.device)
    vq_vae.eval()

    # 特征抽取
    latents_list = []
    labels_list = []
    for data in train_loader:
        sample = data[0].to(config.device)
        labels = data[1].to(config.device)

        latents = vq_vae.model.encode(sample)
        # 每个 sample 的 shape 为 (1, feature_dim)，这里添加到列表中
        latents_list.append(latents.detach().cpu().numpy())
        labels_list.append(labels.detach().cpu().numpy())

    # 拼接所有数据，最终 shape 为 (总样本数, feature_dim)
    latents_all = np.concatenate(latents_list, axis=0)
    labels_all = np.concatenate(labels_list, axis=0)

    # 保存为一个压缩文件
    np.savez(f'{config.features_dir}/features.npz', latents=latents_all, labels=labels_all)


if __name__ == '__main__':
    extract_features(FastDiTModelConfig())