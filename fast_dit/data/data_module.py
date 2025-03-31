# ---encoding:utf-8---
# @Time    : 2025/3/25 21:48
# @Author  : AJAXXJ
# @Email   : 1751353695@qq.com
# @Project : sce
# @Software: PyCharm
import os
import torch
import numpy as np
import lightning as L
from torch.utils.data import DataLoader, Dataset, random_split
from fast_dit.config import FastDiTModelConfig


class FastDiTDataset(Dataset):
    def __init__(self, config: FastDiTModelConfig):
        # 加载统一保存的 npz 文件，文件保存在 config.features_dir 下，文件名为 features.npz
        npz_path = os.path.join(config.features_dir, "features.npz")
        data = np.load(npz_path)
        self.latents = data["latents"]
        self.labels = data["labels"]

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        # 根据索引返回对应的特征和标签数据，并转换为 torch tensor
        feature = self.latents[idx]
        label = self.labels[idx]
        return torch.tensor(feature), torch.tensor(label)



class FastDiTLightningDataModule(L.LightningDataModule):

    def __init__(self, config:FastDiTModelConfig):
        super().__init__()
        self.config = config
        self.seed = config.seed
        self.batch_size = config.batch_size


    def setup(self, stage: str) -> None:
        full_dataset = FastDiTDataset(self.config)

        total_samples = full_dataset.__len__()

        train_size = int(total_samples * 0.8)
        val_size = int(total_samples * 0.1)
        test_size = total_samples - train_size - val_size

        # 使用 torch.utils.data.random_split 进行数据集分割
        generator = torch.Generator().manual_seed(self.seed)
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=generator
        )
        # 根据不同阶段设置数据集
        if stage == 'fit' or stage is None:
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
        if stage == 'test' or stage is None:
            self.test_dataset = test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          drop_last=True,
                          num_workers=getattr(self.config, "num_workers", 0)
                          )

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=getattr(self.config, "num_workers", 0)
                          )

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=getattr(self.config, "num_workers", 0)
                          )
