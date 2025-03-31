# ---encoding:utf-8---
# @Time    : 2025/3/21 13:52
# @Author  : AJAXXJ
# @Email   : 1751353695@qq.com
# @Project : SingleCellExperiment
# @Software: PyCharm
import scipy.sparse as sp
import torch
import scanpy as sc
import lightning as L
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader, random_split

from vq_vae.config import VQVAEModelConfig


class H5ADDataset(Dataset):

    def __init__(self, file_path, label_column='cell_type'):
        # 读取 h5ad 数据
        self.adata = sc.read_h5ad(file_path)
        # 提取表达矩阵，若为稀疏矩阵请转换为稠密矩阵
        self.data = self.adata.X
        if sp.issparse(self.data):
            self.data = self.data.toarray()
        # 从 obs 中提取细胞类型标签
        self.labels = self.adata.obs[label_column].values

        # 创建标签映射
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)  # 将标签映射为 ID

    def __getitem__(self, idx):
        # 获取表达数据
        sample = self.data[idx, :]
        sample = torch.tensor(sample, dtype=torch.float)

        # 获取对应的标签
        label = self.labels[idx]
        return sample, label

    def __len__(self):
        return self.data.shape[0]

class GeneExpressionDataModule(L.LightningDataModule):

    def __init__(self, config:VQVAEModelConfig):
        super().__init__()
        self.config = config
        self.seed = config.seed
        self.batch_size = config.batch_size
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None


    def setup(self, stage: str) -> None:
        full_dataset = H5ADDataset(self.config.file_path, label_column='cell_type')
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
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=getattr(self.config, "num_workers", 0)
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=getattr(self.config, "num_workers", 0)
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=getattr(self.config, "num_workers", 0)
        )

