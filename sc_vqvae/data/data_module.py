import os.path
import torch
import lightning as L
import anndata
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.model_selection import train_test_split

from sc_vqvae.config import VQVAETransformerConfig, VQVAEClassificationConfig


class VQVAETransformerDataset(Dataset):

    def __init__(self, data_path, data_name):
        filename = os.path.join(data_path, data_name, "sc_data.h5ad")
        adata = anndata.read_h5ad(filename)
        use_key = "X"
        sc_exp = adata.layers[use_key] if use_key in adata.layers else adata.X
        if not isinstance(sc_exp, np.ndarray):
            sc_exp = sc_exp.toarray()

        sc_exp = normalize(sc_exp, norm='l1', axis=0)
        self.data = torch.tensor(sc_exp, dtype=torch.float32)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]

class VQVAETransformerDataModule(L.LightningDataModule):

    def __init__(self, config: VQVAETransformerConfig):
        super().__init__()
        self.config  =config

    def setup(self, stage=None):
        full_dataset = VQVAETransformerDataset(self.config.data_path, self.config.data_name)
        self.gene_dim = full_dataset.data.shape[1]

        val_size = int(len(full_dataset) * 0.1)
        train_size = len(full_dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config.seed)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=getattr(self.config, "num_workers", 0),
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=getattr(self.config, "num_workers", 0),
            drop_last=False
        )


class VQVAEClassificationDataset(Dataset):

    def __init__(self, data_path):
        adata = anndata.read_h5ad(os.path.join(data_path, 'sc_data.h5ad'))
        sc_exp = adata.X  # 通常是稀疏矩阵或ndarray

        # 转成numpy数组，如果是稀疏矩阵要先转密集
        if not isinstance(sc_exp, np.ndarray):
            sc_exp = sc_exp.toarray()
        sc_exp = sc_exp.astype(np.float32)

        annotations = pd.read_csv(os.path.join(data_path, 'annotations.csv'))
        cell_type = annotations['celltype'].values
        # 标签编码
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(cell_type)

        # 保存标签映射
        self.label_map = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))

        # 归一化（按列L1归一化）
        sc_exp = normalize(sc_exp, norm='l1', axis=0)

        self.x = torch.tensor(sc_exp, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

        self.gene_dim = self.x.shape[1]
        self.num_classes = len(np.unique(labels))

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)

class VQVAEClassificationDataModule(L.LightningDataModule):

    def __init__(self, config: VQVAEClassificationConfig):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        full_dataset = VQVAEClassificationDataset(self.config.data_path)
        self.gene_dim = full_dataset.gene_dim
        self.num_classes = full_dataset.num_classes
        self.label_map = full_dataset.label_map

        total_size = len(full_dataset)
        indices = np.arange(total_size)

        # 按照索引划分数据
        train_idx, test_idx = train_test_split(indices, test_size=0.1, random_state=self.config.seed, stratify=full_dataset.y)
        train_idx, val_idx = train_test_split(train_idx, test_size=0.1 / (1 -0.1), random_state=self.config.seed, stratify=full_dataset.y[train_idx])

        # 使用 Subset 封装
        self.train_dataset = Subset(full_dataset, train_idx)
        self.val_dataset = Subset(full_dataset, val_idx)
        self.test_dataset = Subset(full_dataset, test_idx)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.config.batch_size,
                          shuffle=True,
                          num_workers=getattr(self.config, "num_workers", 0)
                          )

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.config.batch_size,
                          shuffle=False,
                          num_workers=getattr(self.config, "num_workers", 0)
                          )

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.config.batch_size,
                          shuffle=False,
                          num_workers=getattr(self.config, "num_workers", 0)
                          )

