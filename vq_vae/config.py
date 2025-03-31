# ---encoding:utf-8---
# @Time    : 2025/3/7 20:46
# @Author  : AJAXXJ
# @Email   : 1751353695@qq.com
# @Project : SingleCellExperiment
# @Software: PyCharm
import os
import random
import sys
import scanpy as sc

import numpy as np
import torch


EXC_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(EXC_DIR)

class VQVAEModelConfig:

    def __init__(self, seed=237):
        self.seed = seed
        self.checkpoint = os.path.join(EXC_DIR, "vq_vae/checkpoints/last.ckpt")
        # training
        self.epochs = 100
        self.eval_epoch_step = 5
        # 数据目录
        self.file_path = os.path.join(EXC_DIR, "vq_vae/data/raw_data/example.h5ad")
        self.batch_size = 32
        # self.num_workers = 8
        # transformer
        h5ad_file = sc.read_h5ad(self.file_path)
        self.num_genes = h5ad_file.var.shape[0]
        self.cell_types = h5ad_file.obs['cell_type'].value_counts().__len__()
        self.dropout = 0.5
        self.hidden_dim = [1024,1024]
        self.latent_dim = 128
        self.K = 512
        self.lr = 1e-3
        self.beta = 0.25

    # 保证实现的可重复性
    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        # 确保cuDNN的可重复性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

