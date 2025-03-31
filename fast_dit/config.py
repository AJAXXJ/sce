# ---encoding:utf-8---
# @Time    : 2025/3/25 16:21
# @Author  : AJAXXJ
# @Email   : 1751353695@qq.com
# @Project : sce
# @Software: PyCharm
import os
import random
import sys

import numpy as np
import torch

from vq_vae.config import VQVAEModelConfig

EXC_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(EXC_DIR)

class FastDiTModelConfig:

    def __init__(self, seed=237):
        self.h5ad_name = "example"
        self.checkpoint = os.path.join(EXC_DIR, "fast_dit/checkpoints/last.ckpt")
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # training
        self.epochs = 100
        self.eval_epoch_step = 5
        # VQ-VAE

        self.vq_vae_config = VQVAEModelConfig()

        # dataset
        self.file_path = os.path.join(EXC_DIR, "vq_vae/data/raw_data/example.h5ad")
        self.batch_size = 128
        # self.num_workers = 0
        self.features_dir = os.path.join(EXC_DIR,"fast_dit/data/features/", self.h5ad_name)

        # fast-dit
        self.hidden_dim = 1024
        self.lr = 1e-4
        self.depth = 28
        self.num_heads = 16
        self.mlp_ratio = 4.0
        self.dropout = 0.1
        self.learn_sigma = True
        self.cfg_scale = 4.0
        self.num_sampling_steps = 250

        # sample
        self.samples_num = 12000
        self.samples_batch_size = 3000
        self.sample_dir = os.path.join(EXC_DIR,"fast_dit/data/samples/", self.h5ad_name)


    # 保证实现的可重复性
    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        # 确保cuDNN的可重复性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    print(torch.cuda.is_available())