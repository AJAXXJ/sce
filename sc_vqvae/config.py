# ---encoding:utf-8---
# @Time    : 2025/6/3 10:31
# @Author  : AJAXXJ
# @Email   : 1751353695@qq.com
# @Project : sce-plus
# @Software: PyCharm
import os
import random
import sys

import numpy as np
import torch

EXC_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(EXC_DIR)

class VQVAETransformerConfig:

    def __init__(self, seed = 42):

        self.seed = seed
        self.eval_epoch_step = 10
        self.vqvae_checkpoints_path = os.path.join(EXC_DIR, 'sc_vqvae/checkpoints/vqvae_models')
        # data
        self.data_path = os.path.join(EXC_DIR, 'sc_vqvae/data')
        self.data_name = 'processed'
        self.batch_size = 128
        # self.num_workers = 0
        # training
        self.epochs = 200
        self.lr = 1e-4
        self.gene_dim = None
        self.patch_size = 100
        self.embed_dim = 128
        self.n_codebook = 1024
        self.encoder_layers = 4
        self.decoder_layers = 4
        self.num_heads = 8
        self.patience = 50


    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        # 确保cuDNN的可重复性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class VQVAEClassificationConfig:

    def __init__(self, seed=42):
        self.seed = seed
        self.eval_epoch_step = 10
        self.classifier_checkpoints_path = os.path.join(EXC_DIR, 'sc_vqvae/checkpoints/classifier_models')
        # vqvae
        self.vqvae_config = VQVAETransformerConfig()
        # classification
        self.num_layers = 2
        self.num_classes = 10
        # data
        self.data_path = os.path.join(EXC_DIR, 'sc_vqvae/data/processed')
        self.batch_size = 256
        # self.num_workers = 0
        # training
        self.epochs = 200
        self.patience = 50
        self.lr = 1e-2
        self.weight_decay = 1e-4
        self.dropout_rate = 0.5



    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        # 确保cuDNN的可重复性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

