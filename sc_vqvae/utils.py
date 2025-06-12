import os.path
import sys

import torch

from config import VQVAEClassificationConfig
from models.vqvae import VQVAETransformerLit

EXC_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(EXC_DIR)

def load_vqvae_encoder(config: VQVAEClassificationConfig):
    vae_lit_model = VQVAETransformerLit(config.vqvae_config)

    # 如果有预训练模型，加载它
    ckpt_path = os.path.join(config.vqvae_config.vqvae_checkpoints_path, 'last.ckpt')
    checkpoint = torch.load(ckpt_path)

    state_dict = checkpoint['state_dict']
    vae_lit_model.load_state_dict(state_dict)
    vae_lit_model.train()

    return vae_lit_model.model

