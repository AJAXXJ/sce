# ---encoding:utf-8---
# @Time    : 2025/3/26 10:19
# @Author  : AJAXXJ
# @Email   : 1751353695@qq.com
# @Project : sce
# @Software: PyCharm
import numpy as np
import torch

from fast_dit.config import FastDiTModelConfig
from fast_dit.diffusion import create_diffusion
from fast_dit.models.fast_dit_model import FastDiTLightningModel
from vq_vae.models.vq_vae_model import VQVAELightningModel

def save_data(all_cells, data_dir):
    cell_gen = all_cells
    np.savez(data_dir, cell_gen=cell_gen)
    return

def sample(config:FastDiTModelConfig):
    config.set_seed()
    torch.set_grad_enabled(False)

    # VQVAE模型加载
    vq_vae = VQVAELightningModel.load_from_checkpoint(config.vq_vae_config.checkpoint, config=config.vq_vae_config)
    vq_vae = vq_vae.to(config.device)
    vq_vae.eval()
    # Fast-DiT模型加载
    fast_dit = FastDiTLightningModel.load_from_checkpoint(config.checkpoint, config=config)
    fast_dit = fast_dit.to(config.device)
    fast_dit.eval()
    # 创建扩散模型
    diffusion = create_diffusion(str(config.num_sampling_steps))

    all_cells = []
    while len(all_cells) * config.samples_batch_size < config.samples_num:
        # 细胞类型标签 限制在 cell_types 内
        labels = [6]
        n = len(labels)
        # 创造初始噪声
        z = torch.randn(n, config.vq_vae_config.latent_dim, device=config.device)
        y = torch.tensor(labels, device=config.device)
        # CFG
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([config.vq_vae_config.cell_types] * n, device=config.device)
        y = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=y, cfg_scale=config.cfg_scale)

        samples = diffusion.p_sample_loop(
            fast_dit.model.forward_with_cfg,
            z.shape, z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=True,
            device=config.device
        )
        samples, _ = samples.chunk(2, dim=0)  # 移除无条件类别
        logits  = vq_vae.model.decode(samples / 0.18215) # 0.18215标准化因子
        all_cells.extend(logits.cpu().numpy())
    # 最终保存数据
    arr = np.concatenate(all_cells, axis=0)
    save_data(arr, config.sample_dir)

    print("sampling complete")


if __name__ == '__main__':
    sample(FastDiTModelConfig())