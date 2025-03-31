# ---encoding:utf-8---
# @Time    : 2025/3/25 16:21
# @Author  : AJAXXJ
# @Email   : 1751353695@qq.com
# @Project : sce
# @Software: PyCharm
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from config import FastDiTModelConfig
from data.data_module import FastDiTLightningDataModule
from models.fast_dit_model import FastDiTLightningModel

config = FastDiTModelConfig()
# 设置随机数
config.set_seed()

checkpoint_callback = ModelCheckpoint(
    dirpath="./checkpoints",         # 检查点保存目录
    filename="{epoch:02d}-{val_loss:.2f}",  # 保存文件名格式
    monitor="val_loss",              # 监控验证损失（确保模型中有相应的 metric）
    mode="min",                      # loss 越小越好
    save_top_k=3,                    # 最多保存 3 个最佳检查点
    save_last=True                   # 同时保存最后一次模型
)

# 配置 Trainer
trainer = Trainer(
    default_root_dir="./",
    check_val_every_n_epoch= config.eval_epoch_step,
    max_epochs=config.epochs,
    callbacks=[checkpoint_callback]
)
trainer.fit(FastDiTLightningModel(config), datamodule=FastDiTLightningDataModule(config))