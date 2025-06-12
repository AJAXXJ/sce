from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger

from config import VQVAETransformerConfig
from data.data_module import VQVAETransformerDataModule
from models.vqvae import VQVAETransformerLit


mlflow_logger = MLFlowLogger(
    experiment_name="sce",
    tags={
        '模型': 'VQVAETransformer',
        '任务': 'encode-decode'
    },
    run_name='VQVAETransformer',
    tracking_uri="http://172.29.172.33:5000/"
)

config = VQVAETransformerConfig()
# 设置随机数
config.set_seed()

checkpoint_callback = ModelCheckpoint(
    dirpath="./checkpoints/vqvae_models",         # 检查点保存目录
    filename="{epoch:02d}-{val_loss:.2f}",  # 保存文件名格式
    monitor="val_loss",              # 监控验证损失（确保模型中有相应的 metric）
    mode="min",                      # loss 越小越好
    save_top_k=3,                    # 最多保存 3 个最佳检查点
    save_last=True                   # 同时保存最后一次模型
)

# 加入 EarlyStopping 回调
early_stop_callback = EarlyStopping(
    monitor="val_loss",     # 监控验证损失
    patience=config.patience,  # config 中设定等待多少个 epoch 无提升就停止
    mode="min",             # 越小越好
    verbose=True            # 输出日志
)

# 配置 Trainer
trainer = Trainer(
    default_root_dir="./",
    check_val_every_n_epoch= config.eval_epoch_step,
    max_epochs=config.epochs,
    callbacks=[checkpoint_callback, early_stop_callback],
    logger=mlflow_logger
)

# 初始化
data_module = VQVAETransformerDataModule(config)
data_module.setup()
config.gene_dim = data_module.gene_dim
model = VQVAETransformerLit(config)

trainer.fit(model, data_module)