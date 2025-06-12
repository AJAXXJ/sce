import os.path

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger

from config import VQVAEClassificationConfig
from data.data_module import VQVAEClassificationDataModule
from models.classification import ClassifierLit
from utils import load_vqvae_encoder

config = VQVAEClassificationConfig()
config.set_seed()

mlflow_logger = MLFlowLogger(
    experiment_name="sce",
    tags={
        '模型': 'VQVAE-Classification',
        '任务': 'classifier',
    },
    run_name='VQVAEClassification',
    tracking_uri="http://172.29.172.33:5000/"
)

checkpoint_callback = ModelCheckpoint(
    dirpath=f"./checkpoints/classifier_model",  # 检查点保存目录
    filename="{epoch:02d}-{val_loss:.2f}",  # 保存文件名格式
    monitor="val_loss",  # 监控验证损失（确保模型中有相应的 metric）
    mode="min",  # loss 越小越好
    save_top_k=3,  # 最多保存 3 个最佳检查点
    save_last=True  # 同时保存最后一次模型
)

# 加入 EarlyStopping 回调
early_stop_callback = EarlyStopping(
    monitor="val_loss",  # 监控验证损失
    patience=config.patience,  # config 中设定等待多少个 epoch 无提升就停止
    mode="min",  # 越小越好
    verbose=True  # 输出日志
)

# 配置 Trainer
trainer = Trainer(
    default_root_dir="./",
    check_val_every_n_epoch=config.eval_epoch_step,
    max_epochs=config.epochs,
    callbacks=[checkpoint_callback, early_stop_callback],
    # logger=mlflow_logger
)

data_module = VQVAEClassificationDataModule(config)
data_module.setup()
config.num_classes = data_module.num_classes
config.vqvae_config.gene_dim = data_module.gene_dim

model = ClassifierLit(config, load_encoder=load_vqvae_encoder(config))

trainer.fit(model, data_module)

# trainer.test(model, data_module)


