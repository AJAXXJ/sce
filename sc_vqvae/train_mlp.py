# ---encoding:utf-8---
# @Time    : 2025/6/5 12:58
# @Author  : AJAXXJ
# @Email   : 1751353695@qq.com
# @Project : sce-plus
# @Software: PyCharm
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger

from config import VQVAEClassificationConfig
from data.data_module import VQVAEClassificationDataModule
from models.mlp import MLPClassifierLit

mlflow_logger = MLFlowLogger(
    experiment_name="sce",
    tags={
        '模型': 'MLPClassifier',
        '任务': 'mlp-classifier',

    },
    run_name = 'MLPClassifier',
    tracking_uri="http://172.29.172.33:5000/"
)

config = VQVAEClassificationConfig()
# 设置随机数
config.set_seed()

checkpoint_callback = ModelCheckpoint(
    dirpath="./checkpoints/mlp",         # 检查点保存目录
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
data_module = VQVAEClassificationDataModule(config)
data_module.setup()
config.gene_dim = data_module.gene_dim
config.num_classes = data_module.num_classes

model = MLPClassifierLit(input_dim=config.gene_dim, num_classes=config.num_classes)

trainer.fit(model, data_module)

# trainer.test(model, data_module)