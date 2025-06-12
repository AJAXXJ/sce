# ---encoding:utf-8---
# @Time    : 2025/6/5 10:49
# @Author  : AJAXXJ
# @Email   : 1751353695@qq.com
# @Project : sce-plus
# @Software: PyCharm
import torch.nn as nn
import lightning as L
from torch import optim
from torchmetrics import Accuracy

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=(512, 256, 128)):
        super(MLPClassifier, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU()
        )
        self.classifier = nn.Linear(hidden_dims[2], num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits

class MLPClassifierLit(L.LightningModule):

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.model = MLPClassifier(
            input_dim=input_dim,
            num_classes=num_classes
        )
        self.criterion = nn.CrossEntropyLoss()
        self.lr = 1e-4

        # 评估指标
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.train_acc(logits, y)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.val_acc(logits, y)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.test_acc(logits, y)

        self.log("test_loss", loss, prog_bar=True, on_epoch=True)
        self.log("test_acc", acc, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)