import torch

import torch.nn as nn
import lightning as L
import torch.nn.functional as F
from torchmetrics.classification import Accuracy, F1Score, MulticlassAUROC

from sc_vqvae.config import VQVAEClassificationConfig


class ClassificationHead(nn.Module):
    def __init__(self, num_patches, in_dim=128, num_classes=10, dropout_rate=0.5, num_layers=2):
        super(ClassificationHead, self).__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.num_patches = num_patches

        # flatten (N, T, D) to (N, T*D)
        self.flatten = nn.Sequential(
            nn.Linear(num_patches * in_dim, in_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Define the layers of the classification head
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == num_layers - 1:
                # Last layer should output num_classes
                self.layers.append(nn.Linear(in_dim, num_classes))
            else:
                # Intermediate layers
                self.layers.append(nn.Linear(in_dim, in_dim))
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(dropout_rate))
        
    def forward(self, x):
        """
        Forward pass through the classification head.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_patch, in_dim).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        # Flatten the input tensor

        x = x.view(x.size(0), -1)  # (batch_size, num_patch * in_dim)
        x = self.flatten(x)

        # Pass through the layers
        for layer in self.layers:
            x = layer(x)
        return x
    
class Classifier(nn.Module):
    def __init__(self, encoder, num_patches, in_dim=128, num_classes=10, dropout_rate=0.5, num_layers=2):
        super(Classifier, self).__init__()
        self.encoder = encoder
        self.head = ClassificationHead(
            num_patches=num_patches,
            in_dim=in_dim,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            num_layers=num_layers
        )

        for param in self.encoder.parameters():
            param.requires_grad = True
        
        ## Initialize the classification head with Kaiming He initialization
        for layer in self.head.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        
    def forward(self, x):
        """
        Forward pass through the classifier.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_dim).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        z_q, codes, vq_loss = self.encoder(x)
        x = self.head(z_q)
        return x, vq_loss

class ClassifierLit(L.LightningModule):

    def __init__(self, config: VQVAEClassificationConfig, load_encoder):
        super().__init__()
        self.config = config
        num_patches = (config.vqvae_config.gene_dim + config.vqvae_config.patch_size - 1) // config.vqvae_config.patch_size
        self.model = Classifier(
            encoder=load_encoder.encoder,
            num_patches=num_patches,
            in_dim=config.vqvae_config.embed_dim,
            num_classes=config.num_classes,
            dropout_rate=config.dropout_rate,
            num_layers=config.num_layers
        )
        self.criterion = nn.CrossEntropyLoss()

        # torchmetrics metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=config.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=config.num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=config.num_classes, average="macro")
        self.val_auc = MulticlassAUROC(num_classes=config.num_classes)

        self.test_acc = Accuracy(task="multiclass", num_classes=config.num_classes)
        self.test_f1 = F1Score(task="multiclass", num_classes=config.num_classes, average="macro")
        self.test_auc = MulticlassAUROC(num_classes=config.num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred, vq_loss = self.model(x)

        cls_loss = self.criterion(pred, y)

        loss = cls_loss + vq_loss

        self.log("cls_loss", cls_loss)
        self.log('vq_loss', vq_loss)
        # update and log
        self.train_acc.update(pred, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        # compute & reset metrics
        self.log("train_acc", self.train_acc.compute(), prog_bar=True)
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred, _ = self.model(x)
        loss = self.criterion(pred, y)

        self.val_acc.update(pred, y)
        self.val_f1.update(pred, y)
        self.val_auc.update(F.softmax(pred, dim=1), y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        self.log("val_acc", self.val_acc.compute(), prog_bar=True)
        self.log("val_f1", self.val_f1.compute())
        self.log("val_auc", self.val_auc.compute())

        # reset
        self.val_acc.reset()
        self.val_f1.reset()
        self.val_auc.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred, _ = self.model(x)
        self.test_acc.update(pred, y)
        self.test_f1.update(pred, y)
        self.test_auc.update(F.softmax(pred, dim=1), y)

    def on_test_epoch_end(self):
        self.log("test_acc", self.test_acc.compute())
        self.log("test_f1", self.test_f1.compute())
        self.log("test_auc", self.test_auc.compute())

        self.test_acc.reset()
        self.test_f1.reset()
        self.test_auc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config.patience,
            gamma=0.1
        )
        return [optimizer], [scheduler]