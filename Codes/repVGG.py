import pytorch_lightning as pl
import torch
import timm 
from torch.nn import functional as F # THIS MIGHT NOT BE NEEDED SINCE 
from class_balanced_loss import class_balanced_loss
from constants import BETA
from torch.optim.lr_scheduler import CosineAnnealingLR
import math

class repVGG(pl.LightningModule):
    def __init__(self, lr, model, set_size, batch_size, num_gpus, epochs):
        super().__init__()
        self.model = timm.create_model(model, pretrained=False, num_classes = 2)
        self.learning_rate = lr
        self.save_hyperparameters()
        self.set_size = set_size
        self.batch_size = batch_size
        self.num_gpus = num_gpus
        self.epochs = epochs
        
    def forward(self, x):
        # x shape
        x = self.model(x)
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return {
        'optimizer': optimizer,
        'lr_scheduler': {
            'scheduler': CosineAnnealingLR(optimizer, 
                                           T_max = math.ceil(self.set_size / (self.batch_size * self.num_gpus) * self.epochs),
             'strict': False
            }

    def training_step(self, train_batch, batch_idx):
        image, label = train_batch
        out = self.forward(image)
        loss = class_balanced_loss(out, label, BETA)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        image, label = val_batch
        out = self.forward(image)
        loss = class_balanced_loss(out, label, BETA, set_indicator=1)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        image, label = batch
        out = self.model(image)
        loss = class_balanced_loss(out, label, BETA, set_indicator=2)
        self.log('test_loss', loss)

if __name__ == '__main__':
    print('jotain testailua')