import pytorch_lightning as pl
import torch
import timm 
from constants import BETA
from torch.optim.lr_scheduler import CosineAnnealingLR
from loss_weights import calculate_loss_weights
import math

class repVGG(pl.LightningModule):
    def __init__(self, lr, model, set_size, batch_size, num_gpus, epochs):
        super().__init__()
        self.model = timm.create_model(model, pretrained=False, num_classes = 1)
        self.learning_rate = lr
        self.save_hyperparameters()
        self.set_size = set_size
        self.batch_size = batch_size
        self.num_gpus = num_gpus
        self.epochs = epochs
        self.train_loss_weights = calculate_loss_weights()
        self.validation_loss_weights = calculate_loss_weights(set_indicator = 1)
        self.test_loss_weights = calculate_loss_weights(set_indicator = 2)
        self.loss = torch.nn.functional.binary_cross_entropy_with_logits
        
        
    def forward(self, x):
        # x shape
        x = self.model(x)
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = {
            'scheduler': CosineAnnealingLR(optimizer, 
                                           T_max = math.ceil(self.set_size / (self.batch_size * self.num_gpus) * self.epochs))
        }
        return {
        'optimizer': optimizer,
        'lr_scheduler': scheduler}

    def training_step(self, train_batch, batch_idx):
        image, label = train_batch
        out = self.forward(image)
        loss = self.loss(out.squeeze(), label.float(), weight = self.train_loss_weights[label].to(label.device))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        image, label = val_batch
        out = self.forward(image)
        loss = self.loss(out.squeeze(), label.float(), weight = self.validation_loss_weights[label].to(label.device))
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        image, label = batch
        out = self.model(image)
        loss = self.loss(out.squeeze(), label.float(), weight = self.test_loss_weights[label].to(label.device))
        self.log('test_loss', loss)
        return loss

if __name__ == '__main__':
    print('jotain testailua')