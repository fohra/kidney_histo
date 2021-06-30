import pytorch_lightning as pl
import torch
import timm 
from torch.nn import functional as F # THIS MIGHT NOT BE NEEDED SINCE 

class Classifier(pl.LightningModule):
    def __init__(self, lr, model):
        super().__init__()
        self.model = timm.create_model(model, pretrained=False, num_classes = 2)
        self.learning_rate = lr
        self.save_hyperparameters()
        
    def forward(self, x):
        # x shape
        x = self.model(x)
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        image, label = train_batch
        out = self.forward(image)
        # NEED TO MAKE OWN LOSS
        loss = F.cross_entropy(out, label)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        image, label = val_batch
        out = self.forward(image)
        loss = F.cross_entropy(out, label)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        image, label = batch
        out = self.model(image)
        loss = F.cross_entropy(out, label)
        self.log('test_loss', loss)

if __name__ == '__main__':
    print('jotain testailua')