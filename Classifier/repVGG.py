import pytorch_lightning as pl
import torch
import timm 
from constants import BETA
from torch.optim.lr_scheduler import CosineAnnealingLR
from loss_weights import calculate_loss_weights
import math
from constants import TRAIN_CLASS_NUM, VALID_CLASS_NUM, TEST_CLASS_NUM
import numpy as np
from adjust_LR import adjust_LR
from metrics import calculate_metrics

class repVGG(pl.LightningModule):
    def __init__(self, lr, model, batch_size, epochs, limit_batches, class_balance, pre_train, num_images, num_images_val, w_decay=0.1, spectral= False, sd_lambda=0.1):
        super().__init__()
        self.model = timm.create_model(model, pretrained=pre_train, num_classes = 1)
        self.learning_rate = adjust_LR(lr,batch_size)
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.epochs = epochs
        self.class_balance = class_balance
        if limit_batches==1:
            self.train_loss_weights = calculate_loss_weights(num_images, beta = (sum(num_images)-1)/sum(num_images))
            
        
        elif limit_batches<1:
            self.train_loss_weights = calculate_loss_weights(limit_batches*np.array(num_images), 
                                                             beta=(limit_batches*sum(num_images)-1)/(limit_batches*sum(num_images)))
        
        elif limit_batches>1: 
            self.train_loss_weights = calculate_loss_weights(limit_batches* batch_size *np.array([0.2,0.8]),
                                                            beta=(limit_batches* batch_size-1)/(limit_batches* batch_size))
        
        self.validation_loss_weights = calculate_loss_weights(num_images_val, beta = (sum(num_images_val)-1)/sum(num_images_val))
        self.test_loss_weights = calculate_loss_weights(TEST_CLASS_NUM) 
        self.loss = torch.nn.functional.binary_cross_entropy_with_logits
        self.w_decay = w_decay
        self.spectral = spectral
        self.Lambda = sd_lambda
        
    def forward(self, x):
        # x shape
        x = self.model(x)
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.w_decay)
        scheduler = {
            'scheduler': CosineAnnealingLR(optimizer, 
                                           T_max = self.epochs
                                          )
        }
        return {
        'optimizer': optimizer,
        'lr_scheduler': scheduler}

    def training_step(self, train_batch, batch_idx):
        image, label = train_batch
        out = self.forward(image)
        if self.class_balance and self.spectral:
            loss = self.loss(out.squeeze(), label.float(), weight = self.train_loss_weights[label].to(label.device)) + self.Lambda * (out**2).mean()
        elif self.spectral:
            loss = self.loss(out.squeeze(), label.float()) + self.Lambda * (out**2).mean()
        else:
            loss = self.loss(out.squeeze(), label.float())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        image, label = val_batch
        out = self.forward(image)
        if self.class_balance and self.spectral:
            loss = self.loss(out.squeeze(), label.float(), weight = self.train_loss_weights[label].to(label.device)) + self.Lambda * (out**2).mean()
        elif self.spectral:
            loss = self.loss(out.squeeze(), label.float()) + self.Lambda * (out**2).mean()
        else:
            loss = self.loss(out.squeeze(), label.float())
        self.log('val_loss', loss)
        return (out, label)
    
    def validation_epoch_end(self, validation_step_outputs):
        all_metrics = {}
        outs = np.array([])
        labels = np.array([])
        for out, label in validation_step_outputs:
            # put outputs & labels into numpy array
            outs = np.append(outs, out.squeeze().cpu().numpy())
            labels = np.append(labels, label.squeeze().cpu().numpy())
        # calculate metrics for all predictions
        metrics = calculate_metrics(outs, labels)
        all_metrics.update(metrics)
        self.log_dict(all_metrics, sync_dist=True)

    def test_step(self, batch, batch_idx):
        image, label = batch
        out = self.model(image)
        if self.class_balance and self.spectral:
            loss = self.loss(out.squeeze(), label.float(), weight = self.train_loss_weights[label].to(label.device)) + self.Lambda * (out**2).mean()
        elif self.spectral:
            loss = self.loss(out.squeeze(), label.float()) + self.Lambda * (out**2).mean()
        else:
            loss = self.loss(out.squeeze(), label.float())
        self.log('test_loss', loss)
        return (out, label)

    def test_epoch_end(self, test_step_outputs):
        all_metrics = {}
        outs = np.array([])
        labels = np.array([])
        for out, label in test_step_outputs:
            # put outputs & labels into numpy array
            outs = np.append(outs, out.squeeze().cpu().numpy())
            labels = np.append(labels, label.squeeze().cpu().numpy())
        # calculate metrics for all predictions
        metrics = calculate_metrics(outs, labels)
        all_metrics.update(metrics)
        self.log_dict(all_metrics, sync_dist=True)
        
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        return self.forward(batch)
        
if __name__ == '__main__':
    print('jotain testailua')