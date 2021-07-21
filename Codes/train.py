from dataset import CustomDataset
from constants import NUM_IMAGES
from repVGG import repVGG
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

def train(args):
    # set seed for ddp
    pl.seed_everything(args.seed)
    
    #make dataloaders
    train_set = CustomDataset(spot_dir = args.train_spot_dir,
                              num_cancer = args.num_cancer,
                              num_benign = args.num_benign,
                              seed = args.seed,
                              include_edge = args.include_edge,
                              sample = args.sample
                             )
    
    valid_set = CustomDataset(spot_dir = args.valid_spot_dir,
                              num_cancer = args.num_cancer,
                              num_benign = args.num_benign,
                              seed = args.seed,
                              include_edge = args.include_edge,
                              sample_val = args.sample_val
                             )
    
    trainloader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, drop_last=True)
    validloader = DataLoader(valid_set, batch_size=args.batch, shuffle =False, num_workers=args.num_workers)

    # model lr, model, set_size, batch_size, num_gpus, epochs, limit_batches, class_balance
    model = repVGG(lr = args.lr, 
                   model = args.model, 
                   set_size = len(trainloader), 
                   batch_size = args.batch, 
                   num_gpus = args.num_gpus, 
                   num_nodes = args.num_nodes,
                   epochs = args.epochs, 
                   limit_batches = args.limit_batch, 
                   class_balance = args.class_balance,
                   pre_train = args.pre_train
                  )

    # training
    wandb_logger = WandbLogger(name=args.run_name, project = args.project_name)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    trainer = pl.Trainer(progress_bar_refresh_rate = 25, 
                         max_epochs=args.epochs, 
                         logger=wandb_logger, 
                         gpus=args.num_gpus,
                         num_nodes=args.num_nodes,
                         limit_train_batches=args.limit_batch,
                         callbacks=[lr_monitor],
                         accelerator='ddp')
    
    trainer.fit(model, trainloader, validloader)

    #save final model
    if model.global_rank == 0: #global_rank needed for multigpu runs???
        model_fname = '/data/models/kidney/' + str(args.run_name) + '.pth'
        torch.save(model.state_dict(), model_fname)
        
    
if __name__ == '__main__':
    print('jotain testailua')