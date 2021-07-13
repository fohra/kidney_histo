from dataset import CustomDataset
from constants import NUM_IMAGES
from repVGG import repVGG
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

def train(args):
    # set seed for ddp
    
    #make dataloaders
    train_set = CustomDataset(args.train_spot_dir, args.train_image_paths)
    valid_set = CustomDataset(args.valid_spot_dir, args.valid_image_paths)
    
    trainloader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=args.num_workers)
    validloader = DataLoader(valid_set, batch_size=args.batch, shuffle =False, num_workers=args.num_workers)

    # model
    model = repVGG(args.lr, args.model, len(train_set), args.batch, args.num_gpus, args.epochs)

    # training
    wandb_logger = WandbLogger(name=args.run_name)
    trainer = pl.Trainer(progress_bar_refresh_rate = 25, 
                         max_epochs=args.epochs, 
                         logger=wandb_logger, 
                         gpus=args.num_gpus,
                         num_nodes=args.num_nodes,
                         accelerator='ddp')
    trainer.fit(model, trainloader, validloader)

    #save final model
    if model.global_rank == 0: #global_rank needed for multigpu runs???
        model_fname = 'models/' + str(args.run_name) + '.pth'
        torch.save(model.state_dict(), model_fname)
        
    
if __name__ == '__main__':
    print('jotain testailua')