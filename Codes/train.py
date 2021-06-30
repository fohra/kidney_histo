from dataset import CustomDataset
from constants import NUM_IMAGES
from repVGG import Classifier
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

def train(args):
    ds = CustomDataset(args.spot_dir, args.image_paths)
    val_set, test_set, train_set = torch.utils.data.random_split(ds, [NUM_IMAGES[0],NUM_IMAGES[1], 
                                                                NUM_IMAGES[2]], generator=torch.Generator().manual_seed(args.seed))

    trainloader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=args.num_workers)
    validloader = DataLoader(val_set, batch_size=args.batch, shuffle =False, num_workers=args.num_workers)
    #testloader = DataLoader(test_set, batch_size=args.batch, shuffle=False, num_workers=args.num_workers) #THIS MIGHT NOT BE NEEDED HERE

    # model
    model = Classifier(args.lr, args.model)

    # training
    wandb_logger = WandbLogger(name=args.run_name)
    trainer = pl.Trainer(progress_bar_refresh_rate = 50, max_epochs=1, logger=wandb_logger, gpus=1)
    trainer.fit(model, trainloader, validloader)

if __name__ == '__main__':
    print('jotain testailua')