from dataset import CustomDataset
from constants import NUM_IMAGES
from repVGG import repVGG
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def train(args):
    # set seed for ddp
    pl.seed_everything(args.seed)
    print(args.sample)
    #make dataloaders
    train_set = CustomDataset(spot_dir = args.train_spot_dir,
                              num_cancer = args.num_cancer,
                              num_benign = args.num_benign,
                              seed = args.seed,
                              include_edge = args.include_edge,
                              include_center=args.include_center,
                              sample_train = args.sample,
                              train_relapse = args.relapse_train
                             ) 
    
    valid_set = CustomDataset(spot_dir = args.valid_spot_dir,
                              num_cancer = args.num_cancer_val,
                              num_benign = args.num_benign_val,
                              seed = args.seed,
                              include_edge = args.include_edge_val,
                              include_center=args.include_center,
                              sample_validation = args.sample_val,
                              train_relapse = args.relapse_train
                             )
    
    trainloader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, drop_last=True)
    validloader = DataLoader(valid_set, batch_size=args.batch, shuffle =False, num_workers=args.num_workers)

    # model lr, model, set_size, batch_size, num_gpus, epochs, limit_batches, class_balance
    model = repVGG(lr = args.lr, 
                   model = args.model, 
                   batch_size = args.batch,
                   epochs = args.epochs, 
                   limit_batches = args.limit_batch, 
                   class_balance = args.class_balance,
                   pre_train = args.pre_train,
                   num_images = train_set.get_num_images(),
                   num_images_val = valid_set.get_num_images()
                  )

    # Logger for training
    wandb_logger = WandbLogger(name=args.run_name, project = args.project_name, save_dir=args.output_wandb)
    
    #Callbacks for training
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    early_stop_callback = EarlyStopping(
       monitor='acc_bal',
       patience=args.early_patience,
       mode = 'max'
    )
    
    #Checkpoint save the best models
    checkpoint_callback = ModelCheckpoint(
        monitor='acc_bal',
        dirpath=args.output_wandb+ args.filename_check,
        filename= args.filename_check + '-{epoch:02d}-{acc_bal:.2f}',
        save_top_k=3,
        mode='max',
    )
    
    trainer = pl.Trainer(progress_bar_refresh_rate = 25, 
                         max_epochs=args.epochs, 
                         logger=wandb_logger, 
                         gpus=args.num_gpus,
                         num_nodes=args.num_nodes,
                         limit_train_batches=args.limit_batch,
                         callbacks=[lr_monitor, early_stop_callback,checkpoint_callback],
                         accelerator='ddp',
                         plugins=DDPPlugin(find_unused_parameters=False),
                        )
    
    trainer.fit(model, trainloader, validloader)
    
    #load best model based on monitored metric in checkpoint_callback
    checkpoint_callback.best_model_path
    
    #save final model
    if model.global_rank == 0: #global_rank needed for multigpu runs???
        model_fname = '/data/atte/models/' + str(args.run_name) + '.pth'
        torch.save(model.state_dict(), model_fname)
        
    
if __name__ == '__main__':
    print('jotain testailua')