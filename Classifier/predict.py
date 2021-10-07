import argparse
import numpy as np
from dataset import CustomDataset
from repVGG import repVGG
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

parser = argparse.ArgumentParser(description='Predict function for kidney cancer model', 
    usage='%(prog)s [optional arguments]', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#data files
parser.add_argument('--spot_dir', action='store', type=str, required=False, default='/data/datasets/RCC/all_wsi.csv', help='Path to csv file containing infos of images to be predicted')
parser.add_argument('--dummy_spot_dir', action='store', type=str, required=False, default='/data/datasets/RCC/all_wsi.csv', help='Path to csv file containing infos of dummy images')

#models
parser.add_argument('--model', action='store', type=str, required=False, default='resnet50', help='Model name from timm library')
parser.add_argument('--checkpoint', action='store', type=str, required=False, default='/data/atte/models/wandb/C_e60_LR25-5_GB5_CB_sd-4_pre/C_e60_LR25-5_GB5_CB_sd-4_pre-epoch=54-acc_bal=0.9758-auroc=0.9961.ckpt', help='Checkpoint for trained model')

#arguments for model
parser.add_argument('--batch', action='store', type=int, required=False, default=128, help='Batch size')

#arguments for hardware
parser.add_argument('--num_workers', action='store', type=int, required=False, default=8, help='Number of workers for loading data')
parser.add_argument('--num_gpus', action='store', type=int, required=False, default = 1, help='Number of gpus')
parser.add_argument('--num_nodes', action='store', type=int, required=False, default = 1, help='Number of nodes')

#arguments for dataset
parser.add_argument('--include_edge', action='store', type=bool, required=False, default = False, help='Whether to include edges into training')
parser.add_argument('--include_edge_val', action='store', type=bool, required=False, default = False, help='Whether to include edges into training')
parser.add_argument('--include_center', action='store', type=bool, required=False, default = True, help='Whether to include centers into training')
parser.add_argument('--sample', action='store', type=bool, required=False, default = False, help='Whether to sample images. If true samples num_cancer and num_benign amount of images.')

parser.add_argument('--num_cancer', action='store', type=int, required=False, default=34201, help='Number of cancer images to use')
parser.add_argument('--num_benign', action='store', type=int, required=False, default=17969, help='Number of cancer images to use')

parser.add_argument('--num_cancer_wsi', action='store', type=int, required=False, default=1385648, help='Number of cancer images to use')
parser.add_argument('--num_benign_wsi', action='store', type=int, required=False, default=267239, help='Number of cancer images to use')

parser.add_argument('--mean_std', action='store', type=str, required=False, default = 'WSI', help='Tells, which means and stds to use in normalization. Options: HBP, TMA_WSI, WSI')

parser.add_argument('--output_name', action='store', type=str, required=True, help='name for csv file. Dont include .csv!')


args = parser.parse_args()

#helper functions
def load_model(check_path):
    #intialize model
    model = repVGG(lr=0.001, 
               model=args.model, 
               batch_size=args.batch,
               epochs=50,
               limit_batches=1,
               class_balance=False,
               pre_train=True,
               num_images = [100,100],
               num_images_val = [100,100]
              )
    model = model.load_from_checkpoint(check_path)
    
    model = model.cuda()
    model.eval()
    
    return model

def sig(x):
    return 1/(1 + np.exp(-x))

#load dataset
dataset = CustomDataset(tma_spot_dir= args.spot_dir,
                        wsi_spot_dir = args.dummy_spot_dir,
                        num_cancer = 0, 
                        num_benign = 0,
                        seed = 43,
                        sample_train=False,
                        prediction = True,
                        norm_mean_std = args.mean_std
                        )

loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, drop_last=False)

model = load_model(args.checkpoint)

trainer = pl.Trainer(progress_bar_refresh_rate = 4, 
                         gpus=args.num_gpus,
                         num_nodes=args.num_nodes,
                         accelerator='ddp',
                         )

output = trainer.predict(model=model, dataloaders=loader)

print('Lenght of output')
print(len(output))

#output is a list of batch outputs. Next turn them into one list
li = np.array([])
for i in output:
    li = np.concatenate((li,i.cpu().numpy().flatten()))
    
print('Lenght of list')
print(len(li))

#Put logits into the dataframe & calculate probabilities
infos = dataset.spot_infos.copy()
infos['logits'] = li
#infos['probabilities'] = infos['logits'].apply(lambda x: sig(x))

#save the dataframe
save_dir = '/data/atte/data/' + args.output_name + '.csv'
infos.to_csv(save_dir, index=False)