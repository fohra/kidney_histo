import pandas as pd
import torchvision.transforms as t
from torch.utils.data import Dataset
from PIL import Image
from constants import MEAN, STD
from gaussian_blur import GaussianBlur
import torch
import cv2


class CustomDataset(Dataset):
    def __init__(self, spot_dir, image_paths):
        '''
        Args:
        spot_dir (string): Path to excel file, that contains clinical info about the TMA spots
        pt_model (string): Metadata csv file from Histoprep, that contains only valid images for training and testing
        
        Outputs:
        image (torch.Tensor): Image as torch Tensor. Shape (1,3,512,512)
        label (torch.Tensor): Label indicating if there is cancer in the picture. 1=Cancer, 0=Benign 
        '''
        self.spot_infos = pd.read_csv(spot_dir)
        self.paths = pd.read_csv(image_paths, usecols=['path'])
        self.transformation = t.Compose([
                        t.ToPILImage(),
                        t.RandomResizedCrop(224),
                        t.RandomVerticalFlip(p=0.5),
                        t.RandomHorizontalFlip(p=0.5),
                        t.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.2,hue=0.1,),
                        GaussianBlur(p=0.2),
                        t.ToTensor(),
                        t.Normalize(MEAN['HBP'], STD['HBP']),
                    ])

    def __len__(self):
        return len(self.paths) 
    
    def __getitem__(self, idx):
        #load images
        path = self.paths.loc[idx].path
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transformation(image)
        
        #load labels
        #first need to find the spot where the image is from. The path contains that information
        tma_slide = int(path.split('/')[5].split('_')[1].split('-')[1])
        spot_num = int(path.split('/')[7].split('-')[1])
        label = self.spot_infos[(self.spot_infos['TMA num'] == tma_slide) & (self.spot_infos['own'] == spot_num)].Annotation.values[0]
        if label == 'Edge' or label=='Center':
            label = 1
        else:
            label = 0
        label = torch.tensor(label)
        
        return image, label