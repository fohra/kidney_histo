import pandas as pd
import torchvision.transforms as t
from torch.utils.data import Dataset
from PIL import Image
from constants import MEAN, STD
from gaussian_blur import GaussianBlur
from sample_infos import sample_infos
import torch
import simplejpeg
import numpy as np


class ValidationDataset(Dataset):
    def __init__(self, tma_spot_dir, num_cancer, num_benign, seed, num_relapse=0, num_non_relapse=0, include_edge = False, include_center=True, sample_validation=False, train_relapse = False, norm_mean_std = 'HBP', prob_gaussian=0.05, simple_transformation=False, use_soft=False, days_relapse = 10000):
        '''
        Args:
        spot_dir (string/pandas Dataframe): Path to excel file(or the file itself), that contains clinical info about the TMA spots
        num_cancer (int): Number of cancer images
        num_benign (int): Number of benign images
        seed (int): seed for sampling images
        include_edge (boolean): Whether to include edge spots during sampling. Note needs sampling flag True in order to work
        include_center (boolean): Whether to include center spots during sampling. Note needs sampling flag True in order to work
        sample_train (boolean): Whether to sample a subset of training data
        sample_validation (boolean): Whether to sample a subset of validation data
        prediction (boolean): If True uses fewer transformations. Used while predicting.
        relapse_train (boolean): Whether to train a classifier on relapse data.
        prob_gaussian (float): Probability for blurring images
        simple_transformation (bool): Whether to use minimal transformations
        use_soft (bool): Whether to use soft labels
        
        Outputs:
        image (torch.Tensor): Image as torch Tensor. Shape (1,3,512,512)
        label (torch.Tensor): Label indicating if there is cancer in the picture. 1=Cancer, 0=Benign 
        '''
        self.use_soft = use_soft
        self.days_relapse  =  days_relapse
        
        if (isinstance(tma_spot_dir, str)):
            self.tma_spot_infos = pd.read_csv(tma_spot_dir)
        else:
            raise Exception('Wrong type for spot_dirs. Pass either path to csv file or pandas Dataframe. Type was for tma ' + str(type(tma_spot_dir)))
        
        self.simple_transformation = simple_transformation
        self.relapse = train_relapse
        
        self.spot_infos = self.tma_spot_infos
        
        if sample_validation:
            self.spot_infos = sample_infos(infos = self.tma_spot_infos,
                                           num_cancer=num_cancer, 
                                           num_benign=num_benign,
                                           seed = seed,
                                           include_edge = include_edge,
                                           include_center=include_center,
                                           num_relapse=num_relapse, 
                                           num_non_relapse=num_non_relapse
                                          )
        
        if self.relapse:
            if self.days_relapse == 10000: #DONT TAKE days until relapse into account
                self.num_class_zero = len(self.spot_infos[(self.spot_infos['relapse'] == False)])
                self.num_class_one = len(self.spot_infos[(self.spot_infos['relapse'] == True)])
            else:
                self.num_class_zero = len(self.spot_infos[(self.spot_infos['relapse'] == False) | (self.spot_infos['Days_Relapse'] > self.days_relapse) | (self.spot_infos['Days_Relapse'].isna())])
                self.num_class_one = len(self.spot_infos[(self.spot_infos['relapse'] == True) & (self.spot_infos['Days_Relapse'] <= self.days_relapse)])
        else:
            self.num_class_zero = len(self.spot_infos[self.spot_infos['Annotation'] == 'Normal'])
            self.num_class_one = len(self.spot_infos[(self.spot_infos['Annotation'] == 'Center') | (self.spot_infos['Annotation'] == 'Edge')]) 
        
        # CHOOSE WHICH MEAN args.mean_std
        means = MEAN[norm_mean_std]
        stds = STD[norm_mean_std]
        
        
        if (self.simple_transformation):
            self.transformation = t.Compose([
                            t.ToTensor(),
                            t.Resize(224),
                            t.Normalize(means, stds),
                        ])
        else:
            self.transformation = t.Compose([
                            t.ToTensor(),
                            t.RandomResizedCrop(224),
                            t.RandomVerticalFlip(p=0.5),
                            t.RandomHorizontalFlip(p=0.5),
                            t.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.2,hue=0.1,),
                            t.ToPILImage(),
                            GaussianBlur(p=prob_gaussian),
                            t.ToTensor(),
                            t.Normalize(means, stds),
                        ])

    def __len__(self):
        return len(self.spot_infos) 
    
    def __getitem__(self, idx):
        #load images
        path = self.spot_infos.loc[idx].path
        #image = Image.open(path)
        #CODE FROM jopo666
        with open(path, 'rb') as f:
            image_bytes = np.frombuffer(f.read(), dtype='uint8')
        image = simplejpeg.decode_jpeg(image_bytes, colorspace='RGB')
        #CODE FROM jopo666 ends
        
        image = self.transformation(image)
        
        if self.relapse:
            label = self.spot_infos.loc[idx].relapse
            days= self.spot_infos.loc[idx].Days_Relapse
            if self.use_soft:
                if (label == True) & (days <= self.days_relapse):
                    label = torch.tensor([0,1])
                else:
                    label = torch.tensor([1,0])
            else:
                if (label == True) & (days <= self.days_relapse):
                    label = torch.tensor([1])
                else:
                    label = torch.tensor([0])
        else:
            if self.use_soft:
                label = self.spot_infos.loc[idx].Annotation
                if label == 'Edge' or label=='Center':
                    label = torch.tensor([0,1])
                else:
                    label = torch.tensor([1,0])
            else:
                label = self.spot_infos.loc[idx].Annotation
                if label == 'Edge' or label=='Center':
                    label = torch.tensor(1)
                else:
                    label = torch.tensor(0)
        
        return image, label.float()
    
    def get_num_images(self):
        '''
        Returns a list separating number of benign and cancer images for calculating class balanced loss
        '''
        return [self.num_class_zero, self.num_class_one]