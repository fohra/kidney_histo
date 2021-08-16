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


class CustomDataset(Dataset):
    def __init__(self, spot_dir, num_cancer, num_benign, seed, num_relapse=0, num_non_relapse=0, include_edge = False, include_center=True, sample_train=False, sample_validation=False, prediction=False, train_relapse = False, norm_mean_std = 'HBP'):
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
        
        Outputs:
        image (torch.Tensor): Image as torch Tensor. Shape (1,3,512,512)
        label (torch.Tensor): Label indicating if there is cancer in the picture. 1=Cancer, 0=Benign 
        '''
        if isinstance(spot_dir, str):
            self.spot_infos = pd.read_csv(spot_dir)
        elif isinstance(spot_dir, pd.core.frame.DataFrame):
            self.spot_infos = spot_dir
        else:
            raise Exception('Wrong type for spot_dir. Pass either path to csv file or pandas Dataframe. Type was ' + str(type(spot_dir)))
        
        self.pred = prediction
        self.relapse = train_relapse
        if sample_train:
            self.spot_infos = sample_infos(infos = self.spot_infos,
                                           num_cancer = num_cancer,
                                           num_benign = num_benign,
                                           seed = seed,
                                           include_edge = include_edge,
                                           include_center=include_center,
                                           num_relapse=num_relapse, 
                                           num_non_relapse=num_non_relapse
                                          )
        
        if sample_validation:
            self.spot_infos = sample_infos(infos = self.spot_infos,
                                           num_cancer=num_cancer, 
                                           num_benign=num_benign,
                                           seed = seed,
                                           include_edge = include_edge,
                                           include_center=include_center,
                                           num_relapse=num_relapse, 
                                           num_non_relapse=num_non_relapse
                                          )
        if self.pred:
            self.num_class_zero = 0 # not used in prediction
            self.num_class_one = 0
        else:
            if self.relapse:
                self.num_class_zero = len(self.spot_infos[self.spot_infos['relapse'] == False])
                self.num_class_one = len(self.spot_infos[self.spot_infos['relapse'] == True])
            else:
                self.num_class_zero = len(self.spot_infos[self.spot_infos['Annotation'] == 'Normal'])
                self.num_class_one = len(self.spot_infos[(self.spot_infos['Annotation'] == 'Center') | (self.spot_infos['Annotation'] == 'Edge')]) 
        
        # CHOOSE WHICH MEAN args.mean_std
        means = MEAN[norm_mean_std]
        stds = STD[norm_mean_std]
        
        
        if self.pred:
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
                            GaussianBlur(p=0.2),
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
        
        #load labels 
        if self.pred:
            label = 0 # not used while predicting
        else:
            if self.relapse:
                label = self.spot_infos.loc[idx].relapse
                if label == True:
                    label = 1
                else:
                    label = 0
            else:
                label = self.spot_infos.loc[idx].Annotation
                if label == 'Edge' or label=='Center':
                    label = 1
                else:
                    label = 0
        
        label = torch.tensor(label)
        
        return image, label
    
    def get_num_images(self):
        '''
        Returns a list separating number of benign and cancer images for calculating class balanced loss
        '''
        return [self.num_class_zero, self.num_class_one]