import pandas as pd
import torchvision.transforms as t
from torch.utils.data import Dataset
from PIL import Image
from constants import MEAN, STD
from gaussian_blur import GaussianBlur
from sample_infos import sample_infos
import torch


class CustomDataset(Dataset):
    def __init__(self, spot_dir, num_cancer, num_benign, seed, include_edge = False, include_center=True, sample_train=False, sample_validation=False, prediction=False):
        '''
        Args:
        spot_dir (string): Path to excel file, that contains clinical info about the TMA spots
        pt_model (string): Metadata csv file from Histoprep, that contains only valid images for training and testing
        
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
        if sample_train:
            self.spot_infos = sample_infos(infos = self.spot_infos,
                                           num_cancer = num_cancer,
                                           num_benign = num_benign,
                                           seed = seed,
                                           include_edge = include_edge,
                                           include_center=include_center
                                          )
        
        if sample_validation:
            self.spot_infos = sample_infos(infos = self.spot_infos,
                                           num_cancer=num_cancer, 
                                           num_benign=num_benign,
                                           seed = seed,
                                           include_edge = include_edge,
                                           include_center=include_center
                                          )
        if self.pred:
            self.num_benign = 0 # not used in prediction
            self.num_cancer = 0
        else:
            self.num_benign = len(self.spot_infos[self.spot_infos['Annotation'] == 'Normal'])
            self.num_cancer = len(self.spot_infos[(self.spot_infos['Annotation'] == 'Center') | (self.spot_infos['Annotation'] == 'Edge')])
        
        if self.pred:
            self.transformation = t.Compose([
                            t.Resize(224),
                            t.ToTensor(),
                            t.Normalize(MEAN['HBP'], STD['HBP']),
                        ])
        else:
            self.transformation = t.Compose([
                            t.RandomResizedCrop(224),
                            t.RandomVerticalFlip(p=0.5),
                            t.RandomHorizontalFlip(p=0.5),
                            t.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.2,hue=0.1,),
                            GaussianBlur(p=0.2),
                            t.ToTensor(),
                            t.Normalize(MEAN['HBP'], STD['HBP']),
                        ])

    def __len__(self):
        return len(self.spot_infos) 
    
    def __getitem__(self, idx):
        #load images
        path = self.spot_infos.loc[idx].path
        image = Image.open(path)
        image = self.transformation(image)
        
        if self.pred:
            label = 0 # not used in prediction
        else:
            #load labels
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
        return [self.num_benign, self.num_cancer]