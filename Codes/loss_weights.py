import torch
import torch.nn.functional as F
from constants import TRAIN_CLASS_NUM, VALID_CLASS_NUM, TEST_CLASS_NUM

def calculate_loss_weights(beta = 0.9999, num_classes = 2, set_indicator = 0):
    '''
    Calculates weights for class balanced loss from paper Class-Balanced Loss Based on Effective Number of Samples. 
    Modified from https://github.com/vandit15/Class-balanced-loss-pytorch/blob/master/class_balanced_loss.py
    
    Parameters:
        x (torch.tensor): Softmax outputs from model 
        y (torch.tensor): Class labels 0 = benign, 1 = Cancer
        num_classes (int): How many classes
        set_indicator (int): Tells which set is used. 0= train, 1= valid, 2=test. Needed in calculating scaling
        beta (float): Hyperparameter beta used in scaling
        loss (torch.nn): Classification loss function. Default is binary cross entropy
    '''
    #Check if train or validation loss is calculated
    if set_indicator==0:
        class_sizes = TRAIN_CLASS_NUM
    elif set_indicator==1:
        class_sizes = VALID_CLASS_NUM
    else:
        class_sizes = TEST_CLASS_NUM

    #scales
    scales = (1-beta)/(1 - torch.pow(beta, torch.tensor(class_sizes)))
    
    #scale scales into so that their sum is equal to num_classes
    scales = scales/torch.sum(scales) * num_classes
    
    return scales