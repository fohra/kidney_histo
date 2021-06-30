import torch
import torch.nn as nn
from constants import TOTAL_NUM, CLASS_NUM

def class_balanced_loss(x,  y, beta, num_classes = 2, loss = nn.BCELoss()):
    '''
    x (torch.tensor): Softmax outputs from model 
    y (torch.tensor): Class labels 0 = benign, 1 = Cancer
    beta (float): Hyperparameter beta used in scaling
    loss (torch.nn): Classification loss function. Default is binary cross entropy
    '''
    #scales
    scales = (1-beta)/(1 - torch.pow(beta, torch.tensor(CLASS_NUM)))

    #scale scales into so that their sum is equal to num_classes
    scales = scales/torch.sum(scales) * num_classes

    

    #turn weight into correct shape

    #softmax

    # bceloss
    #Labels one hot. For bceloss?
    one_hot = F.one_hot(y, num_classes).float()
    x = x.softmax() #check dim
    cb_loss = scales * loss(x, one_hot)

    return cb_loss