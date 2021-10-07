import torch
import torch.nn.functional as F

def calculate_loss_weights(class_sizes, beta = 0.9999, num_classes = 2):
    '''
    Calculates weights for class balanced loss from paper Class-Balanced Loss Based on Effective Number of Samples. 
    Modified from https://github.com/vandit15/Class-balanced-loss-pytorch/blob/master/class_balanced_loss.py
    
    Parameters:
        class_sizes (list): list of class sizes
        beta (float): coefficient for Class balanced loss
        num_classes (int): How many classes
        
    '''
    #scales
    scales = (1-beta)/(1 - torch.pow(beta, torch.tensor(class_sizes)))
    
    #scale scales into so that their sum is equal to num_classes
    scales = scales/torch.sum(scales) * num_classes
    
    return scales