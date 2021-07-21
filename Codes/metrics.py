from sklearn import metrics
import torch

def calculate_metrics(preds, targets):
    '''
    Code from Joona Pohjonen, Github @jopo666
    Calculates accuracies, balanced accuracies, precisions, recalls and auroc for predictions and targets
    '''
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
        targets = targets.cpu().numpy()
    binary_preds = preds >= 0.5
    
    # Calculate metrics.
    acc = metrics.accuracy_score(targets, binary_preds)
    bal_acc = metrics.balanced_accuracy_score(targets, binary_preds)
    neg_prec = metrics.precision_score(targets, binary_preds, 
                                       pos_label=0, zero_division=0)
    pos_prec = metrics.precision_score(targets, binary_preds, 
                                       pos_label=1, zero_division=0)
    neg_recall = metrics.recall_score(targets, binary_preds, 
                                      pos_label=0, zero_division=0)
    pos_recall = metrics.recall_score(targets, binary_preds, 
                                      pos_label=1, zero_division=0)
    if (targets.sum() > 0) and not (targets.sum() == len(targets)):
        auroc = metrics.roc_auc_score(targets, preds)
    else:
        auroc = 0
        
    results = {
        'acc': acc,
        'acc_bal': bal_acc,
        'prec_neg': neg_prec,
        'prec_pos': pos_prec,
        'rec_neg': neg_recall,
        'rec_pos': pos_recall,
        'auroc': auroc,
    }
    return results