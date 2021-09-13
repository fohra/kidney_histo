import argparse

parser = argparse.ArgumentParser(description='Train function for repVGG model', 
    usage='%(prog)s [optional arguments]', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#optional 
parser.add_argument('--lr', action='store', type=float, required=False, default=0.001, help='Learning rate for optimizer')
parser.add_argument('--seed', action='store', type=int, required=False, default=1442, help='Seed for splitting data')
parser.add_argument('--batch', action='store', type=int, required=False, default=128, help='Batch size')

parser.add_argument('--train_spot_dir', action='store', type=str, required=False, default='/data/datasets/RCC/HE_cut/train_infos.csv', help='Path to excel file containing information about train spots')

parser.add_argument('--train_wsi_spot_dir', action='store', type=str, required=False, default='/data/atte/data/confident1%_train_only_wsi_labels_ids.csv', help='Path to excel file containing information about WSI train spots')

parser.add_argument('--valid_spot_dir', action='store', type=str, required=False, default='/data/datasets/RCC/HE_cut/valid_infos.csv', help='Path to excel file containing information about validation spots')

parser.add_argument('--test_spot_dir', action='store', type=str, required=False, default='/data/datasets/RCC/HE_cut/test_infos.csv', help='Path to excel file containing information about test spots')

parser.add_argument('--num_workers', action='store', type=int, required=False, default=8, help='Number of workers for loading data')
parser.add_argument('--model', action='store', type=str, required=False, default='resnet50', help='Model name from timm library')
parser.add_argument('--epochs', action='store', type=int, required=True, help='Number of epochs. REQUIRED')
parser.add_argument('--run_name', action='store', type=str, required=True, help='Name for WandB run. REQUIRED')
parser.add_argument('--project_name', action='store', type=str, required=False, default = 'dippa', help='Name for WandB project')
parser.add_argument('--num_gpus', action='store', type=int, required=False, default = 1, help='Number of gpus')
parser.add_argument('--num_nodes', action='store', type=int, required=False, default = 1, help='Number of nodes')
parser.add_argument('--limit_batch', action='store', type=float, required=False, default = 1.0, help='Limit number of training and validation batches')
parser.add_argument('--weight_decay', action='store', type=float, required=False, default = 0.00001, help='weight decay for optimizer')

parser.add_argument('--class_balance', action='store', type=bool, required=False, default = False, help='Whether to use class balanced loss')
parser.add_argument('--include_edge', action='store', type=bool, required=False, default = False, help='Whether to include edges into training')
parser.add_argument('--include_edge_val', action='store', type=bool, required=False, default = False, help='Whether to include edges into training')
parser.add_argument('--include_center', action='store', type=bool, required=False, default = True, help='Whether to include centers into training')
parser.add_argument('--sample', action='store', type=bool, required=False, default = False, help='Whether to sample images. If true samples num_cancer and num_benign amount of images.')
parser.add_argument('--sample_val', action='store', type=bool, required=False, default = False, help='Whether to sample validation images.')


parser.add_argument('--num_cancer', action='store', type=int, required=False, default=34201, help='Number of cancer images to use')
parser.add_argument('--num_benign', action='store', type=int, required=False, default=17969, help='Number of cancer images to use')

parser.add_argument('--num_cancer_wsi', action='store', type=int, required=False, default=1385648, help='Number of cancer images to use')
parser.add_argument('--num_benign_wsi', action='store', type=int, required=False, default=267239, help='Number of cancer images to use')

parser.add_argument('--num_cancer_val', action='store', type=int, required=False, default=4976, help='Number of cancer images to use in validation')
parser.add_argument('--num_benign_val', action='store', type=int, required=False, default=2654, help='Number of cancer images to use in validation')

parser.add_argument('--num_relapse', action='store', type=int, required=False, default=0, help='Number of cancer images to use')
parser.add_argument('--num_non_relapse', action='store', type=int, required=False, default=0, help='Number of cancer images to use')

parser.add_argument('--pre_train', action='store', type=bool, required=False, default = False, help='Whether to use pre_trained network as initialization.')
parser.add_argument('--output_wandb', action='store', type=str, required=False, default = '/data/atte/models/wandb/', help='Where to save wandb logs')
parser.add_argument('--filename_check', action='store', type=str, required=False, help='Filename for checkpoints')
parser.add_argument('--early_patience', action='store', type=int, required=False, default = 1000, help='Early stopping patience. Tells how many epochs to train with no improvement.')
parser.add_argument('--relapse_train', action='store', type=bool, required=False, default = False, help='Whether to train a relapse classifier.')
parser.add_argument('--mean_std', action='store', type=str, required=False, default = 'HBP', help='Tells, which means and stds to use in normalization. Options: HBP, TMA_WSI')

parser.add_argument('--prob_gaussian', action='store', type=float, required=False, default = 0.05, help='Probability for blurring images')

parser.add_argument('--spectral', action='store', type=bool, required=False, default = False, help='Whether to use spectral decoupling.')
parser.add_argument('--sd_lambda', action='store', type=float, required=False, default = 0.1, help='Lambda coefficient for spectral decoupling.')

parser.add_argument('--use_SAM', action='store', type=bool, required=False, default = False, help='Whether to use Sharpness Aware Minimization optimizer(SAM).')
parser.add_argument('--sam_rho', action='store', type=float, required=False, default = 0.05, help='Rho coefficient for Sharpness Aware Minimization optimizer(SAM).')

parser.add_argument('--drop', action='store', type=float, required=False, default = 0.00, help='Dropout rate for model')
parser.add_argument('--drop_path', action='store', type=float, required=False, default = 0.00, help='Drop path rate for model')

args = parser.parse_args()

if __name__ == '__main__':
    print('Jotain testailua')