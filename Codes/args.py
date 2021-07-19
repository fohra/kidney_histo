import argparse

parser = argparse.ArgumentParser(description='Train function for repVGG model', 
    usage='%(prog)s [optional arguments]', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#optional 
parser.add_argument('--lr', action='store', type=float, required=False, default=0.001, help='Learning rate for optimizer')
parser.add_argument('--seed', action='store', type=int, required=False, default=1442, help='Seed for splitting data')
parser.add_argument('--batch', action='store', type=int, required=False, default=128, help='Batch size')

parser.add_argument('--train_spot_dir', action='store', type=str, required=False, default='/data/datasets/RCC/HE_cut/train_infos.csv'
                    , help='Path to excel file containing information about train spots')
parser.add_argument('--valid_spot_dir', action='store', type=str, required=False, default='/data/datasets/RCC/HE_cut/valid_infos.csv'
                    , help='Path to excel file containing information about validation spots')
parser.add_argument('--test_spot_dir', action='store', type=str, required=False, default='/data/datasets/RCC/HE_cut/test_infos.csv'
                    , help='Path to excel file containing information about test spots')

parser.add_argument('--num_workers', action='store', type=int, required=False, default=8, help='Number of workers for loading data')
parser.add_argument('--model', action='store', type=str, required=False, default='repvgg_b1g4', help='Model name from timm library')
parser.add_argument('--epochs', action='store', type=int, required=True, help='Number of epochs. REQUIRED')
parser.add_argument('--run_name', action='store', type=str, required=True, help='Name for WandB run. REQUIRED')
parser.add_argument('--project_name', action='store', type=str, required=False, default = 'dippa', help='Name for WandB project')
parser.add_argument('--num_gpus', action='store', type=int, required=False, default = 1, help='Number of gpus')
parser.add_argument('--num_nodes', action='store', type=int, required=False, default = 1, help='Number of nodes')
parser.add_argument('--limit_batch', action='store', type=float, required=False, default = 1.0, help='Limit number of training and validation batches')

parser.add_argument('--class_balance', action='store', type=bool, required=False, default = False, help='Whether to use class balanced loss')
parser.add_argument('--include_edge', action='store', type=bool, required=False, default = False, help='Whether to include edges into training')
parser.add_argument('--sample', action='store', type=bool, required=False, default = False, help='Whether to sample images. If true samples num_cancer and num_benign amount of images.')


parser.add_argument('--num_cancer', action='store', type=int, required=False, default=34201, help='Number of cancer images to use')
parser.add_argument('--num_benign', action='store', type=int, required=False, default=17969, help='Number of cancer images to use')

parser.add_argument('--pre_train', action='store', type=bool, required=False, default = False, help='Whether to use pre_trained network as initialization.')

args = parser.parse_args()

if __name__ == '__main__':
    print('Jotain testailua')