import argparse

parser = argparse.ArgumentParser(description='Train function for repVGG model', 
    usage='%(prog)s [optional arguments]', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#optional 
parser.add_argument('--lr', action='store', type=float, required=False, default=0.001, help='Learning rate for optimizer')
parser.add_argument('--seed', action='store', type=int, required=False, default=42, help='Seed for splitting data')
parser.add_argument('--batch', action='store', type=int, required=False, default=16, help='Batch size')

parser.add_argument('--train_spot_dir', action='store', type=str, required=False, default='/data/datasets/RCC/HE_cut/train_spots.csv'
                    , help='Path to excel file containing information about train spots')
parser.add_argument('--valid_spot_dir', action='store', type=str, required=False, default='/data/datasets/RCC/HE_cut/valid_spots.csv'
                    , help='Path to excel file containing information about validation spots')
parser.add_argument('--test_spot_dir', action='store', type=str, required=False, default='/data/datasets/RCC/HE_cut/test_spots.csv'
                    , help='Path to excel file containing information about test spots')

parser.add_argument('--train_image_paths', action='store', type=str, required=False, default='/data/datasets/RCC/HE_cut/train_paths.csv'
                    , help='Path to csv file containing paths to train images')
parser.add_argument('--valid_image_paths', action='store', type=str, required=False, default='/data/datasets/RCC/HE_cut/valid_paths.csv'
                    , help='Path to csv file containing paths to validation images')
parser.add_argument('--test_image_paths', action='store', type=str, required=False, default='/data/datasets/RCC/HE_cut/test_paths.csv'
                    , help='Path to csv file containing paths to test images')

parser.add_argument('--num_workers', action='store', type=int, required=False, default=8, help='Number of workers for loading data')
parser.add_argument('--model', action='store', type=str, required=False, default='repvgg_a2', help='Model name from timm library')
parser.add_argument('--epochs', action='store', type=int, required=True, help='Number of epochs. REQUIRED')
parser.add_argument('--run_name', action='store', type=str, required=True, help='Name for WandB run. REQUIRED')
parser.add_argument('--num_gpus', action='store', type=int, required=False, default = 1, help='Number of gpus')
parser.add_argument('--num_nodes', action='store', type=int, required=False, default = 1, help='Number of nodes')



args = parser.parse_args()

if __name__ == '__main__':
    print('Jotain testailua')