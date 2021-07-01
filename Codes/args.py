import argparse

parser = argparse.ArgumentParser(description='Train function for repVGG model', 
    usage='%(prog)s [optional arguments]', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#optional 
parser.add_argument('--lr', action='store', type=float, required=False, default=0.001, help='Learning rate for optimizer')
parser.add_argument('--seed', action='store', type=int, required=False, default=42, help='Seed for splitting data')
parser.add_argument('--batch', action='store', type=int, required=False, default=16, help='Batch size')
parser.add_argument('--spot_dir', action='store', type=str, required=False, default='/home/fohratte/data/final_spot_infos.xlsx'
                    , help='Path to excel file containing information about spots')
parser.add_argument('--image_paths', action='store', type=str, required=False, default='/home/fohratte/data/metadata.csv'
                    , help='Path to csv file containing paths to images')
parser.add_argument('--num_workers', action='store', type=int, required=False, default=8, help='Number of workers for loading data')
parser.add_argument('--model', action='store', type=str, required=False, default='repvgg_a2', help='Model name from timm library')
parser.add_argument('--epochs', action='store', type=int, required=True, help='Number of epochs. REQUIRED')
parser.add_argument('--run_name', action='store', type=str, required=True, help='Name for WandB run. REQUIRED')


args = parser.parse_args()

if __name__ == '__main__':
    print('Jotain testailua')