'''
MEAN (list(float)): Means for normalization
STD (list(float)): Standard deviations for normalization
NUM_IMAGES (list(ints)): Sizes of validation, test and train sets
TRAIN_CLASS_NUM (list): List of class sizes (benign, cancer) in train set
VALID_CLASS_NUM (list): List of class sizes (benign, cancer) in validation set
BETA (float): Beta value for class_balanced loss
'''

MEAN = {'HBP' : [0.67607963, 0.55976693, 0.7969368 ],
        'TMA_WSI' : [0.81415012, 0.63717485, 0.79748713]
       }

STD = {'HBP' : [0.22521308, 0.24468836, 0.13823602],
       'TMA_WSI' : [0.15317514, 0.20072045, 0.130039]
      }

NUM_IMAGES = {'HBP' : [12399, 23291, 86388]}

#THESE ARE NOT NEEDED
#Checked. These are for sets that include edges to cancer label
#TRAIN_CLASS_NUM = [17969, 68419] 
VALID_CLASS_NUM = [2654, 9745]
TEST_CLASS_NUM = [5066, 18225]

#New train class balance. Edges classified as edges or normal images using trained classifier
TRAIN_CLASS_NUM = [20886, 63625] 

BETA = 0.9999

if __name__ == '__main__':
    print('jotain testailua')