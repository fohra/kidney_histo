'''
MEAN (list(float)): Means for normalization
STD (list(float)): Standard deviations for normalization
NUM_IMAGES (list(ints)): Sizes of validation, test and train sets
TOTAL_NUM (int): Total number of images in the dataset
CLASS_NUM (list)
'''

MEAN = {'HBP' : [0.67607963, 0.55976693, 0.7969368 ]}

STD = {'HBP' : [0.22521308, 0.24468836, 0.13823602]}

NUM_IMAGES = {'HBP' : [12207, 24416, 85455]}

TOTAL_NUM = 0 #check
CLASS_NUM = [0,0] #check

if __name__ == '__main__':
    print('jotain testailua')