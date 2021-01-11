# This  program is used to create train.txt and test.txt from a dictionary

import argparse
import glob
import os
import random

ap = argparse.ArgumentParser()
ap.add_argument('-f', '--folder', required=True, help='Folder')
ap.add_argument('-train', '--train', required=True, help='Train')
ap.add_argument('-test', '--test', required=True, help='Test')
ap.add_argument('-rate', '--rate', required=True, help='Percentage of test data, must less than 100')


args = ap.parse_args()


folder_path = str(args.folder)
train_path = str(args.train)
test_path = str(args.test)
rate_string = int(args.rate)


# Create file (blank):
# Create train.txt and valid.txt
file_train = open(train_path, 'w')
file_test = open(test_path, 'w')




os.chdir(folder_path)

root = os.getcwd()
for file in glob.glob("*"):

    number = random.randint(0,100)
    abs_path_file = os.path.abspath(file)
    if (number<= rate_string):
        # Test:
        file_test.write(abs_path_file +"\n")
    else:
        file_train.write(abs_path_file + "\n")
        #print(abs_path_file)
