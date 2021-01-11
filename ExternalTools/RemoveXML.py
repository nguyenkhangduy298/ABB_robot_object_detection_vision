# This  program is used to remove .xml from file name in input dictionary
import argparse
import glob
import os

ap = argparse.ArgumentParser()
ap.add_argument('-f', '--folder', required=True, help='Left Camera: Port')
args = ap.parse_args()


folder_path = str(args.folder)

os.chdir(folder_path)

root = os.getcwd()
for file in glob.glob("*"):
    abs_path_file = os.path.abspath(file)
    if (".xml" in abs_path_file):
        new_name = abs_path_file.replace(".xml", "")
        os.rename(file, new_name)
        #print(abs_path_file)
