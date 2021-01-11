import argparse
import glob
import os

ap = argparse.ArgumentParser()
ap.add_argument('-ido', '--IdOriginal', default=3, required=True, help='ID: Original')
ap.add_argument('-idr', '--IdReplace', default=0, required=True, help='ID:')
ap.add_argument('-f', '--Folder', required=True, help='Left Camera: Port')

args = ap.parse_args()

# Get the old - new id from the args
id_origin = int(args.IdOriginal)
id_new = int(args.IdReplace)

# Get the folder name
folder_name = str(args.Folder)

# Scan the folder
os.chdir(folder_name)  # Change os to the directory (must have)
# Access each text file in the directory "folder_name"
for file in glob.glob("*.txt"):
    abs_path_file = os.path.abspath(file)

    # Open one file
    print("Processing " + str(abs_path_file))
    content = open(abs_path_file, "r").read()

    line_array = []
    line = ""  # blank lines

    for char in content:
        if (char != "\n"):
            line = line + char  # If not new line, then append new char to the blank line
        else:
            line_array.append(line)
            line = ""

    print(line_array)
    new_array = []
    for line in line_array:
        if (int(line[0]) == id_origin):
            new_line = str(id_new) + line[1::]
        else:
            new_line = line
        new_array.append(new_line)
    print(new_array)

    write_file = open(abs_path_file, "w")
    for line in new_array:
        write_file.write(line)
        write_file.write("\n")
    write_file.close()

#
# print(content, end="")
