import os
import shutil

## YES and NO PATHS
path_yes = "./Train/Yes"
path_no = "./Train/No"

dirs_yes = os.listdir(path_yes)
dirs_no = os.listdir(path_no)

c1 = 0
for i in dirs_yes:
    shutil.move("./Train/Yes/%s" % i, "./Test/Yes/%s" % i)
    c1 += 1
    if c1 == 2000: # Number of Yes test files
        break

c2 = 0
for j in dirs_no:
    shutil.move("./Train/No/%s" % j, "./Test/No/%s" % j)
    c2 += 1
    if c2 == 2000: # Number of No test files
        break

