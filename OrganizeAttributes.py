import os
import shutil
import pandas as pd

df = pd.read_csv("list_attr_celeba.csv", sep=",") # Read csv file

## SELECT ATTRIBUTES FOR CLASSIFY
status = df["No_Beard"]
status1 = df["Mustache"]
name = df["image_id"]

c = 0
for i, j in zip(name,status):
    if status[c] == -1 or status1[c] == 1:
        shutil.move("./img_align_celeba/%s" % name[c], "./Train/Yes/%s" % name[c])
    else:
        shutil.move("./img_align_celeba/%s" % name[c], "./Train/No/%s" % name[c])
    
    c += 1
    print(c)
