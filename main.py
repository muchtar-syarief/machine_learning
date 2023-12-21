import splitfolders
import os

from zipfile import ZipFile


with ZipFile("./data/rockpaperscissors.zip", "r") as file:
    file.extractall("./data")


splitfolders.ratio("./data/rockpaperscissors/rps-cv-images", "./data/rps/", ratio=(0.8, 0.2))

train_dir = "./data/rps/train"
trains = os.listdir(train_dir)

val_dir = "./data/rps/val"
vals = os.listdir(val_dir)


count_train = 0
for train in trains:
    list_data = os.path.join(train_dir, train)
    data = os.listdir(list_data)

    count_train += len(data)

count_val = 0
for train in vals:
    list_data = os.path.join(train_dir, train)
    data = os.listdir(list_data)

    count_val += len(data)

print(count_train)
print(count_val)
