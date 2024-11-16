import os
import shutil
import random

rootpath = '/Volumes/data/data/dogs-vs-cats/train'

train_filenames = os.listdir(rootpath)
train_cat = list(filter(lambda x:x[:3] == 'cat', train_filenames))
train_dog = list(filter(lambda x:x[:3] == 'dog', train_filenames))

# Assuming you have rootpath and train_cat/train_dog defined as before
output_cat = [f"{os.path.join(rootpath, filename)} 0" for filename in train_cat]
output_dog = [f"{os.path.join(rootpath, filename)} 1" for filename in train_dog]

output = output_cat + output_dog
random.shuffle(output)

output_file = "/Volumes/data/data/dogs-vs-cats/train.txt"
with open(output_file, "w") as f:
    for line in output:
        f.write(line + "\n")
