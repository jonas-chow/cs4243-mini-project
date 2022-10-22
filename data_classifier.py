import os
import shutil
import json
import math

"""
Splits the data into train, valid, test datasets to be further processed

Download the data from the gdrive and have them in the carrying, normal, threat folders
"""

TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
# train ratio is just 1 - those two

carrying_path = os.path.join(os.path.dirname(__file__), "carrying")
normal_path = os.path.join(os.path.dirname(__file__), "normal")
threat_path = os.path.join(os.path.dirname(__file__), "threat")

carrying_files = os.listdir(carrying_path)
normal_files = os.listdir(normal_path)
threat_files = os.listdir(threat_path)

annotations = {}

pwd = os.path.dirname(__file__)
data_dir = os.path.join(pwd, "data")
os.mkdir(data_dir)
test_dir = os.path.join(data_dir, "test")
os.mkdir(test_dir)
train_dir = os.path.join(data_dir, "train")
os.mkdir(train_dir)
valid_dir = os.path.join(data_dir, "valid")
os.mkdir(valid_dir)

for classification in ["carrying", "normal", "threat"]:
    path = os.path.join(pwd, classification)
    files = os.listdir(path)
    length = len(files)
    train_end_index = math.floor(TRAIN_RATIO * length)
    valid_end_index = math.floor((TRAIN_RATIO + VALID_RATIO) * length)

    train_set = files[:train_end_index]
    valid_set = files[train_end_index:valid_end_index] 
    test_set = files[valid_end_index:]

    for file in train_set:
        annotations[file] = classification
        shutil.move(
            os.path.join(path, file),
            os.path.join(train_dir, file)
        )
    for file in valid_set:
        annotations[file] = classification
        shutil.move(
            os.path.join(path, file),
            os.path.join(valid_dir, file)
        )
    for file in test_set:
        annotations[file] = classification
        shutil.move(
            os.path.join(path, file),
            os.path.join(test_dir, file)
        )
    os.rmdir(path)

with open(os.path.join(data_dir, "annotations.json"), mode="w") as json_f:
    json.dump(annotations, json_f, indent=2)
