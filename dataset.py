import torch
from torch.utils.data import Dataset

import os
import json
import pickle
import numpy as np
from tqdm import tqdm
import cv2

def get_features(image):
    # probably need to expand on this
    # possible ideas: use grayscale, sobel lines

    # note: this is a very extreme resizing, but bigger images will also mean longer training process
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (256, 256))
    img = cv2.Canny(img, 50, 150)
    ret = np.array([img, img, img], dtype=np.float32)
    return ret
    

class RecognitionDataset(Dataset):
    """
        Preprocess the images to produce training and validation data here
    """
    def __init__(self, annotations, image_dir):
        data = []

        for image_name in tqdm(os.listdir(image_dir)):
            # the label
            label = self.get_label(annotations, image_name)

            # get features based on 
            features = get_features(
                cv2.imread(os.path.join(image_dir, image_name))
            )
            data.append((features, label))

        self.data = data

    def get_label(self, annotations, image_name):
        annotation = annotations[image_name]

        # convert to number so computer has a better time categorising? idk
        annotation_map = {
            "normal": 0,
            "carrying": 1,
            "threat": 2,
        }

        index = annotation_map[annotation]
        label = np.zeros(3, dtype=np.float32)
        label[index] = 1.0
        return label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]    

class OneImage(Dataset):
    """
    For preprocessing and preparing testing data of one image...
    """
    def __init__(self, image_path):
        features = get_features(cv2.imread(image_path))
        head, tail = os.path.split(image_path)
        self.data = [(features, tail)]
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    pwd = os.path.dirname(__file__)
    train_dir = os.path.join(pwd, "data", "train")
    valid_dir = os.path.join(pwd, "data", "valid")

    with open(os.path.join(pwd, "data", "annotations.json")) as json_data:
        annotations = json.load(json_data)

    train_dataset = RecognitionDataset(annotations, train_dir)
    with open(os.path.join(pwd, "data", "train.pickle"), 'wb') as f:
        pickle.dump(train_dataset, f)
    valid_dataset = RecognitionDataset(annotations, valid_dir)
    with open(os.path.join(pwd, "data", "valid.pickle"), 'wb') as f:
        pickle.dump(valid_dataset, f)
