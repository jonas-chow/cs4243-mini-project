import torch
from torch.utils.data import Dataset

import os
import json
import pickle
import numpy as np
from tqdm import tqdm
import cv2

def get_features(image_path, device):
    # probably need to expand on this
    # possible ideas: use grayscale, sobel lines
    image = cv2.imread(image_path)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, device=device, _verbose=False)

    # Images
    imgs = [image_path]  # batch of images

    # Inference
    results = model(imgs)

    # Results

    # uncomment this line if you want to see the boxes
    # results.save()
    res = results.pandas().xyxy[0]  # img1 predictions (pandas)
    res["area"] = (res["xmax"] - res["xmin"]) * (res["ymax"] - res["ymin"])
    persons = res[(res.name == "person")]
    has_persons = not persons.empty

    if has_persons:
        persons = persons.sort_values(by=["area"], ascending=False)

        person_xmin = np.int32(persons['xmin'].iloc[0])
        person_ymin = np.int32(persons['ymin'].iloc[0])
        person_xmax = np.int32(persons['xmax'].iloc[0])
        person_ymax = np.int32(persons['ymax'].iloc[0])        

        xmin = person_xmin
        ymin = person_ymin
        xmax = person_xmax
        ymax = person_ymax

        for idx, row in res.iterrows():
            # don't expand with more people
            if row['name'] == "person":
                continue

            row_xmin = np.int32(row['xmin'])
            row_ymin = np.int32(row['ymin'])
            row_xmax = np.int32(row['xmax'])
            row_ymax = np.int32(row['ymax'])
            if not (row_xmax < person_xmin or row_xmin > person_xmax or
                row_ymax < person_ymin or row_ymin > person_ymax):
                # intersects with the person
                xmin = min(xmin, row_xmin)
                xmax = max(xmax, row_xmax)
                ymin = min(ymin, row_ymin)
                ymax = max(ymax, row_ymax)

        img = img[ymin:ymax, xmin:xmax]

    img = cv2.resize(img, (256, 256))
    lines = cv2.Canny(img, 50, 150)
    ret = np.array([lines, lines, lines], dtype=np.float32)

    return (has_persons, ret)
    

class RecognitionDataset(Dataset):
    """
        Preprocess the images to produce training and validation data here
    """
    def __init__(self, annotations, image_dir, device):
        data = []

        for image_name in tqdm(os.listdir(image_dir)):
            # the label
            label = self.get_label(annotations, image_name)

            # get features based on 
            has_person, features = get_features(
                os.path.join(image_dir, image_name), 
                device,
            )

            if has_person:
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
        has_person, features = get_features(image_path)
        head, tail = os.path.split(image_path)
        self.data = [(features, tail)]
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pwd = os.path.dirname(__file__)
    train_dir = os.path.join(pwd, "data", "train")
    valid_dir = os.path.join(pwd, "data", "valid")

    with open(os.path.join(pwd, "data", "annotations.json")) as json_data:
        annotations = json.load(json_data)

    train_dataset = RecognitionDataset(annotations, train_dir, device)
    with open(os.path.join(pwd, "data", "train.pickle"), 'wb') as f:
        pickle.dump(train_dataset, f)
    valid_dataset = RecognitionDataset(annotations, valid_dir, device)
    with open(os.path.join(pwd, "data", "valid.pickle"), 'wb') as f:
        pickle.dump(valid_dataset, f)
