import torch
from torch.utils.data import Dataset

import os
import json
import pickle
import numpy as np
from tqdm import tqdm
import cv2

def get_features(video_path, yolo_model):
    extension = os.path.splitext(video_path)[-1].lower()
    if extension != ".mp4":
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    target_fps = 1
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    hop = np.floor(actual_fps / target_fps)

    i = 0
    frames = []

    ret, frame = cap.read()
    if not ret:
        return []

    width, height, channels = frame.shape
    min_x = width - 1
    min_y = height - 1
    max_x = 0
    max_y = 0

    # read images and get bounding box that contains all people
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        i += 1
        # skip some frames otherwise the dataset is too big, and the movement is too subtle?
        if i % hop != 0:
            continue
        if not ret:
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Inference
        results = yolo_model(frame)

        # Results
        # uncomment this line if you want to see the boxes
        # results.save()
        res = results.pandas().xyxy[0]
        has_persons = 'person' in res['name'].unique()
        if not has_persons:
            continue

        res["area"] = (res["xmax"] - res["xmin"]) * (res["ymax"] - res["ymin"])
        persons = res[(res.name == "person")]
        
        persons = persons.sort_values(by=["area"], ascending=False)

        xmin = np.int32(persons['xmin'].iloc[0])
        ymin = np.int32(persons['ymin'].iloc[0])
        xmax = np.int32(persons['xmax'].iloc[0])
        ymax = np.int32(persons['ymax'].iloc[0])        

        # not including other objects because if the person moves a lot, 
        # he will likely collide with a bunch of objects and nothing will be cropped

        min_x = min(xmin, min_x)
        min_y = min(ymin, min_y)
        max_x = max(xmax, max_x)
        max_y = max(ymax, max_y)

        # only include frames with people
        frames.append(frame)

    if not len(frames):
        return []

    num_frames = len(frames)

    # give a padding because maybe not including what they hold
    min_x = max(0, min_x - 10)
    min_y = max(0, min_y - 10)
    max_x = min(width - 1, max_x + 10)
    max_y = min(height - 1, max_y + 10)

    frames = [frame[min_y:max_y, min_x:max_x] for frame in frames]

    previous_mask = None

    ret = []

    for i in range(num_frames - 1):
        curr = cv2.resize(frames[i], (256, 256))
        next = cv2.resize(frames[i - 1], (256, 256))

        # output: height x width x 2 ndarray which represents how much the pixel moved
        flow = cv2.calcOpticalFlowFarneback(curr, next, None, 0.5, 3, 30, 3, 7, 1.5, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # drop movements that are too small (likely to be background)
        # higher max likely means that the person move fast?
        thresholds = mag > (max(np.mean(mag), np.max(mag) * 0.4))

        # drop videos where it's likely the background movement is too noisy
        if np.sum(thresholds) > (width * height) * 0.4:
            continue
        
        mask = thresholds * np.ones(thresholds.shape)

        # make the lines thicker?
        mask = cv2.dilate(mask, np.ones((11, 11)), iterations=1)

        if previous_mask is None:
            previous_mask = mask
            continue

        # union to prevent too mant missing details
        # there might be a case for intersection here too: more accurate, but lose more info
        curr_mask = np.zeros(mask.shape, dtype=np.bool8)
        curr_mask[np.logical_or(mask == 1, previous_mask == 1)] = True

        # canny before masking so that the circles don't appear as artifacts
        removed = np.uint8(cv2.Canny(curr, 50, 150) * curr_mask)
        # duplicated because model wants 3 channels
        ret.append(removed)

        previous_mask = mask

    return ret

class RecognitionDataset(Dataset):
    """
        Preprocess the images to produce training and validation data here
    """
    def __init__(self, annotations, video_dir, yolo_model):
        data = []

        for video_name in tqdm(os.listdir(video_dir)):
            # the label
            label = self.get_label(annotations, video_name)

            # get features based on 
            frame_features = get_features(
                os.path.join(video_dir, video_name), 
                yolo_model,
            )

            for frame in frame_features:
                data.append((frame, label))

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

class OneVideo(Dataset):
    """
    For preprocessing and preparing testing data of one image...
    """
    def __init__(self, video_path, yolo_model):
        data = []
        frame_features = get_features(video_path, yolo_model)
        head, tail = os.path.split(video_path)
        for feature in frame_features:
            data.append((feature, tail))
        self.data = data
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, _verbose=False)
    yolo_model.to(device)

    pwd = os.path.dirname(__file__)
    train_dir = os.path.join(pwd, "data", "train")
    valid_dir = os.path.join(pwd, "data", "valid")

    with open(os.path.join(pwd, "data", "annotations.json")) as json_data:
        annotations = json.load(json_data)

    train_dataset = RecognitionDataset(annotations, train_dir, yolo_model)
    with open(os.path.join(pwd, "data", "train.pickle"), 'wb') as f:
        pickle.dump(train_dataset, f)
    
    valid_dataset = RecognitionDataset(annotations, valid_dir, yolo_model)
    with open(os.path.join(pwd, "data", "valid.pickle"), 'wb') as f:
        pickle.dump(valid_dataset, f)
