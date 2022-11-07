import cv2
import numpy as np
import argparse
import os
from tqdm import tqdm
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', help='directory of videos', default="vids")
    args = parser.parse_args()

    pwd = os.path.dirname(__file__)
    dir = os.path.join(pwd, args.d)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, device='cpu', _verbose=False)

    for index, vid in enumerate(tqdm(os.listdir(dir))):
        vid_path = os.path.join(dir, vid)
        cap = cv2.VideoCapture(vid_path)

        if not cap.isOpened():
            continue

        target_fps = 1
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        hop = np.floor(actual_fps / target_fps)

        i = 0
        frames = []

        ret, frame = cap.read()
        if not ret:
            break

        height, width, channels = frame.shape
        min_x = width - 1
        min_y = height - 1
        max_x = 0
        max_y = 0

        mog2 = cv2.createBackgroundSubtractorMOG2()

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
            results = model(frame)

            # Results
            # uncomment this line if you want to see the boxes
            #results.save()
            res = results.pandas().xyxy[0]
            res["area"] = (res["xmax"] - res["xmin"]) * (res["ymax"] - res["ymin"])
            persons = res[(res.name == "person")]
            has_persons = not persons.empty

            if has_persons:
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
            continue

        num_frames = len(frames)

        # give a padding because not including what they hold
        min_x = max(0, min_x - 10)
        min_y = max(0, min_y - 10)
        max_x = min(width - 1, max_x + 10)
        max_y = min(height - 1, max_y + 10)

        frames = [frame[min_y:max_y, min_x:max_x] for frame in frames]

        previous_mask = None

        for i in range(num_frames - 1):
            curr = cv2.resize(frames[i], (256, 256))
            next = cv2.resize(frames[i - 1], (256, 256))

            mog_mask = mog2.apply(curr)
            mog_mask = cv2.dilate(mog_mask, np.ones((5, 5)), iterations=1)
            mog_mask = mog_mask > 0

            # output: height x width x 2 ndarray which represents how much the pixel moved
            flow = cv2.calcOpticalFlowFarneback(curr, next, None, 0.5, 3, 30, 3, 7, 1.5, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # drop movements that are too small (likely to be background)
            thresholds = mag > np.mean(mag)
            
            mask = thresholds * np.ones(thresholds.shape)

            # make the lines thicker?
            mask = cv2.dilate(mask, np.ones((9, 9)), iterations=1)

            if previous_mask is None:
                previous_mask = mask
                continue

            # union to prevent too many missing details
            # there might be a case for intersection here too: more accurate, but lose more info
            curr_mask = np.zeros(mask.shape, dtype=np.bool8)
            curr_mask[np.logical_or(mask == 1, previous_mask == 1)] = True

            mog_coords = np.argwhere(mog_mask)
            # median [y, x]
            mog_median = np.median(mog_coords.transpose(), axis=1)
            # essentially average sum of squares of cartesian distance from the median point            
            mog_var = np.mean(np.square(mog_coords - mog_median))

            curr_coords = np.argwhere(curr_mask)
            # median [y, x]
            curr_median = np.median(curr_coords.transpose(), axis=1)
            # essentially average sum of squares of cartesian distance from the median point
            curr_var = np.mean(np.square(curr_coords - curr_median))

            # canny before masking so that the masked out parts don't appear as artifacts
            lines = cv2.Canny(curr, 50, 150)
            
            print(i, "curr" if curr_var < mog_var else "mog")
            mask_used = curr_mask if curr_var < mog_var else mog_mask
            removed = np.uint8(lines * mask_used)

            previous_mask = mask
            
            cv2.imwrite(os.path.join(dir, "vid" + str(index) + "_" + str(i) + "_original.png"), curr)
            cv2.imwrite(os.path.join(dir, "vid" + str(index) + "_" + str(i) + "_of_mask.png"), np.ones(curr_mask.shape) * 255 * curr_mask)
            cv2.imwrite(os.path.join(dir, "vid" + str(index) + "_" + str(i) + "_mog_mask.png"), np.ones(mog_mask.shape) * 255 * mog_mask)
            cv2.imwrite(os.path.join(dir, "vid" + str(index) + "_" + str(i) + "removed.png"), removed)

        # os.remove(vid_path)

