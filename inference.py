import torch
from torch.utils.data import DataLoader

import os
import json
import argparse
from tqdm import tqdm
from pathlib import Path
from main import RecognitionModel
from dataset import OneImage

    
def predict_one_song(model, input_path, results):
    test_dataset = OneImage(input_path)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        pin_memory=False,
        shuffle=False,
        drop_last=False,
    )
    results = model.predict(test_loader, results=results)

    return results


def predict_whole_dir(model, test_dir, results):
    results = {}

    for song in tqdm(os.listdir(test_dir)):
        input_path = os.path.join(test_dir, song)
        results = predict_one_song(model, input_path, results)

    return results


def make_predictions(testset_path, output_path, model):
    results = {}
    if os.path.isfile(testset_path):
        results = predict_one_song(model, testset_path, results)
    elif os.path.isdir(testset_path):
        results = predict_whole_dir(model, testset_path, results)
    else:
        print ("\"input\" argument is not valid")
        return {}

    with open(output_path, 'w') as f:
        output_string = json.dumps(results)
        f.write(output_string)

    return results


if __name__ == '__main__':
    """
    This script performs inference using the trained singing transcription model in main.py.
    
    Sample usage:
    python inference.py --save_model_path ./model_4 --test_path ./data/test
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_model_path', help='path to the trained model')
    parser.add_argument('--test_path', default='./data/test', help='path to the image(s) to be tested, either directory or file')
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    
    best_model = RecognitionModel(device, args.save_model_path)
    pwd = os.path.dirname(__file__)
    output_path = os.path.join(pwd, "predictions.json")

    make_predictions(args.test_path, output_path, best_model)
