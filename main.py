import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import os
import time
import pickle
import numpy as np
from tqdm import tqdm

from model import BaseNN
from dataset import RecognitionDataset

class RecognitionModel:
    """
    This is the main class for training model and making predictions
    """
    def __init__(self, device="cuda:0", model_path=None):
        # Initialize model
        self.device = device
        self.model = BaseNN().to(self.device)
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print("Model loaded")
        else:
            print("Model initialized")

    def fit(self, learning_params):
        # other possible args: train , valid, model save directories
        pwd = os.path.dirname(__file__)
        train_path = os.path.join(pwd, "data", "train.pickle")
        valid_path = os.path.join(pwd, "data", "valid.pickle")

        with open(train_path, 'rb') as f:
            trainset = pickle.load(f)
        with open(valid_path, 'rb') as f:
            validset = pickle.load(f)

        # no idea about these arguments
        trainset_loader = DataLoader(
            trainset,
            batch_size=learning_params['batch_size'],
            num_workers=0,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )

        validset_loader = DataLoader(
            validset,
            batch_size = 1,
            num_workers=0,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )

        # Set optimizer and loss functions... ???
        optimizer = optim.Adam(self.model.parameters(), lr=learning_params['lr'])
        criterion = nn.CrossEntropyLoss(ignore_index=100)

        start_time = time.time()
        best_model_id = -1
        min_valid_loss = 10000
        prev_loss = 100000
        epoch_num = learning_params['epoch']
        valid_every_k_epoch = learning_params['valid_freq']

        trigger_times = 0
        patience = 5

        for epoch in range(epoch_num):  
            running_loss = 0
            num_batches = 0

            for batch_features, batch_labels in trainset_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                scores = self.model(batch_features)

                loss = criterion(scores, batch_labels)

                # magic
                optimizer.zero_grad()
                loss.backward()
                optimizer.step() 
                
                # stats
                running_loss += loss.detach().item()
                num_batches += 1

            loss = running_loss / num_batches
            elapsed_time = time.time() - start_time

            print(' ')
            print('epoch=', epoch, '\t time=', elapsed_time,
                    '\t loss=', loss)

            if prev_loss < loss :
                trigger_times += 1
                if trigger_times >= patience:
                    break
            else :
                trigger_times = 0
                prev_loss = loss   


            if epoch % valid_every_k_epoch == 0 : 
                valid_running_loss = 0
                valid_num_batches = 0

                for valid_batch_features, valid_batch_labels in validset_loader:
                    valid_batch_features = valid_batch_features.to(self.device)
                    valid_batch_labels = valid_batch_labels.to(self.device)
                    valid_scores = self.model(valid_batch_features)

                    valid_loss = criterion(valid_scores, valid_batch_labels)

                    valid_running_loss += valid_loss.detach().item()
                    valid_num_batches += 1
                
                valid_loss = valid_running_loss / valid_num_batches

                if valid_loss < min_valid_loss:
                    best_model_id = epoch
                    min_valid_loss = valid_loss
                
                     

        print('Training done in {:.1f} minutes.'.format((time.time()-start_time)/60))
        torch.save(self.model.state_dict(), os.path.join(pwd, 'model_' + str(best_model_id)))
        return best_model_id

    def predict(self, test_loader, results={}):
        self.model.eval()
        with torch.no_grad():   
            # can tqdm this if we have big batches         
            for batch in test_loader:
                # Parse batch data
                input_tensor = batch[0].to(self.device)
                image_path = batch[1][0]

                categories = ["normal", "carrying", "threat"]

                classification = self.model(input_tensor)

                # not too sure what these do
                probs = torch.sigmoid(classification).cpu()
                classification = classification.cpu()
                classification = torch.argmax(classification).item()

                results[image_path] = categories[classification]

        return results

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    recognition_model = RecognitionModel(device)

    learning_params = {
        'batch_size': 64,
        'epoch': 32,
        'lr': 1e-4,
        'valid_freq': 1,
        'save_freq': 1
    }

    best_model_id = recognition_model.fit(learning_params)
    print("Best model ID: ", best_model_id)