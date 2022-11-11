import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class BaseNN(nn.Module):
    def __init__(self, categories=3):
        super(BaseNN, self).__init__()
        
        # so i have no idea how many input channels we should have
        # is it width * height * 3??? is it just 3?? idkkkkk
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(32, 16, kernel_size=(3, 3), padding=(1, 1))
        self.fc1   = nn.Linear(65536, 256)
        self.fc2   = nn.Linear(256, 128)
        self.fc3   = nn.Linear(128, 128)
        self.fc4   = nn.Linear(128, categories)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv3(out))
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out
