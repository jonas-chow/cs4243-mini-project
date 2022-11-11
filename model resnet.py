import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class BaseNN(nn.Module):
    def __init__(self, pretrained = True, num_classes = 3, drop_rate = 0):
        super(BaseNN, self).__init__()
        resnet = models.resnet18(pretrained) #https://pytorch.org/vision/0.8/models.html
        #for param in resnet.parameters() :
        #    param.requires_grad = False
        
        self.features = nn.Sequential(*list(resnet.children())[:-1]) # after avgpool 512x1

        fc_in_dim = list(resnet.children())[-1].in_features # original fc layer's in dimention 512
        self.fc = nn.Linear(fc_in_dim, 3) # new fc layer 512x8

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc(x)


        return F.softmax(x, dim=1) #classification output
