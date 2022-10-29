import torch
import torch.nn as nn
import torch.nn.functional as F

# note: i'm absolutely not sure about this. didn't touch this in 4347 lmao

class BaseNN(nn.Module):
    '''
        This is a base CNN model.
    '''
    def __init__(self, categories=3):
        super(BaseNN, self).__init__()
        
        # so i have no idea how many input channels we should have
        # is it width * height * 3??? is it just 3?? idkkkkk
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1))
        self.conv4 = nn.Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1))
        self.conv5 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1))
        self.fc1   = nn.Linear(32768, 4096)
        self.fc2   = nn.Linear(4096, categories)
        # i feel like the sudden drop from 4096 to 3 might be too drastic so i left some stuff here in case we want to change back
        # self.fc3   = nn.Linear(128, 128)
        # self.fc4   = nn.Linear(128, categories)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv3(out))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv4(out))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv5(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)

        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        # out = F.relu(self.fc2(out))
        # out = F.relu(self.fc3(out))
        # out = self.fc4(out)

        return F.softmax(out, dim=1)
