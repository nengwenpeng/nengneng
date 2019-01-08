import torch.nn as nn
import torch.nn.functional as F
import torch as t


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(23, 24*24)
        self.conv1 = nn.Conv2d(1, 4, 4, 2, 1)
        self.conv2 = nn.Conv2d(4, 8, 4, 2, 1)
        self.conv3 = nn.Conv2d(8, 16, 6, 1, 0)
        self.fc2 = nn.Linear(16, 1)
        self.fc3 = nn.Linear(1, 16)
        self.convT1 = nn.ConvTranspose2d(16, 8, 6, 1, 0)
        self.convT2 = nn.ConvTranspose2d(8, 4, 4, 2, 1)
        self.convT3 = nn.ConvTranspose2d(4, 1, 4, 2, 1)
        self.fc4 = nn.Linear(24*24, 23)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        # print(x.shape)
        x = x.view(x.size()[0], 1, 24, 24)
        x = F.selu(self.conv1(x))
        x = F.selu(self.conv2(x))
        x = F.selu(self.conv3(x))
        y = x.view(x.size()[0], -1)
        y = F.selu(self.fc2(y))
        y = F.sigmoid(y)
        # x = F.selu(self.fc3(y))
        # x = x.view(x.size()[0], 16, 1, 1)
        x = F.selu(self.convT1(x))
        # print(x.shape)
        x = F.selu(self.convT2(x))
        # print(x.shape)
        x = F.selu(self.convT3(x))
        # print(x.shape)
        x = x.view(x.size()[0], -1)
        x = F.selu(self.fc4(x))
        return x, y


