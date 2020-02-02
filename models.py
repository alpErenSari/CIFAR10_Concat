import torch.nn as nn
import torch.nn.functional as F

class NetWide(nn.Module):
    def __init__(self):
        super(NetWide, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(3, 6, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
#         self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 4 * 8, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # x [b,3,32,64]
        x = self.pool(F.relu(self.conv1(x))) # [b,6,16,32]
        x = self.pool(F.relu(self.conv2(x))) # [b,16,8,16]
        x = self.pool(F.relu(self.conv3(x))) # [b,32,4,8]
#         x = self.pool(F.relu(self.conv4(x))) # [b,3,2,4]
        x = x.view(-1, 32 * 4 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return self.sigmoid(x)
