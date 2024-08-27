import torch.nn as nn
import torch


class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))

        x = torch.flatten(x)

        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)

        return x