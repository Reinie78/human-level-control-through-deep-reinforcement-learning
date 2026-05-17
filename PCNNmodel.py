import torch
import torch.nn as nn
from PCNNLayer import PCNNLayer


class PCNN(nn.Module):
    def __init__(self, input_shape=(4, 84, 84), num_actions=6, pcnn_iters=10):
        super(PCNN, self).__init__()
        c, h, w = input_shape

        self.pcnn = PCNNLayer(h, w, iterations=pcnn_iters)

        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),  # Input channels = 4 (stacked frames)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        # x shape: (batch_size, 4, 84, 84)

        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        pcnn_outputs = []
        for i in range(x.size(1)):
            channel = x[:, i:i + 1, :, :]
            pcnn_out = self.pcnn(channel)
            pcnn_outputs.append(pcnn_out)

        x = torch.cat(pcnn_outputs, dim=1)

        x = self.conv(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x


