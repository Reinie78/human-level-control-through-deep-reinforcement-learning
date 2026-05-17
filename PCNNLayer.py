import torch
import torch.nn as nn
import torch.nn.functional as F

# tutaj przechowujemy zaimplementowaną warstwę sieci PCNN
# synaptic weights to nasze pole charakterystyk

class PCNNLayer(nn.Module):
    def __init__(self, height, width, beta=1.0, decay=0.9, threshold_decay=0.9, iterations=8):
        super(PCNNLayer, self).__init__()
        self.height = height
        self.width = width
        self.beta = beta
        self.decay = decay
        self.threshold_decay = threshold_decay
        self.iterations = iterations

        self.kernel = nn.Parameter(torch.tensor([[0.5, 1.0, 0.5],
                                                 [1.0,  0.0, 1.0],
                                                 [0.5, 1.0, 0.5]]).unsqueeze(0).unsqueeze(0), requires_grad=False)

        self.register_buffer('theta', torch.ones(1, 1, height, width))

    def forward(self, x):
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)

        batch_size = x.size(0)
        y = torch.zeros_like(x)
        f = x.clone()
        theta = self.theta.repeat(batch_size, 1, 1, 1)
        y_out = torch.zeros_like(x)

        for t in range(self.iterations):
            if t > 0:
                L = F.conv2d(y, self.kernel, padding=1)
            else:
                L = torch.zeros_like(f)

            U = f * (1 + self.beta * L)

            y = (U > theta).float()

            theta = self.threshold_decay * theta + y

            y_out += y

        return y_out / self.iterations
