
import torch.nn as nn

def make_mlp(dropout=0.2):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(256, 10)
    )

class SmallCNN(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32*7*7, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 10)
        )
    def forward(self, x): return self.net(x)
