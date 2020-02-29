import pandas as pd
import torch.nn as nn


class BaselineDNN(nn.Module):
    def __init__(self, num_features: int):
        self.num_features = num_features
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.BatchNormalization(256),
            nn.ReLu()
        )

        self.block2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNormalization(128),
            nn.ReLu()
        )

        self.out = nn.Linear(128, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return self.out(x)
