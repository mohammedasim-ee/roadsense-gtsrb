"""Scratch CNN for 32x32 (or 48x48) traffic-sign images."""
import torch.nn as nn


def conv_block(c_in, c_out):
    # Conv -> BatchNorm -> ReLU -> MaxPool: halves the spatial size
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class BaselineCNN(nn.Module):
    def __init__(self, num_classes=43, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32), conv_block(32, 64), conv_block(64, 128),
            nn.AdaptiveAvgPool2d(4),  # works for 32x32 and 48x48 inputs
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
