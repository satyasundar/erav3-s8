import torch.nn as nn

DROPOUT_VALUE = 0.1


# Model definition
class CIFAR10Model(nn.Module):
    def __init__(self):
        super(CIFAR10Model, self).__init__()

        # Convolution Block 1
        self.convblock1 = nn.Sequential(
            # Block 1, Layer 1 : Convolution
            nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(DROPOUT_VALUE),  # Input: 3x32x32, Output:16x32x32, RF:5
            # Block 1, Layer 2 : Convolution
            nn.Conv2d(16, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(DROPOUT_VALUE),  # Input: 16x32x32, Output:32x32x32, RF:7
            # Block 1, Layer 3 : Depthwise Separable Convolution
            nn.Conv2d(32, 32, 3, stride=1, padding=1, groups=32),
            nn.Conv2d(32, 64, 1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),  # Input: 32x32x32, Output:64x32x32, RF:9
            # Block 1, Layer 4 : downsampling with stride 2
            nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(DROPOUT_VALUE),  # Input: 64x32x32, Output:64x16x16, RF:13
        )

        # Transition Block
        self.transition1 = nn.Sequential(
            nn.Conv2d(64, 16, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),  # Input: 64x16x16, Output:16x16x16, RF:13
        )

        # Convolution Block 2
        self.convblock2 = nn.Sequential(
            # Block 2, Layer 1
            nn.Conv2d(16, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(DROPOUT_VALUE),  # Input: 16x16x16, Output:32x16x16, RF:17
            # Block 2, Layer 2
            nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(DROPOUT_VALUE),  # Input: 32x16x16, Output:32x16x16, RF:21
            # Block 2,  Layer 3
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(DROPOUT_VALUE),  # Input: 32x16x16, Output:64x8x8, RF:29
        )

        # Transition Block 2
        self.transition2 = nn.Sequential(
            nn.Conv2d(64, 16, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),  # Input: 64x8x8, Output:16x8x8, RF:29
        )

        # Convolution Block 3
        self.convblock3 = nn.Sequential(
            # Block 3, Layer 1
            nn.Conv2d(16, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(DROPOUT_VALUE),  # Input: 16x8x8, Output:32x8x8, RF:37
            # Block 3, Layer 2
            nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(DROPOUT_VALUE),  # Input: 32x8x8, Output:64x8x8, RF:45
            # Block 3, Layer 3 - Downsampling
            nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(DROPOUT_VALUE),  # Input: 64x8x8, Output:64x4x4, RF:61
        )

        # Transistion Block 3
        self.transition3 = nn.Sequential(
            nn.Conv2d(64, 16, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),  # Input: 64x4x4, Output:16x4x4, RF:61
        )

        # Convolution Block 4, Dilated Convolution
        self.convblock4 = nn.Sequential(
            # Block 4, Layer 1 : Dilated Convolution
            nn.Conv2d(16, 64, 3, stride=1, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),  # Input: 16x4x4, Output:64x4x4, RF:93
        )

        # Output Block
        self.outputblock = nn.Sequential(
            # GAP Layer
            nn.AdaptiveAvgPool2d(1),
            # Convolution after GAP
            nn.Conv2d(64, 10, 1, padding=0, bias=False),
        )

    def forward(self, x):
        x = self.convblock1(x)
        x = self.transition1(x)
        x = self.convblock2(x)
        x = self.transition2(x)
        x = self.convblock3(x)
        x = self.transition3(x)
        x = self.convblock4(x)
        x = self.outputblock(x)
        return x.view(-1, 10)
