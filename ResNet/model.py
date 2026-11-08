import torch
import torch.nn as nn
from torchsummary import summary

class ResidualBlock(nn.Module):
    def __init__(self, use_1conv=False, input_cannels=64, num_channels=64, stride=1):
        super(ResidualBlock, self).__init__()
        self.ReLU = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=input_cannels, out_channels=num_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
        if use_1conv:
            self.conv3 = nn.Conv2d(in_channels=input_cannels, out_channels=num_channels, kernel_size=1, stride=stride, padding=0)
        else:
            self.conv3 = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.ReLU(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.conv3 is not None:
            identity = self.conv3(identity)

        out += identity
        out = self.ReLU(out)

        return out

class RsNet18(nn.Module):
    def __init__(self,ResidualBlock):
        super(RsNet18, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b2 = nn.Sequential(
            ResidualBlock(use_1conv=False, input_cannels=64, num_channels=64, stride=1),
            ResidualBlock(use_1conv=False, input_cannels=64, num_channels=64, stride=1)
        )

        self.b3 = nn.Sequential(
            ResidualBlock(use_1conv=True, input_cannels=64, num_channels=128, stride=2),
            ResidualBlock(use_1conv=False, input_cannels=128, num_channels=128, stride=1)
        )

        self.b4 = nn.Sequential(
            ResidualBlock(use_1conv=True, input_cannels=128, num_channels=256, stride=2),
            ResidualBlock(use_1conv=False, input_cannels=256, num_channels=256, stride=1)
        )

        self.b5 = nn.Sequential(
            ResidualBlock(use_1conv=True, input_cannels=256, num_channels=512, stride=2),
            ResidualBlock(use_1conv=False, input_cannels=512, num_channels=512, stride=1)
        )

        self.b6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.b6(x)

        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RsNet18(ResidualBlock).to(device)
    print(summary(model, (1, 224, 224)))