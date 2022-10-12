from turtle import forward
from sympy import sring
import torch
import torch.nn as nn


def conv_bn_relu(inchannels, outchannels, kernel_size, stride=1):
    return nn.Sequential(
        nn.Conv2d(inchannels, outchannels, kernel_size,
                  padding=kernel_size//2, stride=stride),
        nn.BatchNorm2d(outchannels),
        nn.ReLU()
    )


def aux_classifier(inchannels, num_classes):
    return nn.Sequential(
        nn.AvgPool2d(kernel_size=5, stride=3),
        conv_bn_relu(inchannels, 128, kernel_size=1),
        nn.Flatten(),
        nn.Linear(4*4*128, 1024),
        nn.ReLU(),
        nn.Dropout(0.7),
        nn.Linear(1024, num_classes),
    )


class Inception(nn.Module):
    def __init__(self, inchannels, channels_1x1, channels_3x3_reduce, channels_3x3, channels_5x5_reduce, channels_5x5, channels_pool):
        super().__init__()

        self.branch_1x1 = conv_bn_relu(inchannels, channels_1x1, kernel_size=1)

        self.branch_3x3 = nn.Sequential(conv_bn_relu(inchannels, channels_3x3_reduce, kernel_size=1), conv_bn_relu(
            channels_3x3_reduce, channels_3x3, kernel_size=3))

        self.branch_5x5 = nn.Sequential(conv_bn_relu(inchannels, channels_5x5_reduce, kernel_size=1), conv_bn_relu(
            channels_5x5_reduce, channels_5x5, kernel_size=5))

        self.branch_pool = nn.Sequential(nn.MaxPool2d(
            kernel_size=3, stride=1, padding=1), conv_bn_relu(inchannels, channels_pool, kernel_size=1)
        )

    def forward(self, x):
        b1 = self.branch_1x1(x)
        b3 = self.branch_3x3(x)
        b5 = self.branch_5x5(x)
        bpool = self.branch_pool(x)
        return torch.cat((b1, b3, b5, bpool), dim=1)


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.prelayer = nn.Sequential(conv_bn_relu(
            3, 64, kernel_size=7, stride=2),
            self.maxpool,
            conv_bn_relu(64, 64, kernel_size=1),
            conv_bn_relu(64, 192, kernel_size=3),
            self.maxpool,
        )

        self.inception_3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception_3b = Inception(256, 128, 128, 192, 32, 96, 64)

        self.inception_4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception_4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception_4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception_4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception_4e = Inception(528, 256, 160, 320, 32, 128, 128)

        self.aux_classifier1 = aux_classifier(512, num_classes)
        self.aux_classifier2 = aux_classifier(528, num_classes)

        self.inception_5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception_5b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout2d(0.4)
        self.linear = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.prelayer(x)

        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.maxpool(x)

        x = self.inception_4a(x)
        out1 = self.aux_classifier1(x)

        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)
        out2 = self.aux_classifier2(x)

        x = self.inception_4e(x)
        x = self.maxpool(x)

        x = self.inception_5a(x)
        x = self.inception_5b(x)

        x = self.avg_pool(x)
        x = self.dropout(x)
        x = nn.Flatten()(x)
        out = self.linear(x)

        return out1, out2, out


if __name__ == '__main__':
    model = GoogLeNet()

    input = torch.randn((32, 3, 224, 224))
    out = model(input)
    print(
        f'Number of parameters: {sum(p.numel() for p in model.parameters())}')
