from turtle import forward
from sympy import sring
import torch
import torch.nn as nn


def conv_bn_relu(inchannels, outchannels, kernel_size=(3, 3), stride=(1, 1), padding='same'):
    return nn.Sequential(
        nn.Conv2d(inchannels, outchannels, kernel_size,
                  padding=padding, stride=stride),
        nn.BatchNorm2d(outchannels),
        nn.ReLU()
    )


class InceptionBlock_A(nn.Module):
    def __init__(self, inchannels, poolchannels):
        super().__init__()
        self.branch_1x1 = conv_bn_relu(inchannels, 64, kernel_size=(1, 1))

        self.branch_3x3 = nn.Sequential(conv_bn_relu(inchannels, 64, kernel_size=(1, 1)),
                                        conv_bn_relu(64, 96),
                                        conv_bn_relu(96, 96))

        self.branch_5x5 = nn.Sequential(conv_bn_relu(inchannels, 48, kernel_size=(1, 1)),
                                        conv_bn_relu(48, 64, kernel_size=(5, 5)))

        self.branch_pool = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                                         conv_bn_relu(inchannels, poolchannels, kernel_size=(1, 1)))

    def forward(self, x):
        b1 = self.branch_1x1(x)
        b3 = self.branch_3x3(x)
        b5 = self.branch_5x5(x)
        bp = self.branch_pool(x)

        return torch.cat([b1, b3, b5, bp], dim=1)


class InceptionBlock_B(nn.Module):
    def __init__(self, inchannels, midchannels):
        super().__init__()
        self.branch_1x1 = conv_bn_relu(inchannels, 192, kernel_size=(1, 1))

        self.branch_7x7 = nn.Sequential(conv_bn_relu(inchannels, midchannels, kernel_size=(1, 1)),
                                        conv_bn_relu(
                                            midchannels, midchannels, kernel_size=(1, 7)),
                                        conv_bn_relu(midchannels, 192, kernel_size=(7, 1)))

        self.branch2_7x7 = nn.Sequential(conv_bn_relu(inchannels, midchannels, kernel_size=(1, 1)),
                                         conv_bn_relu(
                                             midchannels, midchannels, kernel_size=(7, 1)),
                                         conv_bn_relu(
                                             midchannels, midchannels, kernel_size=(1, 7)),
                                         conv_bn_relu(
                                             midchannels, midchannels, kernel_size=(7, 1)),
                                         conv_bn_relu(midchannels, 192, kernel_size=(1, 7)))

        self.branch_pool = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                                         conv_bn_relu(inchannels, 192, kernel_size=(1, 1)))

    def forward(self, x):
        b1 = self.branch_1x1(x)
        b7 = self.branch_7x7(x)
        b2_7 = self.branch2_7x7(x)
        bp = self.branch_pool(x)

        return torch.cat([b1, b7, b2_7, bp], dim=1)


class InceptionBlock_C(nn.Module):
    def __init__(self, inchannels):
        super().__init__()
        self.branch_1x1 = conv_bn_relu(inchannels, 320, kernel_size=(1, 1))

        self.branch_3x3 = conv_bn_relu(inchannels, 384, kernel_size=(1, 1))
        self.branch_3x3_1 = conv_bn_relu(384, 384, kernel_size=(1, 3))
        self.branch_3x3_2 = conv_bn_relu(384, 384, kernel_size=(3, 1))

        self.branch2_3x3 = nn.Sequential(conv_bn_relu(inchannels, 448, kernel_size=(1, 1)),
                                         conv_bn_relu(448, 384, kernel_size=(3, 3)))
        self.branch2_3x3_1 = conv_bn_relu(384, 384, kernel_size=(1, 3))
        self.branch2_3x3_2 = conv_bn_relu(384, 384, kernel_size=(3, 1))

        self.branch_pool = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                                         conv_bn_relu(inchannels, 192, kernel_size=(1, 1)))

    def forward(self, x):
        b1 = self.branch_1x1(x)

        b3 = self.branch_3x3(x)
        b3_1 = self.branch_3x3_1(b3)
        b3_2 = self.branch_3x3_2(b3)
        b3 = torch.cat([b3_1, b3_2], dim=1)

        b3_2 = self.branch2_3x3(x)
        b3_1 = self.branch2_3x3_1(b3_2)
        b3_2 = self.branch2_3x3_2(b3_2)
        b3_2 = torch.cat([b3_1, b3_2], dim=1)

        bp = self.branch_pool(x)

        return torch.cat([b1, b3, b3_2, bp], dim=1)


class ReductionBlock_A(nn.Module):
    def __init__(self, inchannels):
        super().__init__()
        self.branch_3x3 = conv_bn_relu(
            inchannels, 384, stride=(2, 2), padding='valid')

        self.branch_3x3_2 = nn.Sequential(conv_bn_relu(inchannels, 64, kernel_size=(1, 1)),
                                          conv_bn_relu(64, 96),
                                          conv_bn_relu(96, 96, stride=(2, 2), padding='valid'))

        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        b3 = self.branch_3x3(x)

        b3_2 = self.branch_3x3_2(x)

        bp = self.branch_pool(x)

        return torch.cat([b3, b3_2, bp], dim=1)


class ReductionBlock_B(nn.Module):
    def __init__(self, inchannels):
        super().__init__()
        self.branch_3x3 = nn.Sequential(conv_bn_relu(inchannels, 192, kernel_size=(1, 1)),
                                        conv_bn_relu(192, 320, stride=(2, 2), padding='valid'))

        self.branch_7x7 = nn.Sequential(conv_bn_relu(inchannels, 192, kernel_size=(1, 1)),
                                        conv_bn_relu(
                                            192, 192, kernel_size=(1, 7)),
                                        conv_bn_relu(
                                            192, 192, kernel_size=(7, 1)),
                                        conv_bn_relu(192, 192, stride=(2, 2), padding='valid'))

        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        b3 = self.branch_3x3(x)

        b7 = self.branch_7x7(x)

        bp = self.branch_pool(x)

        return torch.cat([b3, b7, bp], dim=1)


class InceptionV3(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.prelayer = nn.Sequential(conv_bn_relu(3, 32, stride=(2, 2), padding='valid'),
                                      conv_bn_relu(32, 32, padding='valid'),
                                      conv_bn_relu(32, 64),
                                      nn.MaxPool2d(kernel_size=3, stride=2),
                                      conv_bn_relu(64, 80, kernel_size=(1, 1)),
                                      conv_bn_relu(80, 192, padding='valid'),
                                      nn.MaxPool2d(kernel_size=3, stride=2))

        self.inception_blocks_A = nn.Sequential(InceptionBlock_A(192, 32),
                                                InceptionBlock_A(256, 64),
                                                InceptionBlock_A(288, 64))

        self.reduction_block_A = ReductionBlock_A(288)

        self.inception_blocks_B = nn.Sequential(InceptionBlock_B(768, 128),
                                                InceptionBlock_B(768, 160),
                                                InceptionBlock_B(768, 160),
                                                InceptionBlock_B(768, 192))

        self.reduction_block_B = ReductionBlock_B(768)

        self.inception_blocks_C = nn.Sequential(InceptionBlock_C(1280),
                                                InceptionBlock_C(2048))

        self.class_head = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                        nn.Flatten(),
                                        nn.Linear(2048, num_classes))

    def forward(self, x):
        x = self.prelayer(x)

        x = self.inception_blocks_A(x)
        x = self.reduction_block_A(x)

        x = self.inception_blocks_B(x)
        x = self.reduction_block_B(x)

        x = self.inception_blocks_C(x)

        x = self.class_head(x)

        return x


if __name__ == '__main__':
    model = InceptionV3()

    input = torch.randn((32, 3, 299, 299))
    out = model(input)
    print(
        f'Number of parameters: {sum(p.numel() for p in model.parameters())}')
