import torch
import torch.nn as nn
import numpy as np

# Configurations
num_blocks = {'18': np.array([2, 2, 2, 2]), '34': np.array([3, 4, 6, 3]), '50': np.array(
    [3, 4, 6, 3]), '101': np.array([3, 4, 23, 3]), '152': np.array([3, 8, 36, 3])}
num_filters = {'18': np.array([64, 64]), '34': np.array([64, 64]), '50': np.array(
    [64, 64, 256]), '101': np.array([64, 64, 256]), '152': np.array([64, 64, 256])}
kernels = {'18': np.array([(3, 3), (3, 3)]), '34': np.array([(3, 3), (3, 3)]), '50': np.array(
    [(1, 1), (3, 3), (1, 1)]), '101': np.array([(1, 1), (3, 3), (1, 1)]), '152': np.array([(1, 1), (3, 3), (1, 1)])}


def conv_bn_relu(inchannels, outchannels, kernel_size=(3, 3), stride=(1, 1), padding='same'):
    return [nn.Conv2d(inchannels, outchannels, kernel_size, padding=padding, stride=stride), nn.BatchNorm2d(outchannels), nn.ReLU()]

# debug


def print_output_shape(model, input, output):
    print(output.shape)


class Basic_Block(nn.Module):
    def __init__(self, inchannels, filters, kernel_sizes, stride=(1, 1)):
        super().__init__()
        assert len(filters) == len(kernel_sizes)

        self.res = []
        self.res.extend(conv_bn_relu(
            inchannels, filters[0], kernel_sizes[0], stride=stride, padding=kernel_sizes[0]//2))
        for i in range(1, len(filters)):
            self.res.extend(conv_bn_relu(
                filters[i-1], filters[i], kernel_sizes[i]))
        self.res.pop()
        self.res = nn.Sequential(*self.res)

        self.shortcut = None
        if inchannels != filters[-1]:
            self.shortcut = conv_bn_relu(
                inchannels, filters[-1], kernel_size=(1, 1), stride=stride, padding=0)
            self.shortcut.pop()
            self.shortcut = nn.Sequential(*self.shortcut)

        # debug
        self.register_forward_hook(print_output_shape)

    def forward(self, x):
        res = self.res(x)
        if self.shortcut is not None:
            x = self.shortcut(x)

        return nn.ReLU()(res+x)


class ResNet(nn.Module):
    def __init__(self, num_block, num_filter, num_kernel, num_classes=1000):
        super().__init__()

        self.backbone = conv_bn_relu(
            3, 64, kernel_size=(7, 7), stride=(2, 2), padding=3)
        self.backbone.append(nn.MaxPool2d(
            kernel_size=(3, 3), stride=(2, 2), padding=1))

        inchannels = 64
        stride = (1, 1)
        for stage in range(4):
            if stage > 0:
                stride = (2, 2)
            for block in range(num_block[stage]):
                self.backbone.append(Basic_Block(
                    inchannels, num_filter*(2**stage), num_kernel, stride))
                stride = (1, 1)
                inchannels = num_filter[-1]*(2**stage)

        self.backbone = nn.Sequential(*self.backbone)

        self.class_head = nn.Sequential(nn.AdaptiveAvgPool2d(
            (1, 1)), nn.Flatten(), nn.Linear(inchannels, num_classes))

    def forward(self, x):
        x = self.backbone(x)
        out = self.class_head(x)
        return out


def ResNet18():
    return ResNet(num_blocks['18'], num_filters['18'], kernels['18'])


def ResNet34():
    return ResNet(num_blocks['34'], num_filters['34'], kernels['34'])


def ResNet50():
    return ResNet(num_blocks['50'], num_filters['50'], kernels['50'])


def ResNet101():
    return ResNet(num_blocks['101'], num_filters['101'], kernels['101'])


def ResNet152():
    return ResNet(num_blocks['152'], num_filters['152'], kernels['152'])


if __name__ == '__main__':
    model = ResNet152()

    input = torch.randn((4, 3, 224, 224))
    out = model(input)
    print(out.shape)
    print(
        f'Number of parameters: {sum(p.numel() for p in model.parameters())}')
