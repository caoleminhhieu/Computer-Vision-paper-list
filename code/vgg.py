from turtle import forward
import torch
import torch.nn as nn

cfg = {'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
       'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
       'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
       'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']}


class VGG(nn.Module):
    def __init__(self, cfg, batch_norm=False, num_classes=1000):
        super().__init__()

        self.feature_extractor = self._make_layers(cfg, batch_norm)

        self.class_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        out = self.feature_extractor(x)
        print(out.shape)
        out = self.class_head(out)
        return out

    def _make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for layer in cfg:
            if layer == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                continue

            layers.append(
                nn.Conv2d(in_channels, out_channels=layer, kernel_size=3, padding=1))
            if batch_norm:
                layers.append(nn.BatchNorm2d(layer))

            layers.append(nn.ReLU())

            in_channels = layer

        return nn.Sequential(*layers)


def vgg11_bn():
    return VGG(cfg=cfg['A'], batch_norm=True)


def vgg13_bn():
    return VGG(cfg=cfg['B'], batch_norm=True)


def vgg16_bn():
    return VGG(cfg=cfg['D'], batch_norm=True)


def vgg19_bn():
    return VGG(cfg=cfg['E'], batch_norm=True)


if __name__ == '__main__':
    model = vgg19_bn()
    print(model)

    input = torch.randn((32, 3, 224, 224))
    out = model(input)
    print(out.shape)

    target = torch.randint(low=0, high=1000, size=(32,))
    print(nn.CrossEntropyLoss()(out, target))
    print(
        f'Number of parameters: {sum(p.numel() for p in model.parameters())}')
