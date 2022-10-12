from mimetypes import init
import torch
import torch.nn as nn


def conv_bn_relu(inchannels, outchannels, kernel_size, stride=1, padding='same'):
    return nn.Sequential(
        nn.Conv2d(inchannels, outchannels, kernel_size,
                  padding=padding, stride=stride),
        nn.BatchNorm2d(outchannels),
        nn.ReLU()
    )


def print_output_shape(model, input, output):
    print(output.shape)


class Stem(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(conv_bn_relu(3, 32, kernel_size=(3, 3), stride=(2, 2), padding='valid'),
                                   conv_bn_relu(32, 32, kernel_size=(
                                       3, 3), padding='valid'),
                                   conv_bn_relu(32, 64, kernel_size=(3, 3)))

        self.down1 = nn.ModuleList(modules=[nn.MaxPool2d(kernel_size=(3, 3), stride=(
            2, 2)), conv_bn_relu(64, 96, kernel_size=(3, 3), stride=(2, 2), padding='valid')])

        self.conv2_1 = nn.Sequential(conv_bn_relu(160, 64, kernel_size=(1, 1)),
                                     conv_bn_relu(64, 96, kernel_size=(3, 3), padding='valid'))

        self.conv2_2 = nn.Sequential(conv_bn_relu(160, 64, kernel_size=(1, 1)),
                                     conv_bn_relu(64, 64, kernel_size=(7, 1)),
                                     conv_bn_relu(64, 64, kernel_size=(1, 7)),
                                     conv_bn_relu(64, 96, kernel_size=(3, 3), padding='valid'))

        self.down2 = nn.ModuleList(modules=[nn.MaxPool2d(kernel_size=(3, 3), stride=(
            2, 2)), conv_bn_relu(192, 192, kernel_size=(3, 3), stride=(2, 2), padding='valid')])

        ### For debugging feature map dimension only ###
        # self.conv1.register_forward_hook(print_output_shape)
        # for m in self.down1:
        #     m.register_forward_hook(print_output_shape)
        # self.conv2_1.register_forward_hook(print_output_shape)
        # self.conv2_2.register_forward_hook(print_output_shape)
        # for m in self.down2:
        #     m.register_forward_hook(print_output_shape)

        # self.register_forward_hook(print_output_shape)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([m(x) for m in self.down1], dim=1)

        x = torch.cat([self.conv2_1(x), self.conv2_2(x)], dim=1)
        x = torch.cat([m(x) for m in self.down2], dim=1)

        return x


class InceptionBlock_A(nn.Module):
    def __init__(self, inchannels):
        super().__init__()
        self.branch_pool = nn.Sequential(nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1), padding=1),
                                         conv_bn_relu(inchannels, 96, kernel_size=(1, 1)))

        self.branch_1x1 = conv_bn_relu(inchannels, 96, kernel_size=(1, 1))
        self.branch_3x3 = nn.Sequential(conv_bn_relu(inchannels, 64, kernel_size=(1, 1)),
                                        conv_bn_relu(64, 96, kernel_size=(3, 3)))
        self.branch_3x3db = nn.Sequential(conv_bn_relu(inchannels, 64, kernel_size=(1, 1)),
                                          conv_bn_relu(
                                              64, 96, kernel_size=(3, 3)),
                                          conv_bn_relu(96, 96, kernel_size=(3, 3)))

        ### For debugging feature map dimension only ###
        # self.branch_pool.register_forward_hook(print_output_shape)
        # self.branch_1x1.register_forward_hook(print_output_shape)
        # self.branch_3x3.register_forward_hook(print_output_shape)
        # self.branch_3x3db.register_forward_hook(print_output_shape)
        # self.register_forward_hook(print_output_shape)

    def forward(self, x):
        bp = self.branch_pool(x)
        b1 = self.branch_1x1(x)
        b3 = self.branch_3x3(x)
        b3db = self.branch_3x3db(x)

        return torch.cat([bp, b1, b3, b3db], dim=1)


class InceptionBlock_B(nn.Module):
    def __init__(self, inchannels):
        super().__init__()
        self.branch_pool = nn.Sequential(nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1), padding=1),
                                         conv_bn_relu(inchannels, 128, kernel_size=(1, 1)))

        self.branch_1x1 = conv_bn_relu(inchannels, 384, kernel_size=(1, 1))
        self.branch_7x7 = nn.Sequential(conv_bn_relu(inchannels, 192, kernel_size=(1, 1)),
                                        conv_bn_relu(
                                            192, 224, kernel_size=(1, 7)),
                                        conv_bn_relu(224, 256, kernel_size=(7, 1)))
        self.branch_7x7db = nn.Sequential(conv_bn_relu(inchannels, 192, kernel_size=(1, 1)),
                                          conv_bn_relu(
                                              192, 192, kernel_size=(1, 7)),
                                          conv_bn_relu(
                                              192, 224, kernel_size=(7, 1)),
                                          conv_bn_relu(
                                              224, 224, kernel_size=(1, 7)),
                                          conv_bn_relu(224, 256, kernel_size=(7, 1)))

        ### For debugging feature map dimension only ###
        # self.branch_pool.register_forward_hook(print_output_shape)
        # self.branch_1x1.register_forward_hook(print_output_shape)
        # self.branch_7x7.register_forward_hook(print_output_shape)
        # self.branch_7x7db.register_forward_hook(print_output_shape)
        # self.register_forward_hook(print_output_shape)

    def forward(self, x):
        bp = self.branch_pool(x)
        b1 = self.branch_1x1(x)
        b7 = self.branch_7x7(x)
        b7db = self.branch_7x7db(x)

        return torch.cat([bp, b1, b7, b7db], dim=1)


class InceptionBlock_C(nn.Module):
    def __init__(self, inchannels):
        super().__init__()
        self.branch_pool = nn.Sequential(nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1), padding=1),
                                         conv_bn_relu(inchannels, 256, kernel_size=(1, 1)))

        self.branch_1x1 = conv_bn_relu(inchannels, 256, kernel_size=(1, 1))

        self.branch_3x3 = conv_bn_relu(inchannels, 384, kernel_size=(1, 1))
        self.branch_3x3_1 = conv_bn_relu(384, 256, kernel_size=(1, 3))
        self.branch_3x3_2 = conv_bn_relu(384, 256, kernel_size=(3, 1))

        self.branch_3x3db = nn.Sequential(conv_bn_relu(inchannels, 384, kernel_size=(1, 1)),
                                          conv_bn_relu(
                                              384, 448, kernel_size=(1, 3)),
                                          conv_bn_relu(448, 512, kernel_size=(3, 1)))
        self.branch_3x3db_1 = conv_bn_relu(512, 256, kernel_size=(1, 3))
        self.branch_3x3db_2 = conv_bn_relu(512, 256, kernel_size=(3, 1))

        ### For debugging feature map dimension only ###
        # self.branch_pool.register_forward_hook(print_output_shape)
        # self.branch_1x1.register_forward_hook(print_output_shape)
        # self.branch_3x3_1.register_forward_hook(print_output_shape)
        # self.branch_3x3_2.register_forward_hook(print_output_shape)
        # self.branch_3x3db_1.register_forward_hook(print_output_shape)
        # self.branch_3x3db_2.register_forward_hook(print_output_shape)
        # self.register_forward_hook(print_output_shape)

    def forward(self, x):
        bp = self.branch_pool(x)
        b1 = self.branch_1x1(x)

        b3 = self.branch_3x3(x)
        b3_1 = self.branch_3x3_1(b3)
        b3_2 = self.branch_3x3_2(b3)

        b3db = self.branch_3x3db(x)
        b3db_1 = self.branch_3x3db_1(b3db)
        b3db_2 = self.branch_3x3db_2(b3db)

        return torch.cat([bp, b1, b3_1, b3_2, b3db_1, b3db_2], dim=1)


class ReductionBlock_A(nn.Module):
    def __init__(self, inchannels):
        super().__init__()
        self.branch_pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        self.branch_3x3 = conv_bn_relu(inchannels, 384, kernel_size=(
            3, 3), stride=(2, 2), padding='valid')

        self.branch_3x3db = nn.Sequential(conv_bn_relu(inchannels, 192, kernel_size=(1, 1)),
                                          conv_bn_relu(
                                              192, 224, kernel_size=(3, 3)),
                                          conv_bn_relu(224, 256, kernel_size=(3, 3), stride=(2, 2), padding='valid'))

        ### For debugging feature map dimension only ###
        # self.branch_pool.register_forward_hook(print_output_shape)
        # self.branch_3x3.register_forward_hook(print_output_shape)
        # self.branch_3x3db.register_forward_hook(print_output_shape)
        # self.register_forward_hook(print_output_shape)

    def forward(self, x):
        bp = self.branch_pool(x)
        b3 = self.branch_3x3(x)
        b3db = self.branch_3x3db(x)

        return torch.cat([bp, b3, b3db], dim=1)


class ReductionBlock_B(nn.Module):
    def __init__(self, inchannels):
        super().__init__()
        self.branch_pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        self.branch_3x3 = nn.Sequential(conv_bn_relu(inchannels, 192, kernel_size=(1, 1)),
                                        conv_bn_relu(192, 192, kernel_size=(3, 3), stride=(2, 2), padding='valid'))

        self.branch_7x7 = nn.Sequential(conv_bn_relu(inchannels, 256, kernel_size=(1, 1)),
                                        conv_bn_relu(
                                            256, 256, kernel_size=(1, 7)),
                                        conv_bn_relu(
                                            256, 320, kernel_size=(7, 1)),
                                        conv_bn_relu(320, 320, kernel_size=(3, 3), stride=(2, 2), padding='valid'))

        ### For debugging feature map dimension only ###
        # self.branch_pool.register_forward_hook(print_output_shape)
        # self.branch_3x3.register_forward_hook(print_output_shape)
        # self.branch_7x7.register_forward_hook(print_output_shape)
        # self.register_forward_hook(print_output_shape)

    def forward(self, x):
        bp = self.branch_pool(x)
        b3 = self.branch_3x3(x)
        b7 = self.branch_7x7(x)

        return torch.cat([bp, b3, b7], dim=1)


class InceptionV4(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = Stem()
        self.inception_blocksA = [InceptionBlock_A(384) for _ in range(4)]
        self.reduction_blockA = ReductionBlock_A(384)

        self.inception_blocksB = [InceptionBlock_B(1024) for _ in range(7)]
        self.reduction_blockB = ReductionBlock_B(1024)

        self.inception_blocksC = [InceptionBlock_C(1536) for _ in range(3)]

        self.class_head = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Dropout2d(
            0.2), nn.Conv2d(1536, num_classes, kernel_size=(1, 1)), nn.Flatten())

        ### For debugging feature map dimension only ###
        # self.stem.register_forward_hook(print_output_shape)
        # for m in self.inception_blocksA:
        #     m.register_forward_hook(print_output_shape)
        # self.reduction_blockA.register_forward_hook(print_output_shape)
        # for m in self.inception_blocksB:
        #     m.register_forward_hook(print_output_shape)
        # self.reduction_blockB.register_forward_hook(print_output_shape)
        # for m in self.inception_blocksC:
        #     m.register_forward_hook(print_output_shape)
        # self.class_head.register_forward_hook(print_output_shape)

    def forward(self, x):
        x = self.stem(x)

        for m in self.inception_blocksA:
            x = m(x)
        x = self.reduction_blockA(x)

        for m in self.inception_blocksB:
            x = m(x)
        x = self.reduction_blockB(x)

        for m in self.inception_blocksC:
            x = m(x)
        x = self.class_head(x)

        return x


if __name__ == '__main__':
    model = InceptionV4()

    input = torch.randn((4, 3, 299, 299))
    out = model(input)
