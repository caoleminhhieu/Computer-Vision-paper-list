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


class Stem_v1(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(conv_bn_relu(3, 32, kernel_size=(3, 3), stride=(2, 2), padding='valid'),
                                  conv_bn_relu(32, 32, kernel_size=(
                                      3, 3), padding='valid'),
                                  conv_bn_relu(32, 64, kernel_size=(3, 3)),
                                  nn.MaxPool2d(kernel_size=(
                                      3, 3), stride=(2, 2)),
                                  conv_bn_relu(64, 80, kernel_size=(1, 1)),
                                  conv_bn_relu(80, 192, kernel_size=(
                                      3, 3), padding='valid'),
                                  conv_bn_relu(192, 256, kernel_size=(3, 3), stride=(2, 2), padding='valid'))

        ### For debugging feature map dimension only ###
        # self.register_forward_hook(print_output_shape)

    def forward(self, x):
        return self.stem(x)


class Stem_v2(nn.Module):
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


class Inception_Resnet_A(nn.Module):
    def __init__(self, inchannels, num_filters, scale=0.1):
        super().__init__()
        self.scale = scale
        self.branch_1x1 = conv_bn_relu(
            inchannels, num_filters[0], kernel_size=(1, 1))
        self.branch_3x3 = nn.Sequential(conv_bn_relu(inchannels, num_filters[1], kernel_size=(1, 1)),
                                        conv_bn_relu(num_filters[1], num_filters[2], kernel_size=(3, 3)))
        self.branch_3x3db = nn.Sequential(conv_bn_relu(inchannels, num_filters[3], kernel_size=(1, 1)),
                                          conv_bn_relu(
                                              num_filters[3], num_filters[4], kernel_size=(3, 3)),
                                          conv_bn_relu(num_filters[4], num_filters[5], kernel_size=(3, 3)))
        self.up_channel = nn.Conv2d(
            num_filters[0] + num_filters[2] + num_filters[5], num_filters[6], kernel_size=(1, 1))

        ### For debugging feature map dimension only ###
        # self.branch_1x1.register_forward_hook(print_output_shape)
        # self.branch_3x3.register_forward_hook(print_output_shape)
        # self.branch_3x3db.register_forward_hook(print_output_shape)
        # self.up_channel.register_forward_hook(print_output_shape)
        # self.register_forward_hook(print_output_shape)

    def forward(self, x):
        b1 = self.branch_1x1(x)
        b3 = self.branch_3x3(x)
        b3db = self.branch_3x3db(x)
        up = self.up_channel(torch.cat([b1, b3, b3db], dim=1)) * self.scale

        return nn.ReLU()(up+x)


class Inception_Resnet_B(nn.Module):
    def __init__(self, inchannels, num_filters, scale=0.1):
        super().__init__()
        self.scale = scale

        self.branch_1x1 = conv_bn_relu(
            inchannels, num_filters[0], kernel_size=(1, 1))
        self.branch_7x7 = nn.Sequential(conv_bn_relu(inchannels, num_filters[1], kernel_size=(1, 1)),
                                        conv_bn_relu(
                                            num_filters[1], num_filters[2], kernel_size=(1, 7)),
                                        conv_bn_relu(num_filters[2], num_filters[3], kernel_size=(7, 1)))
        self.up_channel = nn.Conv2d(
            num_filters[0] + num_filters[3], num_filters[4], kernel_size=(1, 1))

        ### For debugging feature map dimension only ###
        # self.branch_1x1.register_forward_hook(print_output_shape)
        # self.branch_7x7.register_forward_hook(print_output_shape)
        # self.up_channel.register_forward_hook(print_output_shape)
        # self.register_forward_hook(print_output_shape)

    def forward(self, x):
        b1 = self.branch_1x1(x)
        b7 = self.branch_7x7(x)
        up = self.up_channel(torch.cat([b1, b7], dim=1)) * self.scale

        return nn.ReLU()(up+x)


class Inception_Resnet_C(nn.Module):
    def __init__(self, inchannels, num_filters, scale=0.1):
        super().__init__()
        self.scale = scale

        self.branch_1x1 = conv_bn_relu(
            inchannels, num_filters[0], kernel_size=(1, 1))
        self.branch_3x3 = nn.Sequential(conv_bn_relu(inchannels, num_filters[1], kernel_size=(1, 1)),
                                        conv_bn_relu(
                                            num_filters[1], num_filters[2], kernel_size=(1, 3)),
                                        conv_bn_relu(num_filters[2], num_filters[3], kernel_size=(3, 1)))
        self.up_channel = nn.Conv2d(
            num_filters[0] + num_filters[3], num_filters[4], kernel_size=(1, 1))

        ### For debugging feature map dimension only ###
        # self.branch_1x1.register_forward_hook(print_output_shape)
        # self.branch_3x3.register_forward_hook(print_output_shape)
        # self.up_channel.register_forward_hook(print_output_shape)
        # self.register_forward_hook(print_output_shape)

    def forward(self, x):
        b1 = self.branch_1x1(x)
        b3 = self.branch_3x3(x)
        up = self.up_channel(torch.cat([b1, b3], dim=1)) * self.scale

        return nn.ReLU()(up+x)


class ReductionBlock_A(nn.Module):
    def __init__(self, inchannels, num_filters):
        super().__init__()
        self.branch_pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        self.branch_3x3 = conv_bn_relu(inchannels, num_filters[0], kernel_size=(
            3, 3), stride=(2, 2), padding='valid')

        self.branch_3x3db = nn.Sequential(conv_bn_relu(inchannels, num_filters[1], kernel_size=(1, 1)),
                                          conv_bn_relu(
                                              num_filters[1], num_filters[2], kernel_size=(3, 3)),
                                          conv_bn_relu(num_filters[2], num_filters[3], kernel_size=(3, 3), stride=(2, 2), padding='valid'))

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
    def __init__(self, inchannels, num_filters):
        super().__init__()
        self.branch_pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        self.branch_3x3_1 = nn.Sequential(conv_bn_relu(inchannels, num_filters[0], kernel_size=(1, 1)),
                                          conv_bn_relu(num_filters[0], num_filters[1], kernel_size=(3, 3), stride=(2, 2), padding='valid'))

        self.branch_3x3_2 = nn.Sequential(conv_bn_relu(inchannels, num_filters[2], kernel_size=(1, 1)),
                                          conv_bn_relu(num_filters[2], num_filters[3], kernel_size=(3, 3), stride=(2, 2), padding='valid'))

        self.branch_3x3db = nn.Sequential(conv_bn_relu(inchannels, num_filters[4], kernel_size=(1, 1)),
                                          conv_bn_relu(
                                              num_filters[4], num_filters[5], kernel_size=(3, 3)),
                                          conv_bn_relu(num_filters[5], num_filters[6], kernel_size=(3, 3), stride=(2, 2), padding='valid'))

        ### For debugging feature map dimension only ###
        # self.branch_pool.register_forward_hook(print_output_shape)
        # self.branch_3x3_1.register_forward_hook(print_output_shape)
        # self.branch_3x3_2.register_forward_hook(print_output_shape)
        # self.branch_3x3db.register_forward_hook(print_output_shape)
        # self.register_forward_hook(print_output_shape)

    def forward(self, x):
        bp = self.branch_pool(x)
        b3_1 = self.branch_3x3_1(x)
        b3_2 = self.branch_3x3_2(x)
        b3db = self.branch_3x3db(x)

        return torch.cat([bp, b3_1, b3_2, b3db], dim=1)


class Inception_Resnet(nn.Module):
    def __init__(self, version=1, num_classes=1000, scale=0.1):
        super().__init__()

        if version == 1:
            self.stem = Stem_v1()
            self.inception_blocksA = [Inception_Resnet_A(
                256, [32, 32, 32, 32, 32, 32, 256], scale) for _ in range(5)]
            self.reduction_blockA = ReductionBlock_A(256, [384, 192, 192, 256])

            self.inception_blocksB = [Inception_Resnet_B(
                896, [128, 128, 128, 128, 896], scale) for _ in range(10)]
            self.reduction_blockB = ReductionBlock_B(
                896, [256, 384, 256, 256, 256, 256, 256])

            self.inception_blocksC = [Inception_Resnet_C(
                1792, [192, 192, 192, 192, 1792], scale) for _ in range(5)]

            self.class_head = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Dropout2d(
                0.2), nn.Conv2d(1792, num_classes, kernel_size=(1, 1)), nn.Flatten())
        else:
            self.stem = Stem_v2()
            self.inception_blocksA = [Inception_Resnet_A(
                384, [32, 32, 32, 32, 48, 64, 384], scale) for _ in range(5)]
            self.reduction_blockA = ReductionBlock_A(384, [384, 256, 256, 384])

            self.inception_blocksB = [Inception_Resnet_B(
                1152, [192, 128, 160, 192, 1152], scale) for _ in range(10)]
            self.reduction_blockB = ReductionBlock_B(
                1152, [256, 384, 256, 288, 256, 288, 320])

            self.inception_blocksC = [Inception_Resnet_C(
                2144, [192, 192, 224, 256, 2144], scale) for _ in range(5)]

            self.class_head = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Dropout2d(
                0.2), nn.Conv2d(2144, num_classes, kernel_size=(1, 1)), nn.Flatten())

        ### For debugging feature map dimension only ###
        self.stem.register_forward_hook(print_output_shape)
        for m in self.inception_blocksA:
            m.register_forward_hook(print_output_shape)
        self.reduction_blockA.register_forward_hook(print_output_shape)
        for m in self.inception_blocksB:
            m.register_forward_hook(print_output_shape)
        self.reduction_blockB.register_forward_hook(print_output_shape)
        for m in self.inception_blocksC:
            m.register_forward_hook(print_output_shape)
        self.class_head.register_forward_hook(print_output_shape)

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
    model = Inception_Resnet(version=2)

    input = torch.randn((4, 3, 299, 299))
    out = model(input)
