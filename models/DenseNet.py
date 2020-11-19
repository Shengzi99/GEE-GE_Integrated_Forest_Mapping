import torch.nn as nn
import torch.nn.functional as F
import torch


class BottleNeck(nn.Module):
    def __init__(self, in_ch, growth_rate):
        super().__init__()
        inner_ch = 4 * growth_rate
        self.bottle_neck = nn.Sequential(nn.BatchNorm2d(in_ch),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(in_ch, inner_ch, kernel_size=1, bias=False),
                                         nn.BatchNorm2d(inner_ch),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(inner_ch, growth_rate, kernel_size=3, padding=1, bias=False))

    def forward(self, x):
        return torch.cat([x, self.bottle_neck(x)], dim=1)


class Transition(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down_sample = nn.Sequential(nn.BatchNorm2d(in_ch),
                                         nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                                         nn.AvgPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        return self.down_sample(x)


class DenseNet(nn.Module):
    """
    Re-implementation of Huang, G., et al. (2017). Densely connected convolutional networks. CVPR
    """

    def __init__(self, n_blocks, growth_rate=12, comp_factor=0.5, in_ch=3, n_classes=5):
        super().__init__()
        # conv1---------------------------------------------------------------------------------------------------------
        self.growth_rate = growth_rate
        inner_ch = 2 * growth_rate
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, inner_ch, kernel_size=5, stride=2, padding=2, bias=False),
                                   nn.BatchNorm2d(inner_ch),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(inner_ch, inner_ch, kernel_size=3, stride=1, padding=2, bias=False))
        # dense blocks (1, 2, ... n)------------------------------------------------------------------------------------
        self.denseBlocks = nn.Sequential()
        for idx in range(len(n_blocks) - 1):
            # -----------------dense block---------------------------
            self.denseBlocks.add_module("DenseBlock%d" % (idx + 1), self.__make_dense_layers__(inner_ch, n_blocks[idx]))
            inner_ch += growth_rate * n_blocks[idx]
            # ---------------transition layer------------------------
            trans_out_ch = int(comp_factor * inner_ch)
            self.denseBlocks.add_module("TransLayer%d" % (idx + 1), Transition(inner_ch, trans_out_ch))
            inner_ch = trans_out_ch
        self.denseBlocks.add_module("DenseBlock%d" % len(n_blocks), self.__make_dense_layers__(inner_ch, n_blocks[-1]))
        inner_ch += growth_rate * n_blocks[-1]
        self.denseBlocks.add_module("bn", nn.BatchNorm2d(inner_ch))
        self.denseBlocks.add_module("relu", nn.ReLU(inplace=True))
        # global average pooling & fc layer-----------------------------------------------------------------------------
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(inner_ch, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.denseBlocks(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        output = self.fc(x)
        return output

    def __make_dense_layers__(self, in_ch, n_blocks):
        dense_block = nn.Sequential()
        for idx in range(n_blocks):
            dense_block.add_module("bottle_neck%d" % idx, BottleNeck(in_ch, self.growth_rate))
            in_ch += self.growth_rate
        return dense_block


def denseNet121(in_ch, n_classes):
    return DenseNet([6, 12, 24, 16], growth_rate=32, comp_factor=0.5, in_ch=in_ch, n_classes=n_classes)


def denseNet169(in_ch, n_classes):
    return DenseNet([6, 12, 32, 32], growth_rate=32, comp_factor=0.5, in_ch=in_ch, n_classes=n_classes)


def denseNet201(in_ch, n_classes):
    return DenseNet([6, 12, 48, 32], growth_rate=32, comp_factor=0.5, in_ch=in_ch, n_classes=n_classes)


def denseNet161(in_ch, n_classes):
    return DenseNet([6, 12, 36, 24], growth_rate=48, comp_factor=0.5, in_ch=in_ch, n_classes=n_classes)
