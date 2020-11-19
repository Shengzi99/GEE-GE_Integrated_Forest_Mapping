import torch.nn as nn
import torch.nn.functional as F
import torch


class BottleNeckResidualSEBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, stride, r=16):
        super().__init__()

        self.residual = nn.Sequential(nn.Conv2d(in_ch, mid_ch, 1),
                                      nn.BatchNorm2d(mid_ch),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1),
                                      nn.BatchNorm2d(mid_ch),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(mid_ch, mid_ch * 4, 1),
                                      nn.BatchNorm2d(mid_ch * 4),
                                      nn.ReLU(inplace=True))

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(nn.Linear(mid_ch * 4, mid_ch * 4 // r),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(mid_ch * 4 // r, mid_ch * 4),
                                        nn.Sigmoid())

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != (mid_ch * 4):
            self.shortcut = nn.Sequential(nn.Conv2d(in_ch, mid_ch * 4, 1, stride=stride),
                                          nn.BatchNorm2d(mid_ch * 4))

    def forward(self, x):
        shortcut = self.shortcut(x)
        residual = self.residual(x)

        squeeze = self.squeeze(residual)
        squeeze = squeeze.view(squeeze.shape[0], -1)
        excitation = self.excitation(squeeze)
        excitation = excitation.view(residual.shape[0], -1, 1, 1)

        output = F.relu(residual * excitation + shortcut)
        return output


class SEResNet(nn.Module):
    """ a modified SE-ResNet inorder to get a short FC output feature vector """
    def __init__(self, n_blocks, in_ch, n_classes=5):
        self.feature_vec = 0
        featureVectorSize = 16
        super().__init__()
        # conv1---------------------------------------------------------------------------------------------------------
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, 64, kernel_size=5, padding=2, stride=2),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
                                   nn.MaxPool2d(kernel_size=3, stride=2))
        # conv2~5-------------------------------------------------------------------------------------------------------
        self.conv2 = self.__makeResBlocks__(n_blocks[0], 64, 64, 1)
        self.conv3 = self.__makeResBlocks__(n_blocks[1], 256, 128, 2)
        self.conv4 = self.__makeResBlocks__(n_blocks[2], 512, 256, 2)
        self.conv5 = self.__makeResBlocks__(n_blocks[3], 1024, 512, 2)

        # global average pooling & fc-----------------------------------------------------------------------------------
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(2048, featureVectorSize)
        self.fc2 = nn.Linear(featureVectorSize, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        self.feature_vec_2048 = x
        self.feature_vec = F.relu(self.fc1(x))
        output = self.fc2(self.feature_vec)

        return output

    def __makeResBlocks__(self, n_blocks, in_ch, mid_ch, stride):
        layers = nn.Sequential()
        layers.add_module("bottleneck_res_SE_block0", BottleNeckResidualSEBlock(in_ch, mid_ch, stride))
        for i in range(n_blocks - 1):
            layers.add_module("bottleneck_res_SE_block%d" % (i + 1), BottleNeckResidualSEBlock(mid_ch * 4, mid_ch, 1))
        return layers


def se_resNet50(in_ch, n_classes):
    return SEResNet(n_blocks=[3, 4, 6, 3], in_ch=in_ch, n_classes=n_classes)


def se_resNet101(in_ch, n_classes):
    return SEResNet(n_blocks=[3, 4, 23, 3], in_ch=in_ch, n_classes=n_classes)


def se_resNet152(in_ch, n_classes):
    return SEResNet(n_blocks=[3, 8, 36, 3], in_ch=in_ch, n_classes=n_classes)
