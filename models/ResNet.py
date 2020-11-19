import torch.nn as nn
import torch.nn.functional as F
import torch


class ResNet101(nn.Module):
    """
    Re-implementation of He, K., et al. (2016). Deep Residual Learning for Image Recognition. CVPR
    """
    def __init__(self, in_ch=3, n_classes=4):
        super().__init__()
        # Conv1---------------------------------------------------------------------------------------------------------
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                                   nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        # Conv2---------------------------------------------------------------------------------------------------------
        self.conv2 = nn.Sequential(nn.MaxPool2d(3, stride=2, padding=1), Resblock(in_ch=64, out_ch=256))
        for i in range(2):
            self.conv2.add_module("res_block_2_%d" % (i+2), Resblock(in_ch=256, out_ch=256))
        # Conv3---------------------------------------------------------------------------------------------------------
        self.conv3 = nn.Sequential(nn.MaxPool2d(2, stride=2, padding=0), Resblock(in_ch=256, out_ch=512))
        for i in range(3):
            self.conv3.add_module("res_block_3_%d" % (i+2), Resblock(in_ch=512, out_ch=512))
        # Conv4---------------------------------------------------------------------------------------------------------
        self.conv4 = nn.Sequential(nn.MaxPool2d(2, stride=2, padding=0), Resblock(in_ch=512, out_ch=1024))
        for i in range(22):
            self.conv4.add_module("res_block_4_%d" % (i+2), Resblock(in_ch=1024, out_ch=1024))
        # Conv5---------------------------------------------------------------------------------------------------------
        self.conv5 = nn.Sequential(nn.MaxPool2d(2, stride=2, padding=0), Resblock(in_ch=1024, out_ch=2048),
                                   Resblock(in_ch=2048, out_ch=2048), Resblock(in_ch=2048, out_ch=2048))
        # avgPool+FC----------------------------------------------------------------------------------------------------
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, n_classes)

    def forward(self, inputs):
        inputs = self.conv1(inputs)
        inputs = self.conv2(inputs)
        inputs = self.conv3(inputs)
        inputs = self.conv4(inputs)
        inputs = self.conv5(inputs)

        inputs = self.avgpool(inputs)
        inputs = inputs.view(inputs.shape[0], -1)
        outputs = self.fc(inputs)

        return outputs


class Resblock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.ch_asc = (in_ch != out_ch)
        mid_ch = out_ch//4
        self.conv = nn.Sequential(nn.Conv2d(in_ch, mid_ch, 1, 1, padding=0), nn.BatchNorm2d(mid_ch), nn.ReLU(),
                                  nn.Conv2d(mid_ch, mid_ch, 3, stride, padding=1), nn.BatchNorm2d(mid_ch), nn.ReLU(),
                                  nn.Conv2d(mid_ch, out_ch, 1, 1, padding=0), nn.BatchNorm2d(out_ch))
        if self.ch_asc:
            self.shortcut = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, 1, 0), nn.BatchNorm2d(out_ch))

    def forward(self, x_in):
        x = self.conv(x_in)
        if self.ch_asc:
            return F.relu(x + self.shortcut(x_in))
        else:
            return F.relu(x + x_in)
