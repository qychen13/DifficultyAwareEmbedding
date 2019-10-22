import torch.nn as nn
import torch.nn.functional as functional


class ResidualBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(ResidualBlock, self).__init__()
        self.downsample = (inplanes != planes)
        stride = 1
        if self.downsample:
            stride = 2

        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if self.downsample:
            self.shotcut = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=2, bias=False),
                                         nn.BatchNorm2d(planes))

    def forward(self, ipt):
        opt = self.conv1(ipt)
        opt = self.bn1(opt)
        opt = functional.relu(opt, inplace=True)
        opt = self.conv2(opt)
        opt = self.bn2(opt)

        if self.downsample:
            opt = self.shotcut(ipt) + opt
        else:
            opt = ipt + opt

        return functional.relu(opt, inplace=True)


class ResNet(nn.Module):
    def __init__(self, n=9, class_num=100):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        layers = []
        for planes in [16, 32, 64]:
            if planes == 16:
                inplanes = planes
            else:
                inplanes = int(planes / 2)
            layers.append(ResidualBlock(inplanes, planes))
            for i in range(n - 1):
                layers.append(ResidualBlock(planes, planes))
        self.res_layers = nn.Sequential(*layers)
        self.fc = nn.Linear(64, class_num)

    def forward(self, ipt):
        opt = self.conv1(ipt)
        opt = self.bn1(opt)
        opt = functional.relu(opt, inplace=True)
        opt = self.res_layers(opt)
        opt = functional.avg_pool2d(opt, kernel_size=8)
        opt = opt.view(opt.size(0), -1)
        opt = self.fc(opt)

        return opt


def split_resent(resenet, split_size=2):
    """return parameters in splits"""
    splits = [[] for i in range(split_size)]
    for i in range(split_size):
        splits[i].append(dict(params=resenet.conv1.parameters()))
        splits[i].append(dict(params=resenet.bn1.parameters()))

    turn = 0
    for residual_block in resenet.res_layers.modules():
        if not isinstance(residual_block, ResidualBlock):
            continue
        if residual_block.downsample:
            for i in range(split_size):
                splits[i].append(
                    {'params': residual_block.shotcut.parameters()})

        splits[turn].append({'params': residual_block.conv1.parameters()})
        splits[turn].append({'params': residual_block.bn1.parameters()})
        splits[turn].append({'params': residual_block.conv2.parameters()})
        splits[turn].append({'params': residual_block.bn2.parameters()})
        turn = (turn + 1) % split_size

    for i in range(split_size):
        splits[i].append(dict(params=resenet.fc.parameters()))

    return splits
