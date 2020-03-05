import math
import torch
import torch.nn as nn

class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * block.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm(planes * block.expansion)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(residual)
        
        out += residual
        return self.relu(out)


class ResNeXtBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, base_width=4, cardinality=32):
        super(ResNeXtBottleneck, self).__init__()
        D = int(math.floor(planes * (base_width / 64.)) * cardinality)

        self.conv1 = nn.Conv2d(inplanes, D, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(D)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1, bias=False, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(D)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(D, planes * ResNeXtBottleneck.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * ResNeXtBottleneck.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(residual)
        
        out += residual
        return self.relu3(out)


class ResNeXt(nn.Module):
    def __init__(self, block, blocks, num_classes=1000):
        super(ResNeXt, self).__init__()
        self.inplanes = 64
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer2 = self._make_layer(block,  64, blocks[0], stride=1)
        self.layer3 = self._make_layer(block, 128, blocks[1], stride=2)
        self.layer4 = self._make_layer(block, 256, blocks[2], stride=2)
        self.layer5 = self._make_layer(block, 512, blocks[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, num_blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * block.expansion),
                # 注意这里没有 ReLU
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, stride=1))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def resnext50():
    return ResNeXt(ResNeXtBottleneck, [3, 4, 6, 3])

def resnext101():
    return ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3])

def resnext152():
    return ResNeXt(ResNeXtBottleneck, [3, 4, 36, 3])
