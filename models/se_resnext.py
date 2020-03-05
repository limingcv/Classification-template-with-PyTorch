import math
import torch
import torch.nn as nn

class SEModule(nn.Module):
    def __init__(self, planes, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Conv2d(planes, planes // reduction, kernel_size=1, padding=0)
        self.fc2 = nn.Conv2d(planes // reduction, planes, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return x * out
        

class SEResNeXtBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, reduction, base_width=4, cardinality=32, downsample=None, stride=1):
        super(SEResNeXtBottleneck, self).__init__()
        D = int(math.floor(planes * (base_width / 64.)) * cardinality)

        self.conv1 = nn.Conv2d(inplanes, D, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(D)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1, bias=False, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(D)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(D, planes * SEResNeXtBottleneck.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * SEResNeXtBottleneck.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.se = SEModule(planes * SEResNeXtBottleneck.expansion, reduction)


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
        
        # 在残差相加之前进行 se 变换, 目的应该是为了在通道数最多的特征图上做 se
        out = self.se(out)
        out = out + residual  
        return self.relu3(out)


class SEResNeXt(nn.Module):
    def __init__(self, block, blocks, reduction, cardinality, base_width, num_classes=1000):
        super(SEResNeXt, self).__init__()
        self.inplanes = 64
        self.layer1 = nn.Sequential(
            # 看官网源码有 nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer2 = self._make_layer(block,  64, blocks[0], reduction, cardinality, base_width, stride=1)
        self.layer3 = self._make_layer(block, 128, blocks[1], reduction, cardinality, base_width, stride=2)
        self.layer4 = self._make_layer(block, 256, blocks[2], reduction, cardinality, base_width, stride=2)
        self.layer5 = self._make_layer(block, 512, blocks[3], reduction, cardinality, base_width, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, num_blocks, reduction, cardinality, base_width, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, reduction=reduction, 
                            base_width=4, cardinality=32, downsample=downsample))
        self.inplanes = planes * block.expansion

        # 注意这里不再需要 downsample 了
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, stride=1, reduction=reduction,
                                base_width=4, cardinality=32))
        
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


def se_resnext50(num_classes):
    return SEResNeXt(SEResNeXtBottleneck, [3, 4,  6, 3], reduction=16, cardinality=32, base_width=4, num_classes=num_classes)

def se_resnext101(num_classes):
    return SEResNeXt(SEResNeXtBottleneck, [3, 4, 23, 3], reduction=16, cardinality=32, base_width=4, num_classes=num_classes)

def se_resnext152(num_classes):
    return SEResNeXt(SEResNeXtBottleneck, [3, 4, 36, 3], reduction=16, cardinality=32, base_width=4, num_classes=num_classes)
