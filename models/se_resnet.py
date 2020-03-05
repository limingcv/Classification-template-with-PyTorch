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



class BasicBlock(nn.Module):
    # 基本残差块的输入通道和输出通道是一样的, 所以这里 expansion = 1
    expansion = 1

    def __init__(self, inplanes, planes, reduction, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.downsample = downsample
        self.se = SEModule(planes, reduction)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = self.se(out)
        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    初始时候 inplanes == planes * Bottleneck.expansion的, 所以第一个卷积先把通道数降为原来的expansion倍, 
    用第三个卷积再升维回到inplanes
    """
    # 瓶颈块的出入通道数是输出通道的4倍, 所以这里 expansion = 4
    expansion = 4

    def __init__(self, inplanes, planes, reduction, stride=1, downsample=None):
        
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)

        self.downsample = downsample
        self.se = SEModule(planes * Bottleneck.expansion, reduction)

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

        out = self.se(out)
        out += residual

        return self.relu(out)


class SEResNet(nn.Module):
    def __init__(self, block, blocks, reduction, num_classes=1000):
        super(SEResNet, self).__init__()
        self.inplanes = 64
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer2 = self._make_layer(block,  64, blocks[0], reduction, stride=1)
        self.layer3 = self._make_layer(block, 128, blocks[1], reduction, stride=2)
        self.layer4 = self._make_layer(block, 256, blocks[2], reduction, stride=2)
        self.layer5 = self._make_layer(block, 512, blocks[3], reduction, stride=2)  # 这里的512是planes数, 最后的得到的通道数应该是 512* block.expansion
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, reduction, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * block.expansion),
                # 注意这里没有 ReLU 层, 因为 shortcut 输出和 identity 相加以后再进行 ReLU 操作
            )
        
        layers = []

        # 最初的self.inplanes == planes == 64
        layers.append(block(self.inplanes, planes, reduction=reduction, stride=stride, downsample=downsample, ))
        self.inplanes = planes * block.expansion  # 如果block为BasicBlock, 那么inplanes=64, 否则inplanes=64*Bottleneck.expansion

        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, reduction=reduction, stride=1))  # 此时 inplanes == planes * block.expansion
        
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

def se_resnet18(num_classes=1000):
    return SEResNet(BasicBlock, [2, 2, 2, 2], reduction=16, num_classes=num_classes)

def se_resnet34(num_classes=1000):
    return SEResNet(BasicBlock, [3, 4, 6, 3], reduction=16, num_classes=num_classes)


def se_resnet50(num_classes=1000):
    return SEResNet(Bottleneck, [3, 4,  6, 3], reduction=16, num_classes=num_classes)

def se_resnet101(num_classes=1000):
    return SEResNet(Bottleneck, [3, 4, 23, 3], reduction=16, num_classes=num_classes)

def se_resnet152(num_classes=1000):
    return SEResNet(Bottleneck, [3, 4, 36, 3], reduction=16, num_classes=num_classes)