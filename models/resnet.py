import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    # 基本残差块的输入通道和输出通道是一样的, 所以这里 expansion = 1
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

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

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)

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


class ResNet(nn.Module):
    def __init__(self, block, blocks, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer2 = self._make_layer(block,  64, blocks[0], stride=1)
        self.layer3 = self._make_layer(block, 128, blocks[1], stride=2)
        self.layer4 = self._make_layer(block, 256, blocks[2], stride=2)
        self.layer5 = self._make_layer(block, 512, blocks[3], stride=2)  # 这里的512是planes数, 最后的得到的通道数应该是 512* block.expansion
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
                # 注意这里没有 ReLU 层, 因为 shortcut 输出和 identity 相加以后再进行 ReLU 操作
            )
        
        layers = []

        # 最初的self.inplanes == planes == 64
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion  # 如果block为BasicBlock, 那么inplanes=64, 否则inplanes=64*Bottleneck.expansion

        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, stride=1))  # 此时 inplanes == planes * block.expansion
        
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

def resnet18():
    '''
    每个block有2个卷积, 所以2*(2+2+2+2)=16, 再加上最开始的一个卷积和最后的全连接层, 得到18层
    '''
    return ResNet(BasicBlock, [2, 2, 2, 2])

def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet50():
    '''
    每个block有3个卷积, 所以2*(2+2+2+2)=16, 再加上最开始的一个卷积和最后的全连接层, 得到18层
    '''
    return ResNet(Bottleneck, [3, 4, 6, 3])

def resnet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])

def resnet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])
