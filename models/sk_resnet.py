import torch
import torch.nn as nn
import numpy as np

class SKModule(nn.Module):
    def __init__(self, planes, sk_groups, num_branchs, ratio, min_len, stride=1):
        """SK模块的实现
        
        Arguments:
            nn {[type]} -- [description]
            planes {int} -- 通道数
            sk_groups {SK模块中分组卷积的分组数} -- SK模块中分组卷积的分组数
            num_branchs {int} -- 分支的个数
            ratio {int} -- 比率, 用于计算第一个fc后的最小通道数, 即图中向量Z的长度
            min_len {int} -- 图中向量Z的最短长度
        
        Keyword Arguments:
            stride {int} -- [description] (default: {1})
        """
        super(SKModule, self).__init__()
        vector_len = max(planes // ratio, min_len)
        self.num_branchs = num_branchs
        self.convs = nn.ModuleList()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.num_branchs = num_branchs
        self.vector_len = vector_len
        self.planes = planes
        self.relu = nn.ReLU(inplace=True)

        for i in range(num_branchs):
            self.convs.append(nn.Sequential(
                nn.Conv2d(planes, planes, kernel_size=2*i+3, stride=stride, padding=i+1, groups=sk_groups),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),
            ))
            
        self.fc = nn.Linear(planes, vector_len)
        self.fcs = nn.ModuleList([])

        for i in range(num_branchs):
            self.fcs.append(
                nn.Linear(vector_len, planes)
            )
        self.softmax = nn.Softmax(dim=1)  # 通道维度
    
    def forward(self, x):
        residual = x
        features = []
        for i, conv in enumerate(self.convs):
            features.append(conv(x))
        U = sum(features)
        '''Test successfully
        x1 = torch.ones((2, 3, 5, 5))
        x2 = torch.ones((2, 3, 5, 5))
        x3 = torch.ones((2, 3, 5, 5))

        print(sum([x1, x2, x3]))
        '''

        S = self.avg_pool(U)
        S = torch.flatten(S, 1)  # resize 为 (batch, channel)
        Z = self.fc(S)

        vectors = []
        for i, fc in enumerate(self.fcs):
            vectors.append(fc(Z))
        
        for i in range(len(vectors)):
            if i == 0:
                attention_vectors = vectors[0]
            else:
                attention_vectors = torch.cat([attention_vectors, vectors[i]], dim=1)
        
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)  # resize到(b, c, h, w)

        out = []
        for i in range(self.num_branchs):
            out.append(attention_vectors[:, i: i+self.planes, :, :] * features[i])
        
        out = sum(out)
        out = out + residual
        return self.relu(out)

'''Test
x = np.random.randn(3, 3, 200, 200)
x = torch.tensor(x, dtype=torch.float32)
print(x.shape)

model = SKModule(3, groups=1, num_branchs=2, ratio=1, min_len=32)
y = model(x)

print(y.shape)
'''

class BasicBlock(nn.Module):
    # 基本残差块的输入通道和输出通道是一样的, 所以这里 expansion = 1
    expansion = 1

    def __init__(self, inplanes, planes, sk_groups, num_branchs, ratio, min_len, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.downsample = downsample
        self.sk = SKModule(planes, sk_groups, num_branchs, ratio, min_len)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = self.sk(out)
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

    def __init__(self, inplanes, planes, sk_groups, num_branchs, ratio, min_len, stride=1, downsample=None):
        
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)

        self.downsample = downsample
        self.sk = SKModule(planes * Bottleneck.expansion, sk_groups, num_branchs, ratio, min_len)

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

        out = self.sk(out)
        out += residual

        return self.relu(out)


class SKResNet(nn.Module):
    """带有SK模块的resnet
        
        Arguments:
            nn {[type]} -- [description]
            block {block type} -- basic block or bottleneck
            blocks {list} -- 每一层有多少个block
            groups {int} -- SK模块中分组卷积的分组数
            num_branchs {int} -- 有多少个分支, 每个分支的kernel_size为3, 5, 7...
            ratio {int} -- 用来计算第一个fc层之后特征减少到了多少, 计算图中Z的长度
            min_len {int} -- 第一个fc层的输出的最小维度, 图中向量Z的长度
        
        Keyword Arguments:
            num_classes {int} -- 最后得到的向量长度 (default: {1000})
        """
    def __init__(self, block, blocks, sk_groups, num_branchs, ratio, min_len, num_classes=1000):
        
        super(SKResNet, self).__init__()
        self.inplanes = 64
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer2 = self._make_layer(block,  64, blocks[0], sk_groups, num_branchs, ratio, min_len, stride=1)
        self.layer3 = self._make_layer(block, 128, blocks[1], sk_groups, num_branchs, ratio, min_len, stride=2)
        self.layer4 = self._make_layer(block, 256, blocks[2], sk_groups, num_branchs, ratio, min_len, stride=2)
        self.layer5 = self._make_layer(block, 512, blocks[3], sk_groups, num_branchs, ratio, min_len, stride=2)  # 这里的512是planes数, 最后的得到的通道数应该是 512* block.expansion
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)): 
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, sk_groups, num_branchs, ratio, min_len, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * block.expansion),
                # 注意这里没有 ReLU 层, 因为 shortcut 输出和 identity 相加以后再进行 ReLU 操作
            )
        
        layers = []

        # 最初的self.inplanes == planes == 64
        layers.append(block(self.inplanes, planes, sk_groups, num_branchs, ratio, min_len, stride=stride, downsample=downsample, ))
        self.inplanes = planes * block.expansion  # 如果block为BasicBlock, 那么inplanes=64, 否则inplanes=64*Bottleneck.expansion

        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, sk_groups, num_branchs, ratio, min_len, stride=1))  # 此时 inplanes == planes * block.expansion
        
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

def sk_resnet18():
    return SKResNet(BasicBlock, [2, 2, 2, 2], sk_groups=32, num_branchs=2, ratio=1, min_len=32)

def sk_resnet34():
    return SKResNet(BasicBlock, [3, 4, 6, 3], sk_groups=32, num_branchs=2, ratio=1, min_len=32)


def sk_resnet50():
    return SKResNet(Bottleneck, [3, 4,  6, 3], sk_groups=32, num_branchs=2, ratio=1, min_len=32)

def sk_resnet101():
    return SKResNet(Bottleneck, [3, 4, 23, 3], sk_groups=32, num_branchs=2, ratio=1, min_len=32)

def sk_resnet152():
    return SKResNet(Bottleneck, [3, 4, 36, 3], sk_groups=32, num_branchs=2, ratio=1, min_len=32)

