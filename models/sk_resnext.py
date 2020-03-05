import torch
import torch.nn as nn
import numpy as np
import math

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


class SKBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, sk_groups, num_branchs, ratio, min_len, base_width=4, cardinality=32, downsample=None, stride=1):
        super(SKBottleneck, self).__init__()
        D = int(math.floor(planes * (base_width / 64.)) * cardinality)

        self.conv1 = nn.Conv2d(inplanes, D, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(D)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1, bias=False, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(D)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(D, planes * SKBottleneck.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * SKBottleneck.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.se = SKModule(planes * SKBottleneck.expansion, sk_groups, num_branchs, ratio, min_len)


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


class SKResNeXt(nn.Module):
    
    def __init__(self, block, blocks, sk_groups, num_branchs, ratio, min_len, cardinality, base_width, num_classes=1000):
        """SK模块的resnext结构
        
        Arguments:
            block {block type} -- bottleneck
            blocks {list} -- 每一层的block数量
            sk_groups {int} -- sk模块中分组卷积的分组个数
            num_branchs {int} -- sk模块中的分支个数
            ratio {int} -- 用于计算图中向量Z的长度
            min_len {int} -- 向量z的最小长度
            cardinality {int} -- resnext的基数
            base_width {int} -- resnext中每个group中的宽度
        
        Keyword Arguments:
            num_classes {int} -- 最后得到的向量长度 (default: {1000})
        """
        super(SKResNeXt, self).__init__()
        self.inplanes = 64
        self.layer1 = nn.Sequential(
            # 看官网源码有 nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer2 = self._make_layer(block,  64, blocks[0], sk_groups, num_branchs, ratio, min_len, cardinality, base_width, stride=1)
        self.layer3 = self._make_layer(block, 128, blocks[1], sk_groups, num_branchs, ratio, min_len, cardinality, base_width, stride=2)
        self.layer4 = self._make_layer(block, 256, blocks[2], sk_groups, num_branchs, ratio, min_len, cardinality, base_width, stride=2)
        self.layer5 = self._make_layer(block, 512, blocks[3], sk_groups, num_branchs, ratio, min_len, cardinality, base_width, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, num_blocks, sk_groups, num_branchs, ratio, min_len, cardinality, base_width, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, sk_groups, num_branchs, ratio, 
                            min_len, stride=stride, base_width=4, cardinality=32, downsample=downsample))
        self.inplanes = planes * block.expansion

        # 注意这里不再需要 downsample 了
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, sk_groups, num_branchs, 
                                ratio, min_len, stride=1, base_width=4, cardinality=32))
        
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

'''
    这里这么多参数是为了和原文保持一样, 实际上 ratio=1, min_len=32 完全可以用一个 reduction 参数来代替
    self.fc1 = nn.Linear(planes, planes // reduction)
    self.fc2 = nn.Linear(planes // reduction planes)
'''

def sk_resnext50():
    return SKResNeXt(SKBottleneck, [3, 4,  6, 3], sk_groups=32, num_branchs=2, ratio=1, min_len=32, cardinality=32, base_width=4)

def sk_resnext101():
    return SKResNeXt(SKBottleneck, [3, 4, 23, 3], sk_groups=32, num_branchs=2, ratio=1, min_len=32, cardinality=32, base_width=4)

def sk_resnext152():
    return SKResNeXt(SKBottleneck, [3, 4, 36, 3], sk_groups=32, num_branchs=2, ratio=1, min_len=32, cardinality=32, base_width=4)
