from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, xavier_normal_
import math
from torch.nn import Parameter
class OCSoftmaxWithLoss(nn.Module):
    """
    OCSoftmaxWithLoss()

    """

    def __init__(self):
        super(OCSoftmaxWithLoss, self).__init__()
        self.m_loss = nn.Softplus()

    def forward(self, inputs, target):
        """
        input:
        ------
          input: tuple of tensors ((batchsie, out_dim), (batchsie, out_dim))
                 output from OCAngle
                 inputs[0]: positive class score
                 inputs[1]: negative class score
          target: tensor (batchsize)
                 tensor of target index
        output:
        ------
          loss: scalar
        """
        # Assume target is binary, positive (genuine) = 0, negative (spoofed) = 1
        #
        # Equivalent to select the scores using if-elese
        # if target = 1, use inputs[1]
        # else, use inputs[0]
        output = inputs[1] * target.view(-1, 1) + \
                 inputs[0] * (1 - target.view(-1, 1))
        loss = self.m_loss(output).mean()
        return loss
class OCAngleLayer(nn.Module):
    """ Output layer to produce activation for one-class softmax

    Usage example:
     batchsize = 64
     input_dim = 10
     class_num = 2

     l_layer = OCAngleLayer(input_dim)
     l_loss = OCSoftmaxWithLoss()

     data = torch.rand(batchsize, input_dim, requires_grad=True)
     target = (torch.rand(batchsize) * class_num).clamp(0, class_num-1)
     target = target.to(torch.long)

     scores = l_layer(data)
     loss = l_loss(scores, target)

     loss.backward()
    """

    def __init__(self, in_planes, w_posi=0.9, w_nega=0.2, alpha=20.0):
        super(OCAngleLayer, self).__init__()
        self.in_planes = in_planes
        self.w_posi = w_posi
        self.w_nega = w_nega
        self.out_planes = 1

        self.weight = Parameter(torch.Tensor(in_planes, self.out_planes))
        # self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        nn.init.kaiming_uniform_(self.weight, 0.25)
        self.weight.data.renorm_(2, 1, 1e-5).mul_(1e5)

        self.alpha = alpha

    def forward(self, input, flag_angle_only=False):
        """
        Compute oc-softmax activations

        input:
        ------
          input tensor (batchsize, input_dim)

        output:
        -------
          tuple of tensor ((batchsize, output_dim), (batchsize, output_dim))
        """
        w = self.weight.renorm(2, 1, 1e-5).mul(1e5)
        x_modulus = input.pow(2).sum(1).pow(0.5)
        inner_wx = input.mm(w)
        cos_theta = inner_wx / x_modulus.view(-1, 1)
        cos_theta = cos_theta.clamp(-1, 1)
        if flag_angle_only:
            pos_score = cos_theta
            neg_score = cos_theta
        else:
            pos_score = self.alpha * (self.w_posi - cos_theta)
            neg_score = -1 * self.alpha * (self.w_nega - cos_theta)
        out = torch.cat([neg_score, pos_score], dim=1)
        # print('oc:',out.size())
        return out
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        # print('se reduction: ', reduction)
        # print(channel // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # F_squeeze
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
                nn.Sigmoid()
        )

    def forward(self, x):  # x: B*C*D*T
        b, c, _, _ = x.size()
        #print("x = ",x.size())
        y = self.avg_pool(x).view(b, c)
        #print("avg : = ",y.size())
        y = self.fc(y).view(b, c, 1, 1)
        #print("y : = ",y.size())
        return x * y.expand_as(x)
class LinearConcatGate(nn.Module):
    def __init__(self, indim, outdim):
        super(LinearConcatGate, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(indim, outdim, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_prev, x):
        x_cat = torch.cat([x_prev, x], dim=1)
        b, c_double, _, _ = x_cat.size()
        c = int(c_double / 2)
        y = self.avg_pool(x_cat).view(b, c_double)
        y = self.sigmoid(self.linear(y)).view(b, c, 1, 1)
        return x_prev * y.expand_as(x_prev)








class SEGatedLinearConcatBottle2neck(nn.Module):
    expansion = 2

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 baseWidth=26,
                 scale=4,
                 stype='normal',
                 gate_reduction=4):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(SEGatedLinearConcatBottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes,
                               width * scale,
                               kernel_size=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(
                nn.Conv2d(width,
                          width,
                          kernel_size=3,
                          stride=stride,
                          padding=1,
                          bias=False))
            bns.append(nn.BatchNorm2d(width))

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        if stype != 'stage':
            gates = []
            for i in range(self.nums - 1):
                gates.append(LinearConcatGate(2 * width, width))
            self.gates = nn.ModuleList(gates)

        self.conv3 = nn.Conv2d(width * scale,
                               planes * self.expansion,
                               kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.se = SELayer(planes * self.expansion, reduction=16)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x
        # print('x: ', x.size())
        out = self.conv1(x)
        # print('conv1: ', out.size())
        out = self.bn1(out)
        out = self.relu(out)
        # print("out = ",out.shape)
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = gate_sp + spx[i]
            sp = self.convs[i](sp)
            bn_sp = self.bns[i](sp)
            if self.stype != 'stage' and i < self.nums - 1:
                gate_sp = self.gates[i](sp, spx[i + 1])
            sp = self.relu(bn_sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        # print('conv2: ', out.size())
        # print(self.width)
        # print(self.scale)
        out = self.conv3(out)
        # print('conv3: ', out.size())
        out = self.bn3(out)
        # print('bn3: ', out.size())
        out = self.se(out)
        # print('se :', out.size())

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
class GatedRes2Net(nn.Module):
    def __init__(self, block, layers, baseWidth=26, scale=4, m=0.35, num_classes=512, loss='softmax', gate_reduction=4,
                 **kwargs):
        self.inplanes = 16
        super(GatedRes2Net, self).__init__()
        self.loss = loss
        self.baseWidth = baseWidth
        self.scale = scale
        self.gate_reduction = gate_reduction
        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                                   nn.Conv2d(16, 16, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                                   nn.Conv2d(16, 16, 3, 1, 1, bias=False))
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, layers[0])  # 64
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)  # 128
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)  # 256
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)  # 512
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier_layer = nn.Linear(256, 512)
        
        # self.stats_pooling = StatsPooling()

        if self.loss == 'softmax':
            # self.cls_layer = nn.Linear(2*8*128*block.expansion, num_classes)
            self.cls_layer = nn.Sequential(nn.Linear(128 * block.expansion, num_classes), nn.LogSoftmax(dim=-1))
            self.loss_F = nn.NLLLoss()
        elif self.loss == 'oc-softmax':
            self.cls_layer = OCAngleLayer(128 * block.expansion, w_posi=0.9, w_nega=0.2, alpha=20.0)
            self.loss_F = OCSoftmaxWithLoss()
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride,
                             stride=stride,
                             ceil_mode=True,
                             count_include_pad=False),
                nn.Conv2d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=1,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes,
                  planes,
                  stride,
                  downsample=downsample,
                  stype='stage',
                  baseWidth=self.baseWidth,
                  scale=self.scale,
                  gate_reduction=self.gate_reduction))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      baseWidth=self.baseWidth,
                      scale=self.scale,
                      gate_reduction=self.gate_reduction))

        return nn.Sequential(*layers)

    def _forward(self, x):
        # print('enter forward')
        # print("x.size() = ",x.size())
        # x = x[:, None, ...]
        x = x.unsqueeze(dim=1)
        x = self.conv1(x)
        # print('conv1: ', x.size())
        x = self.bn1(x)
        # print('bn1: ', x.size())
        x = self.relu(x)
        # print('relu: ', x.size())
      

        x = self.layer1(x)
        # print('layer1: ', x.size())
        x = self.layer2(x)
        # print('layer2: ', x.size())
        x = self.layer3(x)
        # print('layer3: ', x.size())
        x = self.layer4(x)
        # print('layer4: ', x.size())
        # x = self.stats_pooling(x)
        x = self.avgpool(x)
        # print('avgpool:', x.size())
        # x = x.view(x.size(0), -1)
        x = torch.flatten(x, 1)
        # print('flatten: ', x.size())
        
        x = self.cls_layer(x)#(32,256)
        # print('fenlei: ', x.size())
        # x=self.classifier_layer(x)#(32,512)
        # print('cls_layser: ', x.size())
        return x
        # return F.log_softmax(x, dim=-1)

    def extract(self, x):
        # x = x[:, None, ...]
        # print('enter extract')

        # print("x.size() = ",x.size())
        x = self.conv1(x)
        # print('conv1: ', x.size())
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        # print('layer1: ', x.size())
        x = self.layer2(x)
        # print('layer2: ', x.size())
        x = self.layer3(x)
        # print('layer3: ', x.size())
        x = self.layer4(x)
        # print('layer4: ', x.size())

        x = self.avgpool(x)
        # print('avgpool: ', x.size())
        x = torch.flatten(x, 1)
        # print('flatten: ', x.size())
        return x

    # Allow for accessing forward method in a inherited class
    forward = _forward


# =========================DY-ReluB=============================================


if __name__ == '__main__':
   
    def se_gated_linearconcat_res2net50_v1b(**kwargs):
        model = GatedRes2Net(SEGatedLinearConcatBottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
        return model

    model = se_gated_linearconcat_res2net50_v1b(pretrained=False,loss='softmax')#embedding 512
    input = torch.randn(64,192,251)
    out= model(input) 
  