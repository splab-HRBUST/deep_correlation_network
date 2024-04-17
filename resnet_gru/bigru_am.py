import torch.nn as nn
import math
import torch
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck, BasicBlock
from Bigru_Attention4 import Bigru_Attention

# -*- coding: utf-8 -*-
class ReLU(nn.Hardtanh):

    def __init__(self, inplace=False):
        super(ReLU, self).__init__(0, 20, inplace)

    def __repr__(self):
        inplace_str, = 'inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' \
            + inplace_str + ')'


class VGGVox2(nn.Module):

    def __init__(self, block, layers, num_classes, emb_dim,
                 zero_init_residual=False):
        super(VGGVox2, self).__init__()
        self.embedding_size = emb_dim
        self.num_classes = num_classes
        self.num_layers=2
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding_layer = nn.Linear(512 * block.expansion, self.embedding_size)
        self.model_gru = Bigru_Attention(input_size=self.embedding_size,input_embeding=self.embedding_size)               

        self.classifier_layer = nn.Linear(
            self.embedding_size, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        nn.init.xavier_uniform_(self.classifier_layer.weight)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with z
        # eros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to 
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward_once(self, x):
        
        x = x.unsqueeze(dim=1)#[32, 1, 192, 251]
       
        out = self.conv1(x)#([32, 64, 96, 126])
      
        out = self.bn1(out)#([32, 64, 96, 126])
      
        out = self.relu(out)#([32, 64, 96, 126])
       
        out = self.maxpool(out)#([32, 64, 48, 63])
      

        out = self.layer1(out)#([32, 64, 48, 63])
        
        out = self.layer2(out)#([32, 128, 24, 32])
       
        out = self.layer3(out)#([32, 256, 12, 16])
      
        out = self.layer4(out)#([32, 512, 6, 8])
        
        out = self.avgpool(out)#([32, 512, 1, 1])
      
        out = out.view(out.size(0),-1)#(32,512)
        
        #out = self.embedding_layer(out)#(32,512)
      
        out= out.unsqueeze(dim=1)  #(32,1,512)
       
        out=self.model_gru(out)#(32,512)
      
        return out
    
    def forward(self, x, phase):
        if phase == 'evaluation':
            _padding_width = x[0, 0, 0, -1]
            out = x[:, :, :, :-1-int(_padding_width.item())]
            out = self.forward_once(out)
            # out = F.normalize(out, p=2, dim=1)

        elif phase == 'triplet':
            out = self.forward_once(x)
            out = F.normalize(out, p=2, dim=1)

        elif phase == 'pretrain':
            emb = self.forward_once(x)
            # Multiply by alpha as suggested in
            # https://arxiv.org/pdf/1703.09507.pdf (L2-SoftMax)
            # out = F.normalize(out, p=2, dim=1)
            # out = out * self.alpha
            out = self.classifier_layer(emb)#(32,2)
            return out,emb

      
           
        else:
            raise ValueError('phase wrong!')
        

class VGGVox1(nn.Module):

    def __init__(self, num_classes=1211, emb_dim=1024):
        super(VGGVox1, self).__init__()
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96,
                      kernel_size=7, stride=2, padding=1),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256,
                      kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(5, 3), stride=(3, 2))
        )
        self.fc6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=4096, kernel_size=(4, 1)),
            nn.BatchNorm2d(num_features=4096),
            nn.ReLU(inplace=True)
        )
        self.fc7 = nn.Linear(in_features=4096, out_features=self.emb_dim)

        nn.init.xavier_uniform_(self.fc7.weight)

    def forward_once(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.fc6(out)
        # global average pooling layer
        _, _, _, width = out.size()
        self.apool6 = nn.AvgPool2d(kernel_size=(1, width))
        out = self.apool6(out)
        out = out.view(out.size(0), -1)
        out = self.fc7(out)
        return out

    def forward(self, x, phase):
        if phase == 'evaluation':
            _padding_width = x[0, 0, 0, -1]
            out = x[:, :, :, :-1 - int(_padding_width.item())]
            out = self.forward_once(out)
            # out = F.normalize(out, p=2, dim=1)

        elif phase == 'triplet':
            out = self.forward_once(x)
            out = F.normalize(out, p=2, dim=1)

        elif phase == 'pretrain':
            out = self.forward_once(x)
            out = self.fc8(out)
        else:
            raise ValueError('phase wrong!')
        return out


class OnlineTripletLoss(nn.Module):
    """
Online Triplets loss
Takes a batch of embeddings and corresponding labels.
Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
triplets

Reference: https://github.com/adambielski/siamese-triplet
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):
        triplets = self.triplet_selector.get_triplets(
            embeddings.detach(), target)

        if embeddings.is_cuda:
            triplets = triplets.to(c.device)

        # l2 distance三元组损失
        ap_distances = (embeddings[triplets[:, 0]] -
                        embeddings[triplets[:, 1]]).pow(2).sum(1)
        an_distances = (embeddings[triplets[:, 0]] -
                        embeddings[triplets[:, 2]]).pow(2).sum(1)
        # ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1).pow(.5)
        # an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1).pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        # # cosine similarity互信息
        # cos = torch.nn.CosineSimilarity(dim=1)
        # ap_similarity = cos(embeddings[triplets[:, 0]], embeddings[triplets[:, 1]])
        # an_similarity = cos(embeddings[triplets[:, 0]], embeddings[triplets[:, 2]])
        # losses = F.relu(an_similarity - ap_similarity + self.margin)

        return losses.mean(), len(triplets), ap_distances.mean(), an_distances.mean()


class TripletLoss(nn.Module):
    """
Triplet loss
Takes embeddings of an anchor sample, a posi7tive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


if __name__ == '__main__':
    net = VGGVox2(BasicBlock, [2, 2, 2, 2],num_classes=2,emb_dim=512)
    inputdata = torch.randn(32, 192, 251)
    result = net(inputdata, 'pretrain')
    #print('result:{}'.format(result.shape))