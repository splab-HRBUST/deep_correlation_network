import data_utils
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
import librosa
import torch
from torch import nn
import torch.nn as nn
import math
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck, BasicBlock
from gru_am import VGGVox2
from scg_em_model import GatedRes2Net,SEGatedLinearConcatBottle2neck
from scipy.signal import medfilt
from scipy import signal
from dct_self import dct2,idct2
import data_utils
from torch.autograd import Variable
import torch.optim as optim
from pick_data import get_data1,get_data2
from torch.nn.parameter import Parameter
import os





class cqt_mgd(nn.Module):
    def __init__(self, block, layers, num_classes, emb_dim1,emb_dim2,T,Q,E1,E2,
                 zero_init_residual=False):
        super(cqt_mgd, self).__init__()
        self.embedding_size1 = emb_dim1
        self.embedding_size2=emb_dim2
        self.num_classes = num_classes
        self.gru=VGGVox2(BasicBlock, [2, 2, 2, 2],emb_dim=self.embedding_size1)
        self.scg= GatedRes2Net(SEGatedLinearConcatBottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, pretrained=False,loss='softmax')
        self.classifier_layer = nn.Linear(self.embedding_size2, self.num_classes)
        

        self.T=torch.nn.Parameter(torch.transpose(T,dim0=0,dim1=1))
        self.Q=torch.nn.Parameter(torch.transpose(Q,dim0=0,dim1=1))#（512，256）
        self.fai_x=torch.nn.Parameter(torch.ones(512))
        self.fai_y=torch.nn.Parameter(torch.ones(512))

        self.E1=E1
        self.E2=E2
    def forward(self,x,y,device):
        y=self.gru(y) 
        x=self.scg(x)
        x=torch.transpose(x,dim0=0,dim1=1) #(512,32)
        y=torch.transpose(y,dim0=0,dim1=1)#(512,32)
        fai_x,fai_y,T,Q,m,mu_y,E1=self.get_em_param(x,y,device)
        emb=torch.transpose(E1,dim0=0,dim1=1)#(32,256)
        out=self.classifier_layer(emb)
        
        return  x,y,fai_x,fai_y,T,Q,m,mu_y,out,emb
    def get_em_param(self,x,y,device):#x:cqt(512,32) y:mgd(512,32) T:(512,256) Q:(512,256)

        
        N1=x.shape[0]#N=512
        N2 = x.shape[1]#batch:64
        N3=self.T.shape[1]#256
    
        lamda=torch.cat((self.T,self.Q),0)
        fai_z=torch.cat((self.fai_x,self.fai_y),0)#(1024)

        m=torch.sum(x, dim=1)/N2 # (512)
        mu_y=torch.sum(y,dim=1)/N2# (512)
        
        centeredM=x-m.unsqueeze(1)#(512,32)
        variancesM = torch.mean(centeredM ** 2, dim=1)#(512)
        centeredY=y-mu_y.unsqueeze(1)#(512,64)
        variancesY =torch.mean(centeredY**2,dim=1) #(512)
     
        centeredZ=torch.cat((centeredM,centeredY),0)#(1024,32)

        I=torch.eye(256,256)
   
        I=I.to(device)

        # E step 更新E1,E2
       
        B=torch.transpose(lamda,dim0=0,dim1=1)/fai_z#(256,1024)
        
        L=I+B@lamda#(256,256)   (256,1024)@(1024,256)
       
        cov=torch.linalg.pinv(L)#(256,256)
        
        E1=cov@(B@centeredZ)#(256,32) E1为隐变量w
        E2=E1@torch.transpose(E1,dim0=0,dim1=1)+cov*N2#(256,256)
        E1_T=torch.transpose(E1,dim0=0,dim1=1)#(32,256)
        
        # M step 更新T、Q
       
        T=(centeredM@E1_T)@torch.linalg.pinv(E2)#(512,256)
 
        Q=(centeredY@E1_T)@torch.linalg.pinv(E2)#(512,256)

        fai_x=variancesM -torch.mean(T *(T@E2),dim=1) #(512) 
     
        fai_y=variancesY -torch.mean(Q *(Q@E2),dim=1) #(512)
       
        return fai_x,fai_y,T,Q,m,mu_y,E1
    
    
    def loss_em(self,x,y,fai_x,fai_y,T,Q,m,mu_y,device):

        z=torch.cat((x,y),0)
        mu_z=torch.cat((m,mu_y),0)
        fai_z=torch.cat((fai_x,fai_y),0)
        lamda=torch.cat((T,Q),0)
        N=z.shape[1]#32

        
        L_sum1= torch.tensor(0.0)
        L_sum2=torch.tensor(0.0)
       
        z_T=torch.transpose(z,dim0=0,dim1=1)#(32,1024)
        E1_T=torch.transpose(self.E1,dim0=0,dim1=1)

        L_sum1= torch.tensor(0.0)
        L_sum2=torch.tensor(0.0)
        for i in range(z_T.shape[0]):
            zi = z_T[i].unsqueeze(1)#(1024,1)
            Ei = E1_T[i].unsqueeze(1)#(32,1)

            mu=zi-mu_z.unsqueeze(1)#(1024,1)
            mu_T=torch.transpose(mu,dim0=0,dim1=1)#(1,1024)

            cov_eps=torch.ones(fai_z.shape[0])* (1e-12)
            cov_eps=cov_eps.to(device)
            fai_=1/(fai_z+cov_eps)#(1024) fai逆

            L1=-0.5*torch.log(torch.norm(fai_z))-0.5*((mu_T*fai_)@mu)
            
            L2=(mu_T*fai_)@lamda@Ei

            L_sum1=L_sum1+L1
            L_sum2=L_sum2+L2

        l=(torch.transpose(lamda,dim0=0,dim1=1)*fai_)@lamda
            
        L3=torch.sum(self.E2*l)
        
        loss=L_sum1+L_sum2-0.5*L3
        loss=loss/N
       
        return -loss
        
    
    
   
    
def em_param(x,y,T,Q):#x:cqt(512,32) y:mgd(512,32) T:(512,256) Q:(512,256)

    x=torch.transpose(x,dim0=0,dim1=1) #(512,32)
    y=torch.transpose(y,dim0=0,dim1=1)#(512,32)
    T=torch.transpose(T,dim0=0,dim1=1)
    Q=torch.transpose(Q,dim0=0,dim1=1)

    N1=x.shape[0]#N=512
    N2 = x.shape[1]#batch:64
    N3=T.shape[1]#256
    
    lamda=torch.cat((T,Q),0)

    fai_x=torch.ones(512)
    fai_y=torch.ones(512)
    fai_z=torch.cat((fai_x,fai_y),0)#(1024)

    m=torch.mean(x,dim=1) # (512)
    mu_y=torch.mean(y,dim=1)#(512)
    mu_z=torch.cat((m,mu_y),0)

    centeredM=x-m.unsqueeze(1)#(512,64)
    variancesM = torch.mean(centeredM ** 2, dim=1)#(512)
    centeredY=y-mu_y.unsqueeze(1)#(512,64)
    variancesY =torch.mean(centeredY**2,dim=1) #(512)

    centeredZ=torch.cat((centeredM,centeredY),0)#(1024,64)


   
    I=torch.eye(256,256)
    #I=I.to(device)
    # E step 更新E1,E2
    B=torch.transpose(lamda,dim0=0,dim1=1)/fai_z#(256,1024)
   
    L=I+B@lamda#(256,256)   (256,1024)@(1024,256)
   
    cov=torch.linalg.pinv(L)#(256,256)
    E1=cov@(B@centeredZ)#(256,32) E1为隐变量w
    E2=E1@torch.transpose(E1,dim0=0,dim1=1)+cov*N2#(256,256)
    E1_T=torch.transpose(E1,dim0=0,dim1=1)#(32,256)5
    
    # M step 更新T、Q
    
    T=(centeredM@E1_T)@torch.linalg.pinv(E2)#(512,256)
    
    Q=(centeredY@E1_T)@torch.linalg.pinv(E2)#(512,256)
    
    fai_x=variancesM -torch.mean(T *(T@E2),dim=1) #(512) 
     
    fai_y=variancesY -torch.mean(Q *(Q@E2),dim=1) #(512)
       

    return E1,E2

if __name__ == '__main__':
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path='train_first_tensor.pt'

    a,b=get_data1(path)
    x,y=get_data2(path)
    E1,E2=em_param(x,y,a,b)

    batch_size=32
    x1= torch.randn(32, 192, 251)
    y1= torch.randn(32, 192, 251)
    x1=x1.to(device)
    y1=y1.to(device)
    E1=E1.to(device)
    E2=E2.to(device)
    
   
    model = cqt_mgd(BasicBlock, [2, 2, 2, 2],num_classes=2,emb_dim1=512,emb_dim2=256,T=a,Q=b,E1=E1,E2=E2)
    model=model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1 )

    x,y,fai_x,fai_y,T,Q,m,mu_y,E1,E2,out=model(x1,y1,device)
    print('out:',out)
   
    r=model.loss_func(x,y,fai_x,fai_y,T,Q,m,mu_y,device)
    #print('x is leaf',x.is_leaf)
    #print('y is leaf',y.is_leaf)
    #loss,E1=model.loss(x,y)
    print('loss:',r) 

    print("1更新前")
    print('gru:',model.gru.bn1.weight.grad,model.gru.bn1.weight)
    print('resnet',model.resnet.conv1.weight.grad,model.resnet.conv1.weight)
    print('fai_x:',model.fai_x.grad,model.fai_x)
    print('fai_y:',model.fai_y.grad,model.fai_y)
    print('T:',model.T.grad,model.T)
    print('Q:',model.Q.grad,model.Q)
    print('classifier_layer:',model.classifier_layer.weight.grad,model.classifier_layer.weight)
   
   
    optimizer.zero_grad()
 
    r.backward()
    print('梯度')
    print('gru:',model.gru.bn1.weight.grad)
    print('resnet',model.resnet.conv1.weight.grad)
    print('fai_x:',model.fai_x.grad)
    print('fai_y:',model.fai_y.grad)
    print('T:',model.T.grad)
    print('Q:',model.Q.grad)
    
    print('classifier_layer:',model.classifier_layer.weight.grad)
    optimizer.step()
    print('1更新后的数值')
    print('gru:',model.gru.bn1.weight)
    print('resnet',model.resnet.conv1.weight)
    print('fai_x:',model.fai_x)
    print('fai_y:',model.fai_y)
    print('T:',model.T)
    print('Q:',model.Q)
   
    print('classifier_layer:',model.classifier_layer.weight)

    
        

   
   
   
    
    



