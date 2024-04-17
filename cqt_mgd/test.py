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
from resnet import resnet
from scipy.signal import medfilt
from scipy import signal
from dct_self import dct2,idct2
import data_utils
from torch.autograd import Variable
import torch.optim as optim
from pick_data import get_data1,get_data2
from torch.nn.parameter import Parameter
import os

#test程序说明，按照dagmm实验思路进行修改
#初始T、Q初始化仍用本网络，而不从外部选取




class cqt_mgd(nn.Module):
    def __init__(self, block, layers, num_classes, emb_dim1,emb_dim2,
                 zero_init_residual=False):
        super(cqt_mgd, self).__init__()
        self.embedding_size1 = emb_dim1
        self.embedding_size2=emb_dim2
        self.num_classes = num_classes
        self.gru=VGGVox2(BasicBlock, [2, 2, 2, 2],emb_dim=self.embedding_size1)
        self.resnet=resnet(BasicBlock, [2, 2, 2, 2],emb_dim=self.embedding_size1)#（32，512）
        self.classifier_layer = nn.Linear(self.embedding_size2, self.num_classes)
        

        # self.T=torch.nn.Parameter(torch.transpose(T,dim0=0,dim1=1))
        # self.Q=torch.nn.Parameter(torch.transpose(Q,dim0=0,dim1=1))#（512，256）
        self.fai_x=torch.nn.Parameter(torch.ones(512,1))#(512,1)
        self.fai_y=torch.nn.Parameter(torch.ones(512,1))#(512,1)

       
    def forward(self,x,y,device):
        x=self.gru(x) 
        y=self.resnet(y)
        x=torch.transpose(x,dim0=0,dim1=1) #(512,32)
        y=torch.transpose(y,dim0=0,dim1=1)#(512,32)

        #z,mu_z,fai_z,lamda,E1,E2=self.get_em_param(x,y,device)
        z,mu_z,fai_z,E1=self.get_em_param(x,y,device)
        out=torch.transpose(E1,dim0=0,dim1=1)#(32,32)
        out=self.classifier_layer(out)
        
        return  z,mu_z,fai_z,out
    def get_em_param(self,x,y,device):#x:cqt(512,32) y:mgd(512,32) T:(512,32) Q:(512,32)

        
        N1=x.shape[0]#N=512
        N2 = x.shape[1]#batch:64

        m=torch.mean(x,dim=1) # (512)
        mu_y=torch.mean(y,dim=1)#(512)
        centeredM=x-m.unsqueeze(1)#(512,64)
        variancesM = torch.mean(centeredM ** 2, dim=1)#(512)
        centeredY=y-mu_y.unsqueeze(1)#(512,64)
        variancesY = torch.mean(centeredY ** 2, dim=1)
     
        I=torch.eye(64,64)
        I=I.to(device)

        # E step 更新E1,E2
       
        B=torch.transpose(x,dim0=0,dim1=1)/torch.transpose(self.fai_x,dim0=0,dim1=1)#(64,512)
        C=torch.transpose(y,dim0=0,dim1=1)/torch.transpose(self.fai_y,dim0=0,dim1=1)#(64,512)
        
        L=I+B@x+C@y #(64,64)
       
        # cov_eps = torch.eye(L.shape[1]) * (1e-12)
        # cov_eps = cov_eps.to(device)

        cov=torch.linalg.pinv(L)#(64,64)

        E1=cov@(B@centeredM+C@centeredY)#(64,64) E1为隐变量w
        
        E2=E1@torch.transpose(E1,dim0=0,dim1=1)+cov*N2#(64,64)
       
        E1_T=torch.transpose(E1,dim0=0,dim1=1)#(64,64)

        # if torch.isnan(B).any():
        #     print('B is nan')
        # if torch.isnan(C).any():
        #     print('C is nan')
        # if torch.isnan(L).any():
        #     print('B is nan')
        # if torch.isnan(cov).any():
        #     print('cov is nan')
        # if torch.isnan(E1).any():
        #     print('E1 is nan')
        # if torch.isnan(E2).any():
        #     print('E2 is nan')
            
        # if torch.isinf(B).any():
        #      print('B is inf')
        # if torch.isinf(C).any():
        #      print('C is inf')
        # if torch.isinf(L).any():
        #      print('L is inf')
        # if torch.isinf(cov).any():
        #      print('cov is inf')
        # if torch.isinf(E1).any():
        #      print('E1 is inf')
        # if torch.isinf(E2).any():
        #      print('E2 is inf')
        # M step 更新T、Q

        T=(centeredM@E1_T)@torch.linalg.pinv(E2)#(512,64)
 
        Q=(centeredY@E1_T)@torch.linalg.pinv(E2)#(512,64)
       
        fai_x=variancesM -torch.mean(T *(T@E2),dim=1) #(512) 
     
        fai_y=variancesY -torch.mean(Q *(Q@E2),dim=1) #(512)

        # if torch.isnan(T).any():
        #     print('T is nan')
        # if torch.isnan(Q).any():
        #     print('Q is nan')
        # if torch.isnan(fai_x).any():
        #     print('fai_x is nan')
        # if torch.isnan(fai_y).any():
        #     print('fai_y is nan')
            
        # if torch.isinf(T).any():
        #      print('T is inf')
        # if torch.isinf(Q).any():
        #      print('Q is inf')
        # if torch.isinf(fai_x).any():
        #      print('fai_x is inf')
        # if torch.isinf(fai_y).any():
        #      print('fai_y is inf')
       

        z=torch.cat((x,y),0)#(1024,32)
        mu_z=torch.cat((m,mu_y),0)#（1024）
        fai_z=torch.cat((fai_x,fai_y),0)#(1024)
        lamda=torch.cat((T,Q),0)#(1024,32)
      
        
        return z,mu_z,fai_z,E1
    
    def loss_em1(self,z,mu_z,fai_z):
        N=z.shape[1]
        z_T=torch.transpose(z,dim0=0,dim1=1)#(32,1024)
        L_sum= torch.tensor(0.0)
        for i in range(z_T.shape[0]):
            zi = z_T[i].unsqueeze(1)#(1024,1)
            mu=zi-mu_z.unsqueeze(1)#(1024,1)
            mu_T=torch.transpose(mu,dim0=0,dim1=1)#(1,1024)
           
            fai_=1/fai_z#(1024)
          
            e_k=-0.5*mu_T*fai_@mu
           
            e_k = e_k / torch.log(torch.sqrt(2 * math.pi*torch.norm(fai_z)))
          
            L_sum=L_sum+e_k
        loss=-(L_sum/N)
        
        return loss

    def loss_em2(self,z,mu_z,fai_z,lamda,E1,E2,device):
        N=z.shape[1]
        z_T=torch.transpose(z,dim0=0,dim1=1)#(32,1024)
        E1_T=torch.transpose(E1,dim0=0,dim1=1)#(32,32)

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

        l=(torch.transpose(lamda,dim0=0,dim1=1)*fai_)@lamda#(32,32)
            
        L3=torch.sum(E2*l)#(256,256)
        
        loss=L_sum1+L_sum2-0.5*L3
        #loss=-(loss/N)
        
        return loss


       

    
    
    
   
    


if __name__ == '__main__':
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    

    batch_size=64
    x1= torch.randn(64, 192, 251)
    y1= torch.randn(64, 192, 251)
    x1=x1.to(device)
    y1=y1.to(device)
    
    
   
    model = cqt_mgd(BasicBlock, [2, 2, 2, 2],num_classes=2,emb_dim1=512,emb_dim2=64)
    model=model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    #optimizer = optim.SGD(model.parameters(), lr=0.1 )

    z,mu_z,fai_z,out=model(x1,y1,device)
    #print('out:',out)
   
    #r=model.loss_em1(z,mu_z,fai_z)
    loss1=model.loss_em1(z,mu_z,fai_z)
    
    print('loss:',loss1) 

    print("1更新前")
    print('gru:',model.gru.bn1.weight.grad,model.gru.bn1.weight)
    print('resnet',model.resnet.conv1.weight.grad,model.resnet.conv1.weight)
    # print('fai_x:',model.fai_x.grad,model.fai_x)
    # print('fai_y:',model.fai_y.grad,model.fai_y)
    # print('T:',model.T.grad,model.T)
    # print('Q:',model.Q.grad,model.Q)
    print('classifier_layer:',model.classifier_layer.weight.grad,model.classifier_layer.weight)
   
   
    optimizer.zero_grad()
 
    loss1.backward()
    print('梯度')
    print('gru:',model.gru.bn1.weight.grad)
    print('resnet',model.resnet.conv1.weight.grad)
    print('fai_x:',model.fai_x.grad)
    print('fai_y:',model.fai_y.grad)
    # print('T:',model.T.grad)
    # print('Q:',model.Q.grad)
    
    print('classifier_layer:',model.classifier_layer.weight.grad)
    optimizer.step()
    print('1更新后的数值')
    print('gru:',model.gru.bn1.weight)
    print('resnet',model.resnet.conv1.weight)
    print('fai_x:',model.fai_x)
    print('fai_y:',model.fai_y)
    # print('T:',model.T)
    # print('Q:',model.Q)
   
    print('classifier_layer:',model.classifier_layer.weight)

    
        

   
   
   
    
    



