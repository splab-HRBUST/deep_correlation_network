import sys
import os
from librosa.core import spectrum
import data_utils
import numpy as np
from torch import Tensor
from torchvision import transforms
import librosa
import torch
from torch import nn
from torch.nn import functional as F
from scipy.interpolate import interp1d
from torchvision.models.resnet import Bottleneck, BasicBlock
from scipy.signal import medfilt
from scipy import signal
from dct_self import dct2,idct2
import torch.optim as optim
from random import sample
import collections
import soundfile as sf
from torch.utils.data import DataLoader, Dataset
from joblib import Parallel, delayed
import h5py
import data_utils


def get_data1(path,num):
    l1 = torch.empty(1,512)
    l2=torch.empty(1,512)
    T=[]
    Q=[]
    num_total = 0.0
    if os.path.exists(path):
        t,q = torch.load(path)
        #print(len(t),len(q))
        #print(t[0].shape)
        feature1=sample(t,num)
        feature2=sample(q,num)
        for f1 in feature1:
            l1=torch.cat((l1,f1),0)
        for f2 in feature2:
            l2=torch.cat((l2,f2),0)
        r1=l1[1:,:] #(256,512)
        r2=l2[1:,:]
        print("load finish")
    else:
        print('load fail')
    return r1,r2

def get_data2(path,num):
    l1 = torch.empty(1,512)
    l2=torch.empty(1,512)
    T=[]
    Q=[]
    num_total = 0.0
    if os.path.exists(path):
        t,q = torch.load(path)
        #print(len(t),len(q))
        #print(t[0].shape)
        feature1=sample(t,num)
        feature2=sample(q,num)
        for f1 in feature1:
            l1=torch.cat((l1,f1),0)
        for f2 in feature2:
            l2=torch.cat((l2,f2),0)
        r1=l1[1:,:] #(32,512)
        r2=l2[1:,:]
        print("load finish")
    else:
        print('load fail')
    return r1,r2

if __name__ == '__main__':

    path='train_first_tensor.pt'
   
    r1,r2=get_data1(path)
    print('r1.grad:{}'.format(r1.requires_grad))
    print('r1.shape:{}'.format(r1.shape))
    print('r2.grad:{}'.format(r2.requires_grad))
    print('r2.shape:{}'.format(r2.shape))
    
