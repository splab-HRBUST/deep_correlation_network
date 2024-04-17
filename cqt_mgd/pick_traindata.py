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
from gru_am import VGGVox2
from resnet import resnet
from fix_model import cqt_mgd
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#torch.backends.cudnn.enabled = False

embedding_dim1 = 512   # you can change it to 256 
embedding_dim2=256
batch_size = 32 
num_workers = 8
n_classes = 2
n_samples = 10
l1=0.1
l2=0.005
pretrain_lr_init = 0.05
pretrain_lr_last = 0.0001
pretrain_epoch_num = 100

WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9

def cqtgram(y,sr = 16000, hop_length=256, octave_bins=24, 
n_octaves=8, fmin=21, perceptual_weighting=False):
#y:音频时间序列 sr采样率 hop_length:帧移  octave：八度音阶 fmin:最低频率 perceptual_weigh感知权重

    s_complex = librosa.cqt(
        y,
        sr=sr,
        hop_length=hop_length,
        bins_per_octave=octave_bins,
        n_bins=octave_bins * n_octaves,
        fmin=fmin,
    )
    specgram = np.abs(s_complex)
    # if代码块可以不要。
    if perceptual_weighting:
       # 功率谱的感知权重：S_p[f] = frequency_weighting(f, 'A') + 10*log(S[f] / ref);
        freqs = librosa.cqt_frequencies(specgram.shape[0], fmin=fmin, bins_per_octave=octave_bins)#返回每一个cqt频率带的中心频率。
        specgram = librosa.perceptual_weighting(specgram ** 2, freqs, ref=np.max)#功率谱的感知加权。
    else:
        specgram = librosa.amplitude_to_db(specgram, ref=np.max)#将振幅谱转为用分贝表示的谱图。
    return specgram
# ======================================================================================
def mgdgram(y,sr = 16000, hop_length=256, octave_bins=24, 
n_octaves=8, fmin=21, perceptual_weighting=False):
    rho=0.4
    gamma=0.9
    n_xn = y*range(1,len(y)+1)
    X = librosa.cqt(
        y,
        sr=sr,
        hop_length=hop_length,
        bins_per_octave=octave_bins,
        n_bins=octave_bins * n_octaves,
        fmin=fmin,
        window = "hamming"
    )
    Y = librosa.cqt(
        n_xn,
        sr=sr,
        hop_length=hop_length,
        bins_per_octave=octave_bins,
        n_bins=octave_bins * n_octaves,
        fmin=fmin,
        window = "hamming"
    )
    Xr, Xi = np.real(X), np.imag(X)
    Yr, Yi = np.real(Y), np.imag(Y)
    magnitude,_ = librosa.magphase(X,1)# magnitude:幅度，_:相位
    S = np.square(np.abs(magnitude)) # powerspectrum, S =   (192, 126)
    """
    medifilt中值滤波：
    中值滤波的基本原理是把数字图像或数字序列中一点的值用该点的一个邻域中各点值的中值代替，
    让周围的像素值接近真实值，从而消除孤立的噪声点
    """
    a = medfilt(S, 5) #a.shape =  (192, 251)
    dct_spec = dct2(a) # dct_spec.shape =  (192, 251)
    smooth_spec = np.abs(idct2(dct_spec[:,:291]))# smooth_spec.shape =  (192, 251)
    # smooth_spec = np.abs(a)
    gd = (Xr*Yr + Xi*Yi)/np.power(smooth_spec+1e-05,rho)#对振幅的每个值都进行0.4次方处理。
    mgd = gd/(np.abs(gd)*np.power(np.abs(gd),gamma)+1e-10)
    mgd = mgd/np.max(mgd)
    cep = np.log2(np.abs(mgd)+1e-08)
    return cep
# ======================================================================================
def pad(x, max_len=64000):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = (max_len / x_len)+1
    x_repeat = np.repeat(x, num_repeats)
    padded_x = x_repeat[:max_len]
    return padded_x

        
def get_data(path,model,dataset):
    l1 = torch.empty(1,512)
    l2=torch.empty(1,512)
    T=[]
    Q=[]
    num_total = 0.0
    if os.path.exists(path):
        
        t,q = torch.load(path)
        #print(len(t),len(q))
        #print(t[0].shape)
        feature1=sample(t,256)
        feature2=sample(q,256)
        for f1 in feature1:
            l1=torch.cat((l1,f1),0)
        for f2 in feature2:
            l2=torch.cat((l2,f2),0)
        r1=l1[1:,:] #(256,512)
        r2=l2[1:,:]
        print("load finish")
    else:
        data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        with torch.no_grad():
            for data in data_loader:
                
                data1,data2=data[0],data[1]
                batch_x1,batch_y1,batch_meta1=data1
                batch_x2,batch_y2,batch_meta2=data2
                '''
                batch_meta =  ASVFile(speaker_id, file_name, path, sys_id, key)
                '''
                batch_size = batch_x1.size(0)
                num_total += batch_size

                
                batch_x1 = batch_x1.to(device)
                batch_y1 = batch_y1.view(-1).type(torch.int64).to(device)
                batch_x2 = batch_x2.to(device)
                a,b,out= model(batch_x1,batch_x2)#batch_out(32,2)a(32,512)b(32,512)
                #print(a.shape)
                #print(b.shape)
                a,b=a.cpu(),b.cpu()
                for t in a:
                    t=t.unsqueeze(dim=0)#(1,512)
                    T.append(t)
                for q in b:
                    q=q.unsqueeze(dim=0)
                    Q.append(q)
                # print('batch')
            
        print('1')
        torch.save((T,Q),path)
        print('2')
        feature1=sample(T,256)
        feature2=sample(Q,256)
        for f1 in feature1:
            l1=torch.cat((l1,f1),0)
        for f2 in feature2:
            l2=torch.cat((l2,f2),0)
        r1=l1[1:,:] #(256,512)
        r2=l2[1:,:]
        print("load finish")
        #print('load fail')
    return r1,r2

if __name__ == '__main__':

    transforms1 = transforms.Compose([
            lambda x: pad(x),
            lambda x: cqtgram(x),
            lambda x: Tensor(x)
    ])

    transforms2= transforms.Compose([
        lambda x: pad(x),
        lambda x: mgdgram(x),
        lambda x: Tensor(x)
    ])
    
    is_logical = 'logical'
    features1='spect'
    features2='mgd'
    train_set1 = data_utils.ASVDataset(is_train=True, is_logical=is_logical, transform=transforms1,
                                      feature_name=features1)
    train_set2 = data_utils.ASVDataset(is_train=True, is_logical=is_logical, transform=transforms2,
                                      feature_name=features2)
    myTrainset = data_utils.MyDataset(dataset1=train_set1, dataset2=train_set2)
    model_path='models/fix3_epoch_97.pth'

    net = cqt_mgd(BasicBlock, [2, 2, 2, 2],num_classes=n_classes,emb_dim1=embedding_dim1,emb_dim2=embedding_dim2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model=net.to(device)
    #model = nn.DataParallel(model, device_ids=[0,1]  )# Multiple GPUs
    #checkpoint=torch.load(model_path,map_location={'cuda:0':'cuda:1'})
    checkpoint=torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    print('Model loaded : {}'.format(model_path))
    
    path='train_last_tensor.pt'
   
    r1,r2=get_data(path,model,myTrainset)
    print('r1.grad:{}'.format(r1.requires_grad))
    print('r2.grad:{}'.format(r2.requires_grad))
    
