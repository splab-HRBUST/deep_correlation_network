"""
    Author: Moustafa Alzantot (malzantot@ucla.edu)
    All rights reserved.
"""
import argparse
import sys
import os
from librosa.core import spectrum
import data_utils
import data_utils2
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
import librosa
import torch
from torch import nn
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from scipy.interpolate import interp1d
from eval_metrics import compute_eer
from torchvision.models.resnet import Bottleneck, BasicBlock
from scipy.signal import medfilt
from scipy import signal
from dct_self import dct2,idct2
from test4 import cqt_mgd,em_param
from pick_data import get_data1,get_data2
import torch.optim as optim
from itertools import cycle
from random import sample
import random
import pandas as pd
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
#合起来计算loss 不添加正则项 且损失函数中的E1，E2固定不更新
#bigru+resnet
#改变学习率,按照宋修改
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
embedding_dim1 = 512   # you can change it to 256 
embedding_dim2=256
batch_size = 32 
num_workers = 8
n_classes = 2
n_samples = 10


tensor_path='train_first_tensor.pt'
a,b=get_data1(tensor_path,256)
x,y=get_data2(tensor_path,64)
E1,E2=em_param(x,y,a,b)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#--------------------------------------------------------
E1=E1.to(device)
E2=E2.to(device)
net = cqt_mgd(BasicBlock, [2, 2, 2, 2],num_classes=n_classes,emb_dim1=embedding_dim1,emb_dim2=embedding_dim2,T=a,Q=b,E1=E1,E2=E2)




#----------------------------------------------------------------------------------------------

target_scores = np.array([])
nontarget_scores = np.array([])
def cat_scores(batch_score,batch_y,len):
    global target_scores
    global nontarget_scores
    for j in range(len):
        if(batch_y[j]==1):
            target_scores=np.append(target_scores,batch_score[j])
        else:
            nontarget_scores=np.append(nontarget_scores,batch_score[j])

# ======================================计算cqtgram============================================
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
# ======================================================================================


def evaluate_accuracy(data_loader, model, device):
    num_correct = 0.0
    num_total = 0.0
    with torch.no_grad():
        model.eval()
        for data in data_loader:
            data1,data2=data[0],data[1]
            batch_x1,batch_y1,batch_meta1=data1
            batch_x2,batch_y2,batch_meta2=data2
           
            batch_size = batch_x1.size(0)
            num_total += batch_size
            batch_x1 = batch_x1.to(device)
            batch_y1 = batch_y1.view(-1).type(torch.int64).to(device)
            batch_x2 = batch_x2.to(device)

            x,y,fai_x,fai_y,T,Q,m,mu_y,batch_out=model(batch_x1,batch_x2,device)
           
            _, batch_pred = batch_out.max(dim=1)
            num_correct += (batch_pred == batch_y1).sum(dim=0).item()
    return 100 * (num_correct / num_total)

def produce_evaluation_file(dataset1,dataset2, model, device, save_path):
    data_loader = DataLoader(dataset1, batch_size=64, shuffle=True)
   
    num_total = 0.0
   
    fname_list = []
    key_list = []
    sys_id_list = []
    key_list = []
    score_list = []
    arr = np.zeros((1,256))
    with torch.no_grad():
        model.eval()
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
            
            x,y,fai_x,fai_y,T,Q,m,mu_y,batch_out,emb=model(batch_x1,batch_x2,device)
            
            batch_score = (batch_out[:, 1] - batch_out[:, 0]
                        ).data.cpu().numpy().ravel()
            
            emb=emb.cpu().numpy()
            arr=np.concatenate((arr,emb),axis = 0)
          
            # add outputs
            fname_list.extend(list(batch_meta1[1]))
            key_list.extend(
                ['bonafide' if key == 1 else 'spoof' for key in list(batch_meta1[4])])
            sys_id_list.extend([dataset2.sysid_dict_inv[s.item()]
                                for s in list(batch_meta1[3])])
            score_list.extend(batch_score.tolist())
    #============================================================
            # 只判断真假 所以用bonafide和spoof列。
            len = batch_y1.shape[0] 
            cat_scores(batch_score,batch_y1,len)

    eer,threshold = compute_eer(target_scores,nontarget_scores)
    
    arr=arr[1:,:]
    np.save("s1s3_embed.npy",arr)

    print("eer = ",eer)
    print("threshold = ",threshold)
    print("target_scores =",target_scores.shape)
    print("nontarget_scores = ",nontarget_scores.shape)
#==============================================================
    with open(save_path, 'w+') as fh:
        for f, s, k, cm in zip(fname_list, sys_id_list, key_list, score_list):
            if not dataset2.is_eval:
                fh.write('{} {} {} {}\n'.format(f, s, k, cm))
            else:
                 fh.write('{} {} {}\n'.format(f,k,cm))
    print('Result saved to {}'.format(save_path))


  
def train_epoch(data_loader,model,device,lr):
    running_loss = 0
    num_correct = 0.0
    num_total = 0.0
    i = 0
    #optim1 = torch.optim.Adam(model.parameters(), lr=lr,amsgrad=True) # 优化算法
    optim1 = torch.optim.Adam(model.parameters(), lr=lr)
    weight = torch.FloatTensor([1.0, 9.0]).to(device)

    criterion = nn.CrossEntropyLoss(weight=weight)#交叉熵损失函数
    
    #scheduler = MultiStepLR(optim1, [5, 8], 0.1)
    for data in data_loader:
        model.train() 
        data1,data2=data[0],data[1]
        batch_x1,batch_y1,batch_meta1=data1#cqt
        batch_x2,batch_y2,batch_meta2=data2#mgd

        batch_size = batch_x1.size(0)# batch_size=32
        num_total += batch_size
        i += 1
        
        batch_x1 = batch_x1.to(device)
        batch_y1 = batch_y1.view(-1).type(torch.int64).to(device)
        batch_x2 = batch_x2.to(device)
        
        x,y,fai_x,fai_y,T,Q,m,mu_y,batch_out=model(batch_x1,batch_x2,device)#batch_out (64,256)
        em_loss=model.loss_em(x,y,fai_x,fai_y,T,Q,m,mu_y,device)

        batch_score = (batch_out[:, 1] - batch_out[:, 0]).data.cpu().numpy().ravel()
       

        len = batch_y1.shape[0]
        cat_scores(batch_score,batch_y1,len)
       
        cross_loss = criterion(batch_out,batch_y1)
        loss=0.00001*em_loss+cross_loss#将em_loss的量纲更改成同cross_loss l=0.1
       
        _, batch_pred = batch_out.max(dim=1)#找概率最大的按行输出
       
        num_correct += (batch_pred == batch_y1).sum(dim=0).item()
     
        
        running_loss += (loss.item() * batch_size)
       
        if i % 10 == 0:# 输出正确率
            sys.stdout.write('\r \t {:.2f}aaaaaa'.format((num_correct/num_total)*100))
            print('-------------')
            print('loss',loss)
            print('em_loss',em_loss*0.00001)
            print('cross_loss',cross_loss)
       
        #optimizer_pretrain.zero_grad()
        optim1.zero_grad()

        loss.backward()
        optim1.step()
        #optimizer_pretrain.step()
    #scheduler.step()
    running_loss /= num_total
    train_accuracy = (num_correct/num_total)*100
# =============EER======================
    print()
    eer,threshold = compute_eer(target_scores,nontarget_scores)
    print("eer = ",eer)
    print("threshold = ",threshold)
        
    return running_loss, train_accuracy

#----------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    print("开始执行!")#解析命令行参数
    parser = argparse.ArgumentParser('UCLANESL ASVSpoof2019  model')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='eval mode')#action含义一旦有eval这个选项则保存为true,否则默认为False
    parser.add_argument('--model_path', type=str,
                        default=None, help='Model checkpoint')
    parser.add_argument('--eval_output', type=str, default=None,
                        help='Path to save the evaluation result')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved mdoel')
    parser.add_argument('--track', type=str, default='logical')
    parser.add_argument('--features1', type=str, default='spect')#设置feature1
    parser.add_argument('--features2', type=str, default='spect')#设置feature2
    parser.add_argument('--is_eval', action='store_true', default=False)
    parser.add_argument('--eval_part', type=int, default=0)
    parser.add_argument('--break_from', type=int, default=0,help='the checkpoint file to resume from')
    
    
    #model = nn.DataParallel(net, device_ids=[0,1,2]  )# Multiple GPUs
    model=net.to(device)

    if not os.path.exists('s1s3_0.00001model'):
        os.mkdir('s1s3_0.00001model')
    args = parser.parse_args()
    track = args.track # track = logical
    assert args.features1 in ['mfcc', 'spect', 'cqcc','mgd'], 'Not supported feature'
    assert args.features2 in ['mfcc', 'spect', 'cqcc','mgd'], 'Not supported feature'
    model_tag = 'model_{}_{}_{}_{}_{}'.format(
        track, args.features1,args.features2, args.num_epochs, args.batch_size)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join('s1s3_0.00001model', model_tag) #模型保存路径
    assert track in ['logical', 'physical'], 'Invalid track given'
    is_logical = (track == 'logical')
   
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

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

#----------------------------------------------------------------------------------------------------
    dev_set1 = data_utils2.ASVDataset(is_train=False, is_logical=is_logical,
                                    transform=transforms1,
                                    feature_name=args.features1, is_eval=args.eval, eval_part=args.eval_part)


    dev_set2 = data_utils2.ASVDataset(is_train=False, is_logical=is_logical,
                                    transform=transforms2,
                                    feature_name=args.features2, is_eval=args.eval, eval_part=args.eval_part)
    myDevset = data_utils.MyDataset(dataset1=dev_set1, dataset2=dev_set2)                               
    dev_loader = DataLoader(myDevset, batch_size=args.batch_size, shuffle=True,pin_memory=False,drop_last=True)
#----------------------------------------------------------------------------------------------

    start_epoch=0
    if args.break_from and args.model_path:#程序若中断，从上次断掉的模型处开始加载模型
        checkpoint=torch.load(args.model_path)
        start_epoch=checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        print('加载epoch{}成功'.format(start_epoch))
        print('Model loaded : {}'.format(args.model_path))
    elif args.model_path and args.eval:
        #checkpoint=torch.load(args.model_path,map_location={'cuda:2':'cuda:0'})
        checkpoint=torch.load(args.model_path)
        model.load_state_dict(checkpoint['model'])
        print('Model loaded : {}'.format(args.model_path))
    else:
        start_epoch=0
        print('无保存模型，将从头开始训练')

#--------------------------------------------------------------------------------------------------
    if args.eval:
        assert args.eval_output is not None, 'You must provide an output path'
        assert args.model_path is not None, ''
        produce_evaluation_file(myDevset,dev_set1, model, device, args.eval_output)
        sys.exit(0) # 无错误退出，1是有错误退出
#-------------------------------------------------------------------------------------------------
    train_set1 = data_utils.ASVDataset(is_train=True, is_logical=is_logical, transform=transforms1,
                                      feature_name=args.features1)
    train_set2 = data_utils.ASVDataset(is_train=True, is_logical=is_logical, transform=transforms2,
                                      feature_name=args.features2)
    myTrainset = data_utils.MyDataset(dataset1=train_set1, dataset2=train_set2)
    train_loader = DataLoader(
        myTrainset, batch_size=args.batch_size, num_workers=num_workers,shuffle=True,pin_memory=False,drop_last=True)

    
    num_epochs = args.num_epochs
    writer = SummaryWriter('logs/{}'.format(model_tag))
    lr=args.lr
    df = pd.DataFrame(columns=['epoch','train Loss','training accuracy'])#列名
    df.to_csv("s1s3_0.00001.csv",index=False)
    for epoch in range(start_epoch+1,num_epochs+1):
        # if (epoch) % 20==0 and (epoch)<=50:
        #     lr = (lr)/(epoch)
        # elif epoch+1>50:
        #     lr = 1.0e-10
        running_loss, train_accuracy = train_epoch(train_loader,model,device,lr) 
        #valid_accuracy = evaluate_accuracy(dev_loader, model, device)
        writer.add_scalar('train_accuracy', train_accuracy, epoch)
        #writer.add_scalar('valid_accuracy', valid_accuracy, epoch)

        list = [epoch, running_loss, train_accuracy]
        data = pd.DataFrame([list])
        data.to_csv("s1s3_0.00001.csv",mode='a',header=False,index=False)#mode设为a,就可以向csv文件追加数据了

        writer.add_scalar('loss', running_loss, epoch)
        print('\n{} - {} - {:.2f} '.format(epoch,running_loss, train_accuracy))
       
        if epoch%5==0 and epoch<=90:
            state={'model':model.state_dict(),'epoch':epoch}
            torch.save(state,os.path.join(model_save_path, 'epoch_{}.pth'.format(epoch)))
        if epoch>90:
            state={'model':model.state_dict(),'epoch':epoch}
            torch.save(state,os.path.join(model_save_path, 'epoch_{}.pth'.format(epoch)))
      
        