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
from models import SpectrogramModel, CQCCModel, resnet18_cbam, MFCCModel
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from eval_metrics import compute_eer
from torchvision.models.resnet import Bottleneck, BasicBlock
from bigru_am import VGGVox1, OnlineTripletLoss, VGGVox2
from resnet import resnet


import torch.optim as optim
import random
import pandas as pd
import torch.optim as optim
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
embedding_dim = 512   # you can change it to 256 
batch_size = 32 
num_workers = 8
n_classes = 2
n_samples = 10

pretrain_lr_init = 0.05
pretrain_lr_last = 0.0001
pretrain_epoch_num = 60

margin = 0.3
triplet_lr_init = 0.005
triplet_lr_last = 0.00005
triplet_epoch_num = 80 #500

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device_ids = [0,4,6,7]


WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9

#--------------------------------------------------------

net = VGGVox2(BasicBlock, [2, 2, 2, 2],num_classes=n_classes,emb_dim=embedding_dim)


optimizer_triplet = optim.SGD(net.parameters(), lr=triplet_lr_init, momentum=MOMENTUM,
                         weight_decay=WEIGHT_DECAY)
gamma_triplet = 10 ** (np.log10(triplet_lr_last / triplet_lr_init) / (triplet_epoch_num - 1))

lr_scheduler_triplet = optim.lr_scheduler.StepLR(optimizer_triplet, step_size=1, gamma=gamma_triplet)


optimizer_pretrain = optim.SGD(net.parameters(), lr=pretrain_lr_init, momentum=MOMENTUM,
                           weight_decay=WEIGHT_DECAY)#优化器
gamma_pretrain = 10 ** (np.log10(pretrain_lr_last / pretrain_lr_init) / (pretrain_epoch_num - 1))

lr_scheduler_pretrain = optim.lr_scheduler.StepLR(optimizer_pretrain, step_size=1, gamma=gamma_pretrain)
#动态调整学习率

criterion_pretrain = nn.CrossEntropyLoss().to(device)#损失函数

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

def pad(x, max_len=64000):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = (max_len / x_len)+1
    x_repeat = np.repeat(x, num_repeats)
    padded_x = x_repeat[:max_len]
    return padded_x


def evaluate_accuracy(data_loader, model, device):
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    for batch_x, batch_y, batch_meta in data_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x,'pretrain')
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
    return 100 * (num_correct / num_total)


def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    true_y = []
    fname_list = []
    key_list = []
    sys_id_list = []
    key_list = []
    score_list = []
    arr = np.zeros((1,512))
    for batch_x, batch_y, batch_meta in data_loader:
        '''
        batch_meta =  ASVFile(speaker_id, file_name, path, sys_id, key)
        '''

        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_out ,emb= model(batch_x,'pretrain')

        batch_score = (batch_out[:, 1] - batch_out[:, 0]
                       ).data.cpu().numpy().ravel()
        
      
        emb=emb.cpu().detach().numpy()
        arr=np.concatenate((arr,emb),axis = 0)

        # add outputs
        fname_list.extend(list(batch_meta[1]))
        key_list.extend(
            ['bonafide' if key == 1 else 'spoof' for key in list(batch_meta[4])])
        sys_id_list.extend([dataset.sysid_dict_inv[s.item()]
                            for s in list(batch_meta[3])])
        score_list.extend(batch_score.tolist())
#============================================================
        # 只判断真假 所以用bonafide和spoof列。
        len = batch_y.shape[0] 
        cat_scores(batch_score,batch_y,len)
    eer,threshold = compute_eer(target_scores,nontarget_scores)

    arr=arr[1:,:]
    np.save("gru.npy",arr) 
   
    print("eer = ",eer)
    print("threshold = ",threshold)
    print("target_scores =",target_scores.shape)
    print("nontarget_scores = ",nontarget_scores.shape)
#==============================================================
    with open(save_path, 'w') as fh:
        for f, s, k, cm in zip(fname_list, sys_id_list, key_list, score_list):
            if not dataset.is_eval:#？
                fh.write('{} {} {} {}\n'.format(f, s, k, cm))
            else:
                 fh.write('{} {} {}\n'.format(f,k,cm))
    print('Result saved to {}'.format(save_path))


#----------------------------------------------------------------------------
def train_epoch(data_loader, model, device):
    running_loss = 0
    num_correct = 0.0
    num_total = 0.0
    i = 0
    model.train() 
    
    for batch_x, batch_y, batch_meta in data_loader:
        batch_size = batch_x.size(0)# batch_size=32
        num_total += batch_size
        i += 1

        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out,emb= model(batch_x,'pretrain')

       
        batch_score = (batch_out[:, 1] - batch_out[:, 0]).data.cpu().numpy().ravel()#ravel函数将数组维度拉为一维数组
      

        len = batch_y.shape[0]
        cat_scores(batch_score,batch_y,len)

        loss = criterion_pretrain(batch_out,batch_y)
        
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()#?
        running_loss += (loss.item() * batch_size)
        if i % 10 == 0:# 输出正确率
            sys.stdout.write('\r \t {:.2f}aaaaaa'.format((num_correct/num_total)*100))
        optimizer_pretrain.zero_grad()
        loss.backward()
        optimizer_pretrain.step()
      
    running_loss /= num_total
    train_accuracy = (num_correct/num_total)*100
# =============EER======================
    print()
    eer,threshold = compute_eer(target_scores,nontarget_scores)
    print("eer = ",eer)
    print("threshold = ",threshold)
        
    return running_loss, train_accuracy


#----------------------------------------------------------------------------------------------------
def get_log_spectrum(x):
    s = librosa.core.stft(x, n_fft=2048, win_length=2048, hop_length=512)
    a = np.abs(s)**2
    #melspect = librosa.feature.melspectrogram(S=a)
    feat = librosa.power_to_db(a)
    return feat


def compute_mfcc_feats(x):
    mfcc = librosa.feature.mfcc(x, sr=16000, n_mfcc=24)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(delta)
    feats = np.concatenate((mfcc, delta, delta2), axis=0)
    return feats

if __name__ == '__main__':
    print("开始执行!")#解析命令行参数
    parser = argparse.ArgumentParser('UCLANESL ASVSpoof2019  model')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='eval mode')#action含义一旦有eval这个选项则保存为true,否则默认为False
    parser.add_argument('--model_path', type=str,
                        default=None, help='Model checkpoint')
    parser.add_argument('--eval_output', type=str, default=None,
                        help='Path to save the evaluation result')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=100)
    #parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved mdoel')
    parser.add_argument('--track', type=str, default='logical')
    parser.add_argument('--features', type=str, default='spect')
    parser.add_argument('--is_eval', action='store_true', default=False)
    parser.add_argument('--eval_part', type=int, default=0)
    parser.add_argument('--break_from', type=int, default=0,help='the checkpoint file to resume from')


    #device = 'cuda:7' if torch.cuda.is_available() else 'cpu'
    #print("device:",device)

    if not os.path.exists('gru3_models'):
        os.mkdir('gru3_models')
    args = parser.parse_args()
    track = args.track # track = logical
    assert args.features in ['mfcc', 'spect', 'cqcc'], 'Not supported feature'
    model_tag = 'model_{}_{}_{}_{}'.format(
        track, args.features, args.num_epochs, args.batch_size)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join('gru3_models', model_tag) #模型保存路径
    assert track in ['logical', 'physical'], 'Invalid track given'
    is_logical = (track == 'logical')
    # 该if语句不执行
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    if args.features == 'mfcc':
        feature_fn = compute_mfcc_feats
        model_cls = MFCCModel()

    elif args.features == 'spect':
        feature_fn = cqtgram
        model_cls = net

    elif args.features == 'cqcc':
        feature_fn = None  # cqcc feature is extracted in Matlab script
        model_cls = CQCCModel()

    transforms = transforms.Compose([
        lambda x: pad(x),
        lambda x: feature_fn(x),
        lambda x: Tensor(x)
    ])
#-----------------------------------------------------------------------------------
    train_set = data_utils.ASVDataset(is_train=True, is_logical=is_logical, transform=transforms,
                                      feature_name=args.features)
    
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, num_workers=num_workers,shuffle=True)
    
#------------------------------------------------------------------------------------------------------

    dev_set = data_utils2.ASVDataset(is_train=False, is_logical=is_logical,
                                    transform=transforms,
                                    feature_name=args.features, is_eval=args.eval, eval_part=args.eval_part)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=True)
#--------------------------------------------------------------------------------------------------

    #model = nn.DataParallel(model_cls, device_ids=device_ids)  # Multiple GPUs
    model=model_cls.to(device)
    start_epoch=0
    if args.break_from and args.model_path:#程序若中断，从上次断掉的模型处开始加载模型
        checkpoint=torch.load(args.model_path)
        start_epoch=checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        print('加载epoch{}成功'.format(start_epoch))
        print('Model loaded : {}'.format(args.model_path))
    elif args.model_path and args.eval:
       
        checkpoint=torch.load(args.model_path)
        # model.load_state_dict(torch.load(args.model_path,map_location={'cuda:4': 'cuda:0'}))
        # print('Model loaded : {}'.format(args.model_path))
        model.load_state_dict(checkpoint['model'])
       
        print('Model loaded : {}'.format(args.model_path))
    else:
        start_epoch=0
        print('无保存模型，将从头开始训练')

    if args.eval:
        assert args.eval_output is not None, 'You must provide an output path'
        assert args.model_path is not None, ''
        produce_evaluation_file(dev_set, model, device, args.eval_output)
        sys.exit(0) # 无错误退出，1是有错误退出

    #创建train_acc.csv和var_acc.csv文件，记录loss和accuracy
    df = pd.DataFrame(columns=['epoch','train Loss','training accuracy'])#列名
    df.to_csv("gru.csv",index=False)

    num_epochs = args.num_epochs
    writer = SummaryWriter('logs/{}'.format(model_tag))
    for epoch in range(start_epoch+1,num_epochs+1):
        running_loss, train_accuracy = train_epoch(train_loader,model,device) 
        #valid_accuracy = evaluate_accuracy(dev_loader, model, device)
        
        
        list = [epoch, running_loss, train_accuracy]
        data = pd.DataFrame([list])
        data.to_csv("gru.csv",mode='a',header=False,index=False)#mode设为a,就可以向csv文件追加数据了

        writer.add_scalar('train_accuracy', train_accuracy, epoch)
        #writer.add_scalar('valid_accuracy', valid_accuracy, epoch)
        writer.add_scalar('loss', running_loss, epoch)
        
        print('\n{} - {} - {:.2f} - '.format(epoch,
                                                   running_loss, train_accuracy))
       
        if epoch%5==0:
            state={'model':model.state_dict(),'epoch':epoch}
            torch.save(state,os.path.join(model_save_path, 'epoch_{}.pth'.format(epoch)))