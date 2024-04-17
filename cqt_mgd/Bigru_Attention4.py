import math
import torch
import torch.nn as nn
# -*- coding: utf-8 -*-

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(MLP, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, out_size)
        )
    def forward(self, x):
        out = self.linear(x)
        return out

class Bigru_Attention(nn.Module):
    def __init__(self,input_size,input_embeding):
        super(Bigru_Attention, self).__init__()

        self.hidden_size = 512 #隐藏层维度
        self.num_layers = 2  #网络层数  几层GRU的意思
        self.input_size = input_size  #输入维度 
        self.embedding_size  = input_embeding

        #batch_first :batch_first 表示输入数据的形式，默认是 False，就是这样形式，(seq, batch, feature)，也就是将序列长度放在第一位，batch 放在第二位
        self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers,
                          batch_first=True,bidirectional=True)  #GRU单元
    
    
        self.fc = nn.Linear(2*self.hidden_size, self.embedding_size)
        self.dropout = nn.Dropout(0.75)  #随机失活
        self._initialize_weights()  #参数权重初始化 h0

    def forward(self, x):
        self.gru.flatten_parameters()
        out, _ = self.gru(x, None)
        #print('gru:{}'.format(out.shape))
        out=out.view(out.size(0),out.size(1),2,self.hidden_size)#(32,1,2,512)
        #print('tgru:{}'.format(out.shape))
        out_forward=out[:,:,0,:]
        #print('fgru:{}'.format(out_forward.shape))
        out_backward=out[:,:,1,:]
        #print('bgru:{}'.format(out_backward.shape))
        
        query1,  key1, value1 = self.dropout(out_forward), self.dropout(out_forward), self.dropout(out_forward)
        out1, weight1 = self.attention_net(query1, key1, value1)

        query2,  key2, value2 = self.dropout(out_backward), self.dropout(out_backward), self.dropout(out_backward)
        out2, weight2 = self.attention_net(query2, key2, value2)

        final=torch.cat((out1,out2),2)#(32,1,1024)
        #print('cat:{}'.format(final.shape))

        out = final.view(final.size(0),-1)#(32,1024)
        #print('pre_fc:{}'.format(out.shape))
       
        out = self.fc(out)#(32,512)
        #print('fc:{}'.format(out.shape))
        return out



    #自注意力层
    def attention_net(self, query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = torch.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.normal_(0.0, 0.001)



if __name__ == '__main__':
    #from config import cfg
    model = Bigru_Attention(512,512)#embedding 512
    input = torch.randn(32, 1, 512)
    out= model(input) 
  