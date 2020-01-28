from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


import math
import scipy.io as scio
import random
import pandas as pd
import numpy as np

from copula_estimate import GaussianCopula
from utils import macro_context_embedding, macro_gating_func, margin_estimate, stock_price_emb


class CoCPC(nn.Module):
    def __init__(self, timestep, batch_size, seq_len, rep_dim, feat_dim, var_num, dataset):

        super(CoCPC, self).__init__()

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.timestep = timestep
        self.rep_dim = rep_dim
        self.dataset = dataset
        self.encoder = nn.Sequential(
            nn.Linear(feat_dim, rep_dim),
            # nn.BatchNorm1d(rep_dim),
            nn.ReLU()
        ).cuda()

        self.gat_layer = nn.Sequential(
            nn.Linear(var_num, 1),
            nn.Sigmoid()
        )
        self.g_cop = GaussianCopula(dim=var_num).cuda()
        self.gru = nn.GRU(rep_dim, rep_dim//2, num_layers=1, bidirectional=False, batch_first=True)
        self.Wk  = nn.ModuleList([nn.Linear(rep_dim//2+64, rep_dim).cuda() for i in range(timestep)]) #256, 512
        self.softmax  = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()
        self.predictor = nn.Sequential(
            # nn.Linear(64, 32),
            nn.Linear(self.batch_size, 2),
            nn.Softmax()
        )

        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # initialize gru
        for layer_p in self.gru._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(_weights_init)

    def init_hidden(self, batch_size, use_gpu=True):
        if use_gpu: return torch.zeros(1, batch_size, self.rep_dim//2).cuda()
        else: return torch.zeros(1, batch_size, self.rep_dim//2)

    def regenerate_data(self, x, t, y, adj_p):
        '''
        generate positive and negative samples in each batch
        :param x: batchsize*seq_len*dim
        :param t: selected time for split the term and pred term
        :param y: seq_len*batchsize*1
        :param adj_p: seq_len*batchsize
        :return: x:batchsize*seq_len*dim, sample_labels:batchsize*1
        '''
        data = x
        negative_num = self.batch_size//4
        pred_terms = t + 1
        sentence_labels = np.ones((self.batch_size))
        for b in range(negative_num):
            # set random predictions for negative samples
            # each predicted term draws a result from a distribution that excludes itself
            numbers = np.arange(0, self.batch_size)
            rand_num = np.random.choice(numbers[numbers!=b], 1)
            data[b, -pred_terms:, :] = x[rand_num, -pred_terms:, :]
            sentence_labels[b] = 0

        #shuffle
        idxs = np.random.choice(self.batch_size, self.batch_size, replace=False)
        return data[idxs, ...], torch.from_numpy(sentence_labels[idxs]).float(), y[:, idxs, ...], adj_p[:, idxs]

    def macro_context(self, stock_close_price, macro_loader, time_list, t_index):
        '''

        :param stock_close_price: t*batchsize
        :param macro_loader:
        :param time_list: [datetime.date]*T
        :param t_index:
        :return: macro aggregated context embedding at time t; shape:1*dim
        '''

        x_time = time_list[t_index]
        print('type x_time:', type(x_time))
        batchsize = stock_close_price.shape[1]
        ## get macro embedding
        macro_x = macro_context_embedding(macro_loader)  #macro_x dict:{'DATE':[m1_date,...], 'emb':[m1_emb,...]} m1_date:list  m1_emb:tensor shape:1*T*dim
        macro_num = len(macro_x['emb'])
        macro_dim = macro_x['emb'][0].shape[2]
        macro_emb = torch.zeros(macro_num, macro_dim)
        macro_valid_len_list = []
        macro_dates_list = macro_x['DATE']
        macro_embs_list = macro_x['emb']


        for i in range(macro_num):
            m_date = macro_dates_list[i]
            valid_m_date = [d for d in m_date if d<=x_time.strftime("%Y-%m-%d")]  #probably empty
            # print('valid_m_date:',valid_m_date)
            t = len(valid_m_date)
            if t!=0:
                m_emb = macro_embs_list[i][:,t-1,:].squeeze()
            else:
                m_emb = torch.zeros(macro_dim)
            # print('shape m_emb:',m_emb.shape)
            macro_emb[i, :] = m_emb
            macro_valid_len_list.append(t)

        ## get gating function for macro
        stock_emb = stock_price_emb(stock_close_price)  #batchsize*seqlen*dim
        margin_params, batch_merged_emb = margin_estimate(stock_emb, macro_emb)  #mu: batchsize*variable_num*1  L:batchsize*variable_num*d
        batch_merged_emb = batch_merged_emb.cuda()
        # print('test gat matmul:', torch.matmul(self.gat_layer(test_macro_gat).permute(0, 2, 1), batch_merged_emb).shape)
        # print('batch_merged_emb shape:', batch_merged_emb.shape)
        macro_gat, loss_c = macro_gating_func(self.g_cop, margin_params, stock_close_price, macro_loader, macro_valid_len_list, time_list[:t_index+1],self.dataset)  #batchsize*macro_num
        macro_gat = macro_gat.cuda()

        # print('macro_gat shape:', macro_gat.shape)
        # print('get a batch coefficients:', macro_gat)
        # random_stock_ind = random.randrange(1,batchsize,1) - 1

        ##get R without back propagation
        # scio.savemat('raw_coefficient.mat', {'raw_coef': list(macro_gat[0,:,:].cpu().detach().numpy())})
        # return

        return torch.matmul(self.gat_layer(macro_gat).permute(0,2,1), batch_merged_emb), loss_c, self.gat_layer(macro_gat)[0,:,:]



    def forward(self, x, macro_loader, adj_close_prices, x_time_list, hidden, label):
        '''

        :param x: stock features shape:batchsize*T*dim
        :param hidden: initialized hidden for gru
        :param adj_close_prices: T*batchsize
        :param x_time_list: str list, list the time of this batch data
        :param label: T*batchsize*1
        :return:
        '''
        batch = x.size()[0]
        label = torch.from_numpy(label).float().long()
        # t_samples = torch.randint(self.seq_len/160-self.timestep, size=(1,)).long() # randomly pick one time stamp in [0,128-12)
        # t_samples = random.randint(0, self.seq_len-self.timestep-1)  # the right one

        t_samples = random.randint(14, self.seq_len - self.timestep - 1)  #test for macro
        # input sequence is N*C*L, e.g. 8*1*20480
        re_x, samples_label, label, adj_close_prices = self.regenerate_data(x, t_samples, label, adj_close_prices)
        re_x = re_x.cuda()
        label = label.cuda()
        samples_label = samples_label.cuda()
        z = self.encoder(re_x) #batchsize*seq_len*dim 32*20*64
        # encoded sequence is N*C*L, e.g. 8*512*128
        # reshape to N*L*C for GRU, e.g. 8*128*512
        loss_nce = 0 # average over timestep and batch
        correct = 0
        # loss = 0
        # loss_func = nn.CrossEntropyLoss()
        encode_samples = torch.empty((self.timestep,batch,self.rep_dim)).float().cuda() # e.g. size 12*8*512
        y = torch.empty((self.timestep, 1, batch)).long().cuda()
        # print('z shape:', z.shape)
        for i in np.arange(1, self.timestep+1):
            # print(z[:,t_samples+i,:].shape)
            encode_samples[i-1] = z[:,t_samples+i,:].view(batch,self.rep_dim) # z_tk e.g. size 8*512 samples need to predict
            y[i-1] = label[t_samples+i, :, :].view(batch)
        forward_seq = z[:,:t_samples+1,:] # e.g. 32*11*64  e.g. size 8*100*512
        output, hidden = self.gru(forward_seq, hidden) # output size e.g. 8*100*256
        c_x_t = output[:,t_samples,:].view(batch, self.rep_dim//2) # c_x_t e.g. size 8*256
        # print('c_x_t shape:', c_x_t.shape)
        print('t_samples:', t_samples)
        c_q_t, loss_c, coef = self.macro_context(adj_close_prices[:t_samples+1, :], macro_loader, x_time_list, t_samples) #
        c_q_t = c_q_t.squeeze()
        # print('c_1_t shape:', c_q_t.shape)
        c_t = torch.cat((c_x_t, c_q_t), 1)

        pred = torch.empty((self.timestep, batch, self.rep_dim)).float().cuda() # e.g. 12*32*64 size 12*8*512
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t) # Wk*c_t e.g. 32*64 size 8*512


        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i],0,1)) # e.g. 32*32 size 8*8  z_{t+k}W_{k}c_{t}
            pred_v = self.predictor(total)
            correct += torch.sum(torch.eq(torch.argmax(pred_v.long(), dim=1), samples_label.long())) # correct if predict the true positive samples or negative samples
            loss_nce += torch.sum(torch.diag(self.lsoftmax(total))/torch.sum(torch.triu(total))) # loss_nce is a tensor
            # loss += loss_func(pred_v, samples_label.long())
        loss_nce /= -1.*batch*self.timestep
        loss_c /= batch*self.timestep
        accuracy = float(correct) / (batch*self.timestep)

        # print('acc:', accuracy, '\tloss_nce:',loss_nce, '\tloss_c:', loss_c)
        return accuracy, loss_nce+loss_c, hidden, coef


    def predict(self, x):
        z = self.encoder(x)
        return z

class StockClassifier(nn.Module):
    def __init__(self, inputsize):
        super(StockClassifier, self).__init__()
        self.hidden_size = int(inputsize//2)
        self.classifier = nn.LSTM(
            input_size=inputsize,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )

    def forward(self, encoder, x):
        x = encoder.predict(x)
        x, _ = self.classifier(x)
        return self.out(x)