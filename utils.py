import pandas as pd
import os
from _datetime import datetime
import torch
from torch import nn
import numpy as np
from copula_estimate import GaussianCopula
import scipy
# from scipy.stats import norm
import torch.optim as optim


class MarginNet(torch.nn.Module):
    def __init__(self, param_dim, L_param_dim=32):
        super(MarginNet, self).__init__()
        self.mu_layer = nn.Linear(param_dim, 1)
        self.L_layer = nn.Linear(param_dim, L_param_dim)


    def forward(self, x): #tau
        mu = self.mu_layer(x)
        L = self.L_layer(x)
        return mu,L




# add last one year macro info
def macro_data_load(start_date_str='2013-01-01', end_date_str='2016-01-02', type = 'whole'):

    path = './data/macroeconomics/'
    files = os.listdir(path)
    files.sort()
    print('macro data name:')
    # valid_data_list = [d.strftime("%Y-%m-%d") for d in pd.date_range(start_date_str,end_date_str,freq='D')]
    macro_data = []
    # print('end date str split yyear type:', str(end_date_str.split('-')[0]))
    year_interval = int(end_date_str.split('-')[0]) - int(start_date_str.split('-')[0])
    for file in files:
        print(file.split('_')[0])
        csv_data = pd.read_csv(path+file)
        valid_data = csv_data[(csv_data['DATE']>=start_date_str)&(csv_data['DATE']<=end_date_str)]
        column_name = valid_data.columns.to_numpy()[1]
        valid_data = valid_data[(True ^ valid_data[column_name].isin(['.']))]  # delete non value

        if type == 'daily':
            if len(valid_data) > 53*year_interval:
                macro_data.append(valid_data)
        elif type == 'weekly':
            if len(valid_data) > 12*year_interval+1 and len(valid_data) <= 53*year_interval:
                macro_data.append(valid_data)
        elif type == 'monthly':
            if len(valid_data) <=12*year_interval+1 and len(valid_data) > 4*year_interval:
                macro_data.append(valid_data)
        else:
            macro_data.append(valid_data)

    return macro_data

def macro_aligned(macro_data, dataset):
    '''
    fill up macro data in stock timeline
    :param macro_data:
    :return: filled macro data  type:numpy shape:[macro_num, whole_len_time]
    '''
    macro_num = len(macro_data)
    if dataset == 'acl18':
        start_date = '2013-01-01'
        end_date = '2016-01-02'
    else:
        start_date = '2007-01-01'
        end_date = '2017-01-01'
    unified_date = [d.strftime('%Y-%m-%d') for d in pd.date_range(start=start_date, end=end_date, freq='D')]
    unified_value = np.zeros((macro_num, len(unified_date)))
    for i, m_data in enumerate(macro_data):
        m_date = m_data['DATE'].tolist()
        column_name = m_data.columns.to_numpy()[1]
        # print('unified_date:', unified_date[0])
        # print('type unified_date:', type(unified_date[0]))
        idxs = [unified_date.index(k) for k in m_date]
        m_value = m_data[column_name].tolist()
        for j, idx in enumerate(idxs):
            unified_value[i, idx] = m_value[j]
            if j==0 & idx!=0:
                unified_value[i, :idx] = m_value[j]

            if j!=len(idxs)-1:
                unified_value[i, idx:idxs[j+1]] = m_value[j]
            else:
                unified_value[i, idx:] = m_value[j]
    return unified_date, unified_value


def macro_stock_combine(stock_price, aligned_macro_data, unified_date, stock_prev_time):
    '''

    :param stock_price: t
    :param aligned_macro_data: numpy macro_num*whole_len_time
    :param stock_prev_time: time list before the current time
    :return: combined stock price and macros  shape: time_len * macro_num+1
    '''
    # print('each stock_price shape:', stock_price.shape)
    macro_num = aligned_macro_data.shape[0]
    merged = np.zeros((len(stock_prev_time), macro_num+1))

    merged[:, 0] = stock_price
    for i, s_time in enumerate(stock_prev_time):
        idx = unified_date.index(s_time.strftime('%Y-%m-%d'))
        merged[i, 1:] = aligned_macro_data[:, idx]
    # print(merged[0,:])
    return merged


def macro_gating_func(g_cop, margin_params, stock_price, macro_data, macro_valid_len_list, stock_prev_time, dataset):
    '''
    apply copula model to learn gating function for aggregating macros
    :param margin_params (batch_mu, batch_L)  tensor batch_mu: batchsize*var_num*1  batch_L: batchsize*var_num *L_dim
    :param stock_price: t*batchsize
    :param macro_data: [dataframe,...]
    :param macro_valid_len_list: valid len of each macro, some may be zeros
    :param stock_prev_time: list: time before current step
    :return: gating value size: batchsize*macro_num
    '''


    batch_size = stock_price.shape[1]
    vr_num = len(macro_data) + 1

    # print('shape stock_price:', stock_price.shape)


    gat = torch.zeros(batch_size, vr_num, vr_num)

    macro_unified_date, macro_unified_value = macro_aligned(macro_data, dataset)
    batch_mu, batch_L = margin_params


    loss = 0
    for i in range(batch_size):
        data = macro_stock_combine(stock_price[:, i], macro_unified_value, macro_unified_date, stock_prev_time) #14 variables
        # print('data shape:', data.shape)
        # print('data:', data)
        hyperparam = []
        for k in range(vr_num):
            mu = batch_mu[i, k, :]
            vec_l = batch_L[i, k, :].unsqueeze(0)
            sigma = torch.matmul(vec_l, vec_l.permute(1,0))

            # print('sigma:',sigma)
            hyperparam.append({'loc':mu, 'scale':sigma})

        loss += g_cop(data, margins='Normal', hyperparam=hyperparam)
        gat[i, :, :] = g_cop.get_R()

    return gat, loss


def macro_context_embedding(macro_data, embedding_dim=32):
    macro_num = len(macro_data)
    # m_data:dataframe ['DATE','xx']
    macro_embedding = dict()
    macro_embedding['DATE'] = []
    macro_embedding['emb'] = []

    multi_grain_num = 0
    existed_len = []
    hidden = []
    autoreg = []
    macro_label = []

    for i, m_data in enumerate(macro_data):
        if len(m_data['DATE']) not in existed_len:
            existed_len.append(len(m_data['DATE']))
            hidden.append(torch.zeros(1, 1, 64))
            autoreg.append(nn.GRU(embedding_dim * 2 + 1, 64, num_layers=1, bidirectional=False, batch_first=True))  # 64
            multi_grain_num += 1

        macro_label.append(existed_len.index(len(m_data['DATE'])))

    for i, m_data in enumerate(macro_data):

        multi_grain_index = macro_label[i]
        # date embedding
        # print('m_data:', m_data)
        dayofweeks = pd.to_datetime(m_data['DATE']).dt.dayofweek.tolist()
        # print('dayofweeks:', type(dayofweeks))
        dayofweeks = torch.from_numpy(np.array(dayofweeks)).long()
        # print('dayofweeks:', type(dayofweeks))
        dayofweeks_embedding = nn.Embedding(7, embedding_dim)(dayofweeks)
        dayofyear = pd.to_datetime(m_data['DATE']).dt.dayofyear.tolist()
        for m in dayofyear:
            if m > 366:
                print('out of range data:', m)

        dayofyear = torch.from_numpy(np.array(dayofyear)).long()
        # print('dayofyear shape:',dayofyear.shape)

        dayofyear_embedding = nn.Embedding(367, embedding_dim)(dayofyear)

        # value embedding
        # print(m_data.icol(1))
        # print(dayofweeks_embedding.shape)
        column_name = m_data.columns.to_numpy()[1]
        # print(column_name)
        # print(type(m_data[column_name]))
        value = torch.from_numpy(np.array(list(map(float,m_data[column_name].tolist())))).float()
        value = value.unsqueeze(1)
        merged = torch.cat((dayofweeks_embedding,dayofyear_embedding),1)
        # print(merged.shape)
        # print(value.shape)
        merged = torch.cat((merged, value),1)
        merged = merged.unsqueeze(0) #1*seq_len*dim

        e, _ = autoreg[multi_grain_index](merged, hidden[multi_grain_index])

        macro_embedding['DATE'].append(m_data['DATE'].tolist())
        macro_embedding['emb'].append(e)
    #     print(macro_each_dict['emb'].shape)
    # print(macro_embedding[0]['DATE'].tolist())
    return macro_embedding

def stock_price_emb(stock_close_price):
    '''

    :param stock_close_price: t*batchsize
    :return: stock embedding of this batch
    '''
    # print('stock close price shape:', stock_close_price.shape)
    scp = torch.from_numpy(np.transpose(stock_close_price)[...,np.newaxis]).float() #batchsize*t*1
    # print('scp shape:',scp.shape)
    batchsize = scp.shape[0]
    enc = nn.Sequential(
        nn.Linear(1, 64),
        nn.ReLU()
    )
    hidden = torch.zeros(1, batchsize, 64)
    autoreg = nn.GRU(64, 64, num_layers=1, bidirectional=False, batch_first=True)
    z = enc(scp)
    e, _ = autoreg(z, hidden)
    # print('e shape:', e.shape)
    return e



def margin_estimate(stock_embs, macro_embs):
    '''
    estimate margin distribution based on  variable embedding
    :param stock_embs: tensor: batchsize * t * dim
    :param macro_embs: tensor: macro_num * dim
    :return:
    '''

    emb_dim = macro_embs.shape[-1]
    MN = MarginNet(emb_dim)
    batchsize = stock_embs.shape[0]
    batch_mu = torch.zeros(batchsize, macro_embs.shape[0]+1, 1)
    batch_L = torch.zeros(batchsize, macro_embs.shape[0]+1, 32)
    batch_merged_embs = torch.zeros(batchsize, macro_embs.shape[0]+1, emb_dim)
    for i in range(batchsize):
        s_emb = stock_embs[i,-1,:].unsqueeze(0)
        # print('shape s_emb:', s_emb.shape)
        # print('macro embs:', macro_embs.shape)
        stacked_emb = torch.cat((s_emb, macro_embs), 0)
        mu, L = MN(stacked_emb)
        batch_mu[i,:, :] = mu
        batch_L[i,:, :] = L
        batch_merged_embs[i, :, :] = stacked_emb
    return (batch_mu, batch_L), batch_merged_embs




if __name__ == '__main__':
    macro_data = macro_data_load()
    macro_context_embedding(macro_data)
