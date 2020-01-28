import torch
import logging
import torch.nn as nn
import os
import math
import numpy as np


# logger = logging.getLogger('cdc')

loss_func = nn.CrossEntropyLoss()
def get_accurate(pred, target):
    # pred1: T*batchsize*2
    # target1: T*batchsize

    T = pred.shape[0]
    batchsize = pred.shape[1]
    pred_ = torch.argmax(pred, dim=2)

    acc = float(torch.eq(pred_, target).sum()) / (T*batchsize)
    return acc

def create_confision_matrix(target, pred):
    """
           shape: [T x batch_size, y_size]
           pred: [T x batch_size * 2]
           target: [T x batch_size * 1]
    """
    # target = target.long()
    n_samples = pred.shape[0]
    label_ref = target.reshape(n_samples)
    # label_hyp = torch.gt(pred, 0.5).long().reshape(n_samples)  #T*batch_size
    label_hyp = torch.argmax(pred, dim=1).reshape(n_samples)

    p_in_hyp = torch.sum(label_hyp)
    n_in_hyp = n_samples - p_in_hyp

    # positive class: up
    tp = torch.sum(label_hyp.mul(label_ref))
    fp = p_in_hyp - tp



    # negative class: down
    # print('label_ref + label_hyp nonzero:', len(torch.nonzero(label_ref + label_hyp)))
    tn = n_samples - len(torch.nonzero(label_ref + label_hyp))
    fn = n_in_hyp - tn

    return float(tp), float(fp), float(tn), float(fn)


def get_mcc(pred, target):
    tp, fp, tn, fn = create_confision_matrix(target.reshape(-1, 1), pred.reshape(-1, 2))
    core_de = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    # print('core_de:', core_de)
    return (tp * tn - fp * fn) / math.sqrt(core_de) if core_de else None

def stock_train(logger, args, cpc_model, stock_model, train_loader, epoch, optimizer):
    logger.info('********* start stock training *********')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    stock_model.train()
    acc_list = []
    mcc_list = []
    perm = np.arange(len(train_loader))
    np.random.shuffle(perm)
    perm = list(perm)
    total_loss = 0
    for idx, batch_idx in enumerate(perm):
        # if idx >= 100:  #varify efficient classfication with less labeled data
        #     break
        data, label, _, _= train_loader[batch_idx]
        # print('data shape:', data.shape)
        data = torch.from_numpy(data).float()
        data = data.cuda()
        data = data.permute(1, 0, -1)  # 32*20*108  batchsize*seq_len*feat_dims
        if args.just_price == 'True':  # just consider price info
            data = data[:, :, -8:]
        seq_len = data.shape[1]
        # print('reshaped data shape:', data.shape)
        # data = data.float().unsqueeze(1).to(device) # add channel dimension
        label = torch.from_numpy(label).float().long()  #seq_len*batchsize*1
        label = label.cuda().squeeze()

        optimizer.zero_grad()
        data = data.cuda()
        pred = stock_model(cpc_model, data)
        pred = pred.permute(1, 0, -1)  # seq_len*batchsize*2
        loss = 0
        for j in range(seq_len):
            temp = loss_func(pred[j,:,:], label[j,:])
            loss += temp
        loss = loss/seq_len
        total_loss += loss.item()
        acc = get_accurate(pred, label)
        mcc = get_mcc(pred, label)
        mcc = (mcc if mcc != None else 0)

        acc_list.append(acc)
        mcc_list.append(mcc)

        loss.backward()
        optimizer.step()
        # lr = optimizer.update_learning_rate()
        if idx % args.log_interval == 0:
            logger.info('Stock Train Epoch: {} [{}/{} ({:.0f}%)]\tacc:{:.4f}\tmcc:{:.4f}\tLoss: {:.6f}'.format(
                epoch, idx, len(train_loader),
                100. * idx / len(train_loader), acc, mcc, loss.item()))
    total_loss /= len(train_loader)
    logger.info(' ====> Stock Training Set: Average loss: {:.4f}\t Average acc: {:.4f}\t Var acc: {:.4f}\t Average mcc: {:.4f}\t Var mcc:{:.4f}\n'.format(
        total_loss, np.mean(acc_list), np.var(acc_list), np.mean(mcc_list), np.var(mcc_list)
    ))
    return total_loss



def stock_validation(logger, args, cpc_model, stock_model, valid_loader):
    logger.info('********* start stock validation *********')
    stock_model.eval()
    total_loss = 0
    total_acc = 0
    total_mcc = 0
    with torch.no_grad():
        for batch_data in valid_loader:
            data, label, _, _ = batch_data
            data = torch.from_numpy(data).float()
            data = data.cuda()
            data = data.permute(1, 0, -1)  # 32*20*106  batchsize*seq_len*feat_dims

            if args.just_price == 'True':  # just consider price info
                data = data[:, :, -8:]
            seq_len = data.shape[1]

            label = torch.from_numpy(label).float().long()  # seq_len*batchsize*1
            label = label.cuda().squeeze()

            pred = stock_model(cpc_model, data)
            pred = pred.permute(1, 0, -1)  # seq_len*batchsize*2
            loss = 0
            for j in range(seq_len):
                loss += loss_func(pred[j, :, :], label[j, :])
            loss /= seq_len

            acc = get_accurate(pred, label)
            mcc = get_mcc(pred, label)
            mcc = (mcc if mcc != None else 0)
            total_loss += loss.item()
            total_acc += acc
            total_mcc += mcc
        total_loss /= len(valid_loader)
        total_acc /= len(valid_loader)
        total_mcc /= len(valid_loader)

    logger.info('===>Stock Validation set: Average loss: {:.4f}\tAccuracy:{:.4f}\t MCC:{:.4f}\n'.format(
        total_loss, total_acc, total_mcc))

    return total_loss, total_acc, total_mcc

