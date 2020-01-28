## Utilities
import argparse
import random
import time
import os
import logging
from timeit import default_timer as timer
import pickle
import sys
import scipy.io as scio

## Libraries
import numpy as np

## Torch
import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import torch.optim as optim



from model import CoCPC, StockClassifier
from utils import *
from training import train, snapshot
from validation import validation
from logger_v1 import setup_logs
from stock_train_pred import stock_train, stock_validation

run_name = "our_model-"

class ScheduledOptim(object):
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, n_warmup_steps):
        self.optimizer = optimizer
        self.d_model = 128
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.delta = 1

    def state_dict(self):
        self.optimizer.state_dict()

    def step(self):
        """Step by the inner optimizer"""
        self.optimizer.step()

    def zero_grad(self):
        """Zero out the gradients by the inner optimizer"""
        self.optimizer.zero_grad()

    def increase_delta(self):
        self.delta *= 2

    def update_learning_rate(self):
        """Learning rate scheduling per step"""

        self.n_current_steps += self.delta
        new_lr = np.power(self.d_model, -0.5) * np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr


def CoCPC_main(logger, args, train_loader, validation_loader, macro_loader):
    global_timer = timer()  # global timer
    print('multivariable_num:', args.vr_num)

    if args.gpu:
        model = CoCPC(args.timestep, args.batch_size, args.seq_len, args.rep_dim, args.feat_dim, args.vr_num, args.dataset).cuda()


    # optimizer
    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True),
        args.n_warmup_steps)

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('### Model summary below###\n {}\n'.format(str(model)))
    logger.info('===> Model total parameter: {}\n'.format(model_params))
    ## Start co_cpc training
    best_acc = 0
    best_loss = np.inf
    best_epoch = -1
    val_acc_list = []
    val_loss_list = []
    all_epoch_coef = []
    for epoch in range(1, args.epochs + 1):
        epoch_timer = timer()

        # Train and validate
        epoch_coef = train(args, model, train_loader, macro_loader, optimizer, epoch, args.batch_size)
        val_acc, val_loss = validation(args, model, validation_loader, macro_loader, args.batch_size)

        all_epoch_coef.append(epoch_coef)
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)

        #Save
        # if val_loss < best_loss:
        #     best_loss = min(val_loss, best_loss)
        #     snapshot(args.logging_dir, run_name, {
        #         'epoch': epoch + 1,
        #         'validation_acc': val_acc,
        #         'state_dict': model.state_dict(),
        #         'validation_loss': val_loss,
        #         'optimizer': optimizer.state_dict(),
        #     })
        #     best_epoch = epoch + 1
        #
        if val_acc > best_acc:
            best_acc = max(val_acc, best_acc)
            snapshot(args.logging_dir, run_name, {
                'epoch': epoch + 1,
                'validation_acc': val_acc,
                'state_dict': model.state_dict(),
                'validation_loss': val_loss,
                'optimizer': optimizer.state_dict(),
            })
            best_epoch = epoch + 1

        elif epoch - best_epoch > 2:
            optimizer.increase_delta()
            best_epoch = epoch + 1


        end_epoch_timer = timer()
        logger.info("#### End epoch {}/{}, elapsed time: {}".format(epoch, args.epochs, end_epoch_timer - epoch_timer))
    # scio.savemat('all_epoch_coef.mat', {'all_epoch_coef':all_epoch_coef})
    ## end

    print('val loss mean: {:.4f}, val loss var:{:.4f}, val acc mean: {:.4f}, val acc var: {:.4f}'.format(np.mean(val_loss_list), np.var(val_loss_list), np.mean(val_acc_list), np.var(val_acc_list)))
    end_global_timer = timer()
    logger.info("################## Success #########################")
    logger.info("Total elapsed time: %s" % (end_global_timer - global_timer))


if __name__ == '__main__':
    ## Settings
    parser = argparse.ArgumentParser(description='Copula-CPC Stock Prediction')
    parser.add_argument('--logging-dir', type=str, default='log',
                        help='model save directory')
    parser.add_argument('--dataset', type=str, default='acl18', help='choose dataset to run')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--rep_dim', type=int, default=108, help='representation dimension')
    parser.add_argument('--feat_dim', type=int, default=108, help='if just stock price then 8')
    parser.add_argument('--vr_num', type=int, default=14, help='variable num for joint distribution')
    parser.add_argument('--seq_len', type=int, default=20,
                        help='window length to sample from each stock')
    parser.add_argument('--timestep', type=int, default=1, help='prediction steps')
    parser.add_argument('--gpu', type=str, default='0',help='CUDA training')
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--version', type=str, default='v1', help='define runing version')
    parser.add_argument('--cpc_train', type=str, default='False', help='whether to train our co-cpc model or prediction based on trained model')
    parser.add_argument('--just_price', type=str, default='False', help='whether features contain tweet info')
    parser.add_argument('--macro_type', type=str, default='whole', help='choose macros depends on their time interval')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    run_name = run_name + args.version
    logger = setup_logs(args.logging_dir, run_name)  # setup logs

    logger.info('===> loading train, validation and eval dataset')

    if args.dataset == 'acl18':
        train_loader = pickle.load(open('./data/stock_datasets/train_stock_200_batchs.pkl', 'rb'))
        validation_loader = pickle.load(open('./data/stock_datasets/valid_stock_50_batchs.pkl', 'rb'))
        test_loader = pickle.load(open('./data/stock_datasets/test_stock_50_batchs.pkl', 'rb'))
        macro_loader = macro_data_load(type=args.macro_type)
        if args.just_price == 'True':
            args.feat_dim = 8

    else:
        train_loader = pickle.load(open('./data/kdd17/train_stock_200_batchs.pkl', 'rb'))
        validation_loader = pickle.load(open('./data/kdd17/valid_stock_50_batchs.pkl', 'rb'))
        test_loader = pickle.load(open('./data/kdd17/test_stock_50_batchs.pkl', 'rb'))
        args.feat_dim = 7
        macro_loader = macro_data_load(start_date_str='2007-01-01', end_date_str='2017-01-01', type=args.macro_type)

    args.vr_num = len(macro_loader) + 1  #add a stock

    if args.cpc_train == 'True':
        CoCPC_main(logger, args, train_loader[:30], validation_loader[:10], macro_loader)  #just use small dataset for training Co-CPC

    ### load encoder model for prediction

    cpc_model = CoCPC(args.timestep, args.batch_size, args.seq_len, args.rep_dim, args.feat_dim, args.vr_num, args.dataset)
    checkpoint = torch.load(os.path.join(args.logging_dir,
                                         run_name + '-model_best.pth'))
    cpc_model.load_state_dict(checkpoint['state_dict'])
    cpc_model.eval()
    # print(type(cpc_model))

    clsf_model = StockClassifier(args.rep_dim)
    if args.gpu:
        clsf_model = clsf_model.cuda()

    optimizer = optim.Adagrad(clsf_model.parameters(), lr=0.002)
    all_train_loss = []
    all_valid_loss = []
    valid_acc_list = []
    valid_mcc_list = []
    for epoch in range(1, args.epochs + 1):
        train_loss = stock_train(logger, args, cpc_model, clsf_model, train_loader, epoch, optimizer)
        valid_loss, valid_acc, valid_mcc = stock_validation(logger, args, cpc_model, clsf_model, test_loader)
        all_train_loss.append(train_loss)
        all_valid_loss.append(valid_loss)
        valid_acc_list.append(valid_acc)
        valid_mcc_list.append(valid_mcc)
    best_index = valid_acc_list.index(max(valid_acc_list))
    logger.info('***** best validation acc :{:.4f}\t mcc:{:.4f}\n'.format(valid_acc_list[best_index],
                                                                          valid_mcc_list[best_index]))
    # scio.savemat('./result/our_model_loss_'+args.version+'.mat', {'train_loss': all_train_loss, 'valid_loss': all_valid_loss})