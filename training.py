import torch
import logging
import os
import torch.nn.functional as F
import numpy as np
import time
import scipy.io as scio
## Get the same logger from main"
logger = logging.getLogger("our_model")


def train(args, model, train_loader, macro_loader, optimizer, epoch, batch_size):
    model.train()
    perm = np.arange(len(train_loader))
    np.random.shuffle(perm)
    perm = list(perm)
    # all_coef = []

    for idx, batch_idx in enumerate(perm):
        data, label, adj_close_prices, time_list = train_loader[batch_idx]
        # print('data shape:', data.shape)
        data = torch.from_numpy(data).float()
        data = data.cuda()
        data = data.permute(1,0,-1)   #32*20*108  batchsize*seq_len*feat_dims

        if args.just_price == 'True':  #just consider price info
            data = data[:,:,-8:]
        # print('reshaped data shape:', data.shape)
        # data = data.float().unsqueeze(1).to(device) # add channel dimension
        if data.shape[1] != 20:
            continue
        optimizer.zero_grad()
        hidden = model.init_hidden(len(data), use_gpu=True)
        acc, loss, hidden, coef = model(data, macro_loader, adj_close_prices, time_list, hidden, label)

        loss.backward()
        optimizer.step()
        lr = optimizer.update_learning_rate()
        # if idx % args.log_interval == 0:
        logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tlr:{:.5f}\tacc:{:.4f}\tLoss: {:.6f}'.format(
            epoch, idx, len(train_loader),
            100. * idx / len(train_loader), lr, acc, loss.item()))
        # all_coef.append(coef.cpu().detach().numpy())

    # scio.savemat('coeficients_v3.mat', {'coef': all_coef})
    return coef.cpu().detach().numpy()


def snapshot(dir_path, run_name, state):

    snapshot_file = os.path.join(dir_path,
                                 run_name + '-model_best.pth')

    torch.save(state, snapshot_file)
    logger.info("Snapshot saved to {}\n".format(snapshot_file))
