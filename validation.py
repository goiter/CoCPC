import numpy as np
import logging
import torch
import torch.nn.functional as F

## Get the same logger from main"
logger = logging.getLogger("our_model")

def validation(args, model, data_loader, macro_loader, batch_size):
    logger.info("Starting Validation")
    model.eval()
    total_loss = 0
    total_acc  = 0

    with torch.no_grad():
        for batch_data in data_loader:
            data, label, adj_close_prices, time_list = batch_data
            data = torch.from_numpy(data).float()
            data = data.cuda()
            data = data.permute(1, 0, -1)  #32*20*106  batchsize*seq_len*feat_dims

            if args.just_price == 'True':  # just consider price info
                data = data[:, :, -8:]

            if data.shape[1] != 20:
                continue
            # data = data.float().unsqueeze(1).to(device) # add channel dimension
            hidden = model.init_hidden(len(data), use_gpu=True)
            acc, loss, hidden, coef = model(data, macro_loader, adj_close_prices, time_list, hidden, label)
            total_loss += loss
            total_acc += acc


    total_loss /= len(data_loader) # average loss
    total_acc /= len(data_loader)


    logger.info('===> Validation set: Average loss: {:.4f}\tAccuracy:{:.4f}\n'.format(
                total_loss.item(), total_acc))

    return total_acc, total_loss.item()