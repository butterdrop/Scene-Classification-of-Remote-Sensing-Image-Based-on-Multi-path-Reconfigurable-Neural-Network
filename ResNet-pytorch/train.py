import numpy as np
from tqdm import tqdm
from utils.utils import AverageMeter
from utils.accuracy import accuracy
import torch

#-------------------------------------------------------------------------------
# train model
def train_epoch(model, train_loader, criterion, optimizer, e, epoch, device):
    loss_show = AverageMeter()
    acc1 = AverageMeter()
    acc3 = AverageMeter()
    loop = tqdm(enumerate(train_loader), total = len(train_loader))
    for batch_idx, (batch_data, batch_label) in loop:
        batch_data = batch_data.to(device).float()
        batch_label = batch_label.to(device)

        optimizer.zero_grad()
        batch_prediction = model(batch_data)
        loss = criterion(batch_prediction, batch_label)
        loss.backward()
        optimizer.step()       

        # calculate the accuracy
        acc_batch = accuracy(batch_prediction, batch_label, topk=(1,2))
        n = batch_data.shape[0]

        # update the loss and the accuracy 
        loss_show.update(loss.data, n)
        acc1.update(acc_batch[0].data, n)
        acc3.update(acc_batch[1].data, n)

        loop.set_description(f'Train Epoch [{e+1}/{epoch}]')
        # loop.set_postfix({"train_loss":loss_show.average.item(),
        #                   "train_accuracy1": str(round(acc1.average.item(), 3)) + "%",
        #                   "train_accuracy3": str(round(acc3.average.item(), 3)) + "%"})

    return acc1.average.item(), acc3.average.item(), loss_show.average.item()
#-------------------------------------------------------------------------------

