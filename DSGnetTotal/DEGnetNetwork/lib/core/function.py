# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#tensorboard
from tensorboardX import SummaryWriter

import time
import logging

import torch

from core.evaluate import accuracy


logger = logging.getLogger(__name__)

def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict,writer):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    accuracy = AverageMeter()


    # switch to train mode
    model.train()
    end = time.time()
    for i, (input,maskinput, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        #target = target - 1 # Specific for imagenet
        #print(input.shape)
        # compute output
        output = model(input,maskinput)
        target = target.cuda(non_blocking=True)
        loss = criterion(output, target)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        #开始注释 top1,top5
        #prec1, prec5 = accuracy(output, target, (1, 5))
        _, output = torch.max(output, dim = -1)
        output = torch.eq(output,target)
        accuracy.update(torch.sum(output, dim = -1, out=None).item(), input.size(0))

        #top1.update(prec1[0], input.size(0))
        #top5.update(prec5[0], input.size(0))
        # measure elapsed time

        batch_time.update(time.time() - end)
        end = time.time()
        if i % config.PRINT_FREQ == 0:
            #print('epoch : {} , i : {} , loss : {} '.format(str(epoch),str(i),str(losses.avg)) , 'acc : {} '.format(str(accuracy.acc)))
            print('epoch : {} , i : {} , loss : {} '.format(str(epoch), str(i), str(losses.avg)))
            writer.add_scalar('train_loss', losses.avg , (34000 / 32 + 1) * epoch + i)
            #writer.add_scalar('acc', accuracy.acc, (34000 / 32 + 1) * epoch + i)
            losses.val = 0
            losses.avg = 0
            losses.sum = 0
            losses.count = 0
        '''
        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy@1 {top1.val:.3f} ({top1.avg:.3f})\t' \
                  'Accuracy@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, top1=top1, top5=top5)
            logger.info(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer.add_scalar('train_top1', top1.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1
        '''


def validate(config, val_loader, model, criterion, output_dir, tb_log_dir,
             writer,epoch,writer_dict=None,):

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    accuracy = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input,maskinput, target)  in enumerate(val_loader):

            # compute output
            output = model(input,maskinput)

            target = target.cuda(non_blocking=True)

            loss = criterion(output, target)

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            #prec1, prec5 = accuracy(output, target, (1, 5))
            #top1.update(prec1[0], input.size(0))
            #top5.update(prec5[0], input.size(0))
            _, output = torch.max(output, dim = -1)
            output = torch.eq(output, target)
            accuracy.update(torch.sum(output, dim = -1, out=None).item(),input.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        print(' val losses :  {} '.format(str(losses.avg)) , 'acc : {} '.format(str(accuracy.avg2)))
        #print(' val losses :  {} '.format(str(losses.avg)))
        writer.add_scalar('val_loss', losses.avg, epoch)
        #writer.add_scalar('acc', accuracy.avg2, epoch)

        '''
        
        msg = 'Test: Time {batch_time.avg:.3f}\t' \
              'Loss {loss.avg:.4f}\t' \
              'Error@1 {error1:.3f}\t' \
              'Error@5 {error5:.3f}\t' \
              'Accuracy@1 {top1.avg:.3f}\t' \
              'Accuracy@5 {top5.avg:.3f}\t'.format(
                  batch_time=batch_time, loss=losses, top1=top1, top5=top5,
                  error1=100-top1.avg, error5=100-top5.avg)
        logger.info(msg)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('valid_loss', losses.avg, global_steps)
            writer.add_scalar('valid_top1', top1.avg, global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1
            
        '''
    return losses.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.acc = 0
        self.sum2 = 0
        self.avg2 = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.sum2 += val
        self.count += n
        self.acc = val / n
        self.avg = self.sum / self.count
        self.avg2 = self.sum2 / self.count
