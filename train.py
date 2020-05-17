import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from dataloader import getDataLoader
from models import build_model
from utils import util


# Schedule learning rate--------------------------------------------
def adjust_lr(epoch, optimizer, args):
    step_size = 40
    lr = args.lr * (0.1 ** (epoch // step_size))
    for g in optimizer.param_groups:
        g['lr'] = lr * g.get('lr_mult', 1)


# ---------------------- Train function ----------------------
def train(model, criterion, optimizer, scheduler, dataloader, num_epochs, device, save_dir_path, args):
    '''
        train
    '''
    # start_time -----------------------------------------------
    start_time = time.time()

    # Logger instance--------------------------------------------
    logger = util.Logger(save_dir_path)
    logger.info('-' * 10)
    logger.info(vars(args))
    logger.info(model)

    # +++++++++++++++++++++++++++++++++start++++++++++++++++++++++++++++++++++++++++
    for epoch in range(num_epochs):
        logger.info('Epoch {}/{}'.format(epoch + 1, num_epochs))

        model.train()
        adjust_lr(epoch, optimizer, args)

        # ===================one epoch====================
        # Training
        running_loss = 0.0
        batch_num = 0
        for data in dataloader:
            inputs, labels, _ = data
            batch_num += 1

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            # Sum up the stripe softmax loss-------------------
            loss = 0
            if isinstance(outputs, (list,)):
                for logits in outputs:
                    stripe_loss = criterion(logits, labels)
                    loss += stripe_loss
            elif isinstance(outputs, (torch.Tensor,)):
                loss = criterion(outputs, labels)
            else:
                raise Exception('outputs type is error !')

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
        # ===================one epoch end================

        epoch_loss = running_loss / len(dataloader.dataset)
        logger.info('Training Loss: {:.4f}'.format(epoch_loss))

        # Save result to logger---------------------------------
        logger.x_epoch_loss.append(epoch + 1)
        logger.y_train_loss.append(epoch_loss)

        # Testing / Validating-----------------------------------
        if (epoch + 1) % 20 == 0 or epoch + 1 == num_epochs:
            torch.cuda.empty_cache()
            CMC, mAP = test(model, args.dataset, args.dataset_path, 512)
            logger.info('Testing: top1:%.2f top5:%.2f top10:%.2f mAP:%.2f' % (CMC[0], CMC[4], CMC[9], mAP))

            logger.x_epoch_test.append(epoch + 1)
            logger.y_test['top1'].append(CMC[0])
            logger.y_test['mAP'].append(mAP)
            if epoch + 1 != num_epochs:
                util.save_network(model, save_dir_path, str(epoch + 1))
        logger.info('-' * 10)

    # +++++++++++++++++++++++++++++++++start end+++++++++++++++++++++++++++++++++

    # Save the loss curve-----------------------------------
    logger.save_curve()

    time_elapsed = time.time() - start_time
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # Save final model weights-----------------------------------
    util.save_network(model, save_dir_path, 'final')
