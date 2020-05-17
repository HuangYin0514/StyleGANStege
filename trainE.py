import argparse
import os
import time
from random import random

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F

from utils import checkpointNet, util_logger
from utils.util import *
from test import test

# speed up
cudnn.benchmark = True


# ---------------------- Train function ----------------------
def train(train_dataloader, model,stylegan, device, save_dir_path, args):
    '''
        train
    '''
    # start_time -----------------------------------------------
    start_time = time.time()

    # Logger instance--------------------------------------------
    logger = util_logger.Logger(save_dir_path)
    logger.info('-' * 10)
    logger.info(vars(args))
    logger.info(model)
    

        # Testing / Validating-----------------------------------
        # if (step + 1) % args.test_every == 0 or step + 1 == args.num_train_steps:
        #     torch.cuda.empty_cache()
        #     test(model, save_dir_path, args, num=step)

    # stop time ---------------------------------------------------
    time_elapsed = time.time() - start_time
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # Save final model weights-----------------------------------
    checkpointNet.save_network(model, save_dir_path, 'final')
