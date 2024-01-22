import argparse
import logging
import os
import threading
import time
import numpy as np
import shutil
from os.path import join, exists, split
from math import log10
from train import train_rest

import sys
from PIL import Image
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
from datetime import datetime
from dataset import RestList
import data_transforms as transforms


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-s', '--crop-size', default=0, type=int) #
    parser.add_argument("--scale_factor", type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=16, metavar='N') #
    parser.add_argument('--epochs', type=int, default=10, metavar='N') #
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR')
    parser.add_argument('--gpu', default='mars0', type=str)
    parser.add_argument('--gamma', default='0.5', type=float)
    parser.add_argument('--n_steps', type=int, default=30)
    parser.add_argument('--memo', default=' ', type=str)
    parser.add_argument('--flag', default='', type=str)
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    
    return args
def main():
    args = parse_args()
    
    dt_now = datetime.now()
    timeName = "{:4d}{:02d}{:02d}{:02d}{:02d}".format(dt_now.year, dt_now.month, \
    dt_now.day, dt_now.hour, dt_now.minute)
    saveDirName = './runs/train/' + timeName + '_' + args.flag + '_' + args.gpu
    if not os.path.exists(saveDirName):
        os.makedirs(saveDirName, exist_ok=True)

    # logging configuration
    FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(saveDirName + '/log_training.log')
    logger.addHandler(file_handler)

    # if args.cmd == 'train':
    train_rest(args, saveDirName=saveDirName, logger=logger)
    # elif args.cmd == 'test':
    #     test_rest(args, logger=logger)

if __name__ == '__main__':
    main()
