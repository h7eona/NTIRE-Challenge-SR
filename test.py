import argparse
from tokenize import Name
from unittest import removeResult
import torch
from torch import nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import data_transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import numpy as np
import glob
import os
import time
import shutil
import sys
import logging
from datetime import datetime
from network import iPASSRNet, Discriminator
from dataset import RestList
from utils import save_checkpoint, psnr, AverageMeter, GANLoss, Folder_Create, Evaluation, save_output_images
from PIL import Image

def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--scale_factor", type=int, default=4, help="how much the model will SR. default=4")
    parser.add_argument("--mode", type=str, default="T", help="validation: V, test: T")
    parser.add_argument("--load_weight", type=str, default="bestmodel.pth.tar", help="weight file without default=bestmodel")
    # parser.add_argument("--testset", type=str, default="Validation")
    parser.add_argument("--saveres", type=str, default="test_result", help="folder name to save results. default=test_result")
    args = parser.parse_args() 

    print(' '.join(sys.argv))
    print(args)
    
    return args

def val(model, valloader):
    model.eval()
    criterion_eval = Evaluation().cuda()
    score_psnr = AverageMeter()
    for i, (HR_left, HR_right, LR_left, LR_right, name) in enumerate(valloader):
        HR_left = Variable(HR_left).float().cuda()
        HR_right = Variable(HR_right[0]).float().cuda()
        LR_left = Variable(LR_left[0]).float().cuda()
        LR_right = Variable(LR_right[0]).float().cuda()
        # name = name[0]
        
        with torch.no_grad():
            SR_left, SR_right = model(LR_left, LR_right, is_training=0)
            SR_left = torch.clamp(SR_left, min=0, max=1)
            SR_right = torch.clamp(SR_right, min=0, max=1)

            score_psnr_L  = criterion_eval(SR_left, HR_left, 'PSNR')
            score_psnr.update(score_psnr_L, LR_left.size(0))
            score_psnr_R  = criterion_eval(SR_right, HR_right, 'PSNR')
            score_psnr.update(score_psnr_R, LR_right.size(0))
    print(' * PSNR  Score is {s.avg:.3f}'.format(s=score_psnr))


def test(model, testloader, savedir):
    model.eval()
    print("imgs to be tested:",len(testloader))
    for i, (LR_left, LR_right, name) in enumerate(testloader):
        # HR_left = Variable(HR_left).float().cuda()
        # HR_right = Variable(HR_right).float().cuda()
        LR_left = Variable(LR_left[0]).float().cuda()
        LR_right = Variable(LR_right[0]).float().cuda()
        name = name[0]
        
        with torch.no_grad():
            SR_left, SR_right = model(LR_left, LR_right, is_training=0)
            print(name)
            print("saving",name+'_L.png', 'and',name+'_R.png', "in progress....")
            SR_left = torch.clamp(SR_left, min=0, max=1)
            SR_right = torch.clamp(SR_right, min=0, max=1)

            savepng(SR_left, '_L', name, savedir)
            savepng(SR_right, '_R', name, savedir)
            # savepng(SR_left, '', name, savedir)
            # savepng(SR_right, '', name, savedir)
            

def savepng(tensor, dr, name, savedir):
    SR = torch.clamp(tensor, min=0, max=1)
    SR = SR.cpu().data.numpy() * 255.0
    savename = os.path.join(savedir, name+dr+'.png')
    # print(SR.shape)
    result = Image.fromarray(np.transpose(SR[0,:,:,:], (1, 2, 0)).astype(np.uint8))
    result.save(savename, quality=100)

def main():
    args = parse_args()
    scale = args.scale_factor
    root = "../../01_Data/"
    # testfolder = root+args.testset
    savefolder = args.saveres
    # weights = args.load + '.pth.tar'
    # print(weights)
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)
    device = 'cuda'
    # load model
    
    # make testloader
    t_pair = [#transforms.RandomCrop(crop_size),
                # transforms.Resize(crop_size),
                transforms.RandomFlip(),
                transforms.ToTensor()] # transform function for paired training data
    v_pair = [transforms.Resize_pair(), transforms.ToTensor()] # transform function for paired validation data

    v_unpair = [transforms.ToTensor_One()] # transform function for unpaired validation training data

    # Define dataloaders
    Val_loader = torch.utils.data.DataLoader(
        RestList('Validation', transforms.Compose(v_pair), transforms.Compose(v_unpair), out_name=True),
        batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    Val_NTIRE_loader = torch.utils.data.DataLoader(
        RestList('NTIRE', transforms.Compose(t_pair), transforms.Compose(v_unpair), out_name=True),
        batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    
    if args.mode == "V":

        Net = iPASSRNet(scale).to(device)
        # checkpoint = torch.load(args.load + '.pth.tar')
        checkpoint = torch.load(args.load_weight)
        # print(checkpoint.keys())
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['Net'].items():
            if k[0:6]=='module':
                name = k[7:] # remove `module.`
                # print('module....')
            else:
                name = k[0:]
            new_state_dict[name] = v
        Net.load_state_dict(new_state_dict)
        val(Net, Val_loader)

    else:
        Net = iPASSRNet(scale).to(device)
        # checkpoint = torch.load(args.load + '.pth.tar')
        checkpoint = torch.load(args.load_weight)
        # print(checkpoint.keys())
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['Net'].items():
            if k[0:6]=='module':
                name = k[7:] # remove `module.`
                # print('module....')
            else:
                name = k[0:]
            new_state_dict[name] = v
        Net.load_state_dict(new_state_dict)
        test(Net, Val_NTIRE_loader, savefolder)

    print('test finished')



if __name__ == '__main__':
    main()
