import os
import time
import shutil
import sys
import logging
from datetime import datetime
from network import iPASSRNet, Discriminator
from dataset import RestList
from utils import save_checkpoint, psnr, AverageMeter, GANLoss, Folder_Create, Evaluation, save_output_images

import torch
from torch import nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import data_transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import numpy as np

def Train(loaders, models, optims, criterions,  epoch, scale, best_score, output_dir, eval_score=None, print_freq=10, logger=None):
    # Counters
    losses_Recon = AverageMeter()
    losses_Recon_masked = AverageMeter()
    losses_G = AverageMeter()
    losses_D = AverageMeter()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    score_psnr = AverageMeter()
    score_ssim = AverageMeter()
    score_lpip = AverageMeter()
    
    # Loaders, criterions, models, optimizers
    Train_loader, Val_loader, Val_NTIRE_loader = loaders
    criterion_GAN, criterion_eval = criterions    
    Net = models
    optim = optims # 


    Net.train()

    end = time.time()
    save_dir = os.path.join(output_dir, 'epoch_{:04d}'.format(epoch+1),'Val')
    for i, (HR_left, HR_right,LR_left, LR_right) in enumerate(Train_loader):
        data_time.update(time.time() - end)

        ############################################################
        # (1) Load and construct data
        ############################################################
        HR_left = Variable(HR_left).float().cuda()
        HR_right = Variable(HR_right).float().cuda()
        
        _, _, H, W = HR_left.shape
        # LR_left = F.interpolate(HR_left, size=(H//4, W//4), mode='bicubic')
        # LR_right = F.interpolate(HR_right, size=(H//4, W//4), mode='bicubic')
        LR_left = Variable(LR_left).float().cuda()
        LR_right = Variable(LR_right).float().cuda()
        b, c, h, w = LR_left.shape

        # save_output_images(LR_left, '_LRLeft', name, save_dir)
        # save_output_images(LR_right, '_LRRight', name, save_dir)
        # print("## dataloader", H,W)
        
        ############################################################
        # (2) Feed the image to the generator
        ############################################################

        SR_left, SR_right, (M_right_to_left, M_left_to_right), (V_left, V_right)\
                = Net(LR_left, LR_right, is_training=1)

        ############################################################
        # (3) Compute loss functions
        # ##########################################################
        # loss_Disc = criterion_GAN(model_D(output.detach()), False) + criterion_GAN(model_D(img_var), True)

        # optim_D.zero_grad()
        # loss_Disc.backward()
        # optim_D.step()
        
        ### loss_SR
        loss_SR = F.l1_loss(SR_left, HR_left) + F.l1_loss(SR_right, HR_right)

        ### loss_photometric
        Res_left = torch.abs(HR_left - F.interpolate(LR_left, scale_factor=scale, mode='bicubic', align_corners=False))
        Res_left = F.interpolate(Res_left, scale_factor=1 / scale, mode='bicubic', align_corners=False)
        Res_right = torch.abs(HR_right - F.interpolate(LR_right, scale_factor=scale, mode='bicubic', align_corners=False))
        Res_right = F.interpolate(Res_right, scale_factor=1 / scale, mode='bicubic', align_corners=False)
        Res_leftT = torch.bmm(M_right_to_left.contiguous().view(b * h, w, w), Res_right.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
        Res_rightT = torch.bmm(M_left_to_right.contiguous().view(b * h, w, w), Res_left.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
        loss_photo = F.l1_loss(Res_left * V_left.repeat(1, 3, 1, 1), Res_leftT * V_left.repeat(1, 3, 1, 1)) + \
                        F.l1_loss(Res_right * V_right.repeat(1, 3, 1, 1), Res_rightT * V_right.repeat(1, 3, 1, 1))

        ### loss_smoothness
        loss_h = F.l1_loss(M_right_to_left[:, :-1, :, :], M_right_to_left[:, 1:, :, :]) + \
                F.l1_loss(M_left_to_right[:, :-1, :, :], M_left_to_right[:, 1:, :, :])
        loss_w = F.l1_loss(M_right_to_left[:, :, :-1, :-1], M_right_to_left[:, :, 1:, 1:]) + \
                F.l1_loss(M_left_to_right[:, :, :-1, :-1], M_left_to_right[:, :, 1:, 1:])
        loss_smooth = loss_w + loss_h

        ### loss_cycle
        Res_left_cycle = torch.bmm(M_right_to_left.contiguous().view(b * h, w, w), Res_rightT.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                        ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
        Res_right_cycle = torch.bmm(M_left_to_right.contiguous().view(b * h, w, w), Res_leftT.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                        ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
        loss_cycle = F.l1_loss(Res_left * V_left.repeat(1, 3, 1, 1), Res_left_cycle * V_left.repeat(1, 3, 1, 1)) + \
                    F.l1_loss(Res_right * V_right.repeat(1, 3, 1, 1), Res_right_cycle * V_right.repeat(1, 3, 1, 1))


        ### loss_consistency
        SR_left_res = F.interpolate(torch.abs(HR_left - SR_left), scale_factor=1 / scale, mode='bicubic', align_corners=False)
        SR_right_res = F.interpolate(torch.abs(HR_right - SR_right), scale_factor=1 / scale, mode='bicubic', align_corners=False)
        SR_left_resT = torch.bmm(M_right_to_left.detach().contiguous().view(b * h, w, w), SR_right_res.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                    ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
        SR_right_resT = torch.bmm(M_left_to_right.detach().contiguous().view(b * h, w, w), SR_left_res.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                    ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
        loss_cons = F.l1_loss(SR_left_res * V_left.repeat(1, 3, 1, 1), SR_left_resT * V_left.repeat(1, 3, 1, 1)) + \
                    F.l1_loss(SR_right_res * V_right.repeat(1, 3, 1, 1), SR_right_resT * V_right.repeat(1, 3, 1, 1))

        
        ### loss_total
        loss = loss_SR + 0.1 * loss_cons + 0.1 * (loss_photo + loss_smooth + loss_cycle)

        
        ### ----------new old----------- ###


        # ### loss_smoothness
        # loss_h = F.l1_loss(M_right_to_left[:, :-1, :, :], M_right_to_left[:, 1:, :, :]) + \
        #             F.l1_loss(M_left_to_right[:, :-1, :, :], M_left_to_right[:, 1:, :, :])
        # loss_w = F.l1_loss(M_right_to_left[:, :, :-1, :-1], M_right_to_left[:, :, 1:, 1:]) + \
        #             F.l1_loss(M_left_to_right[:, :, :-1, :-1], M_left_to_right[:, :, 1:, 1:])
        # loss_smooth = loss_w + loss_h

        # ### loss_cycle
        # Identity = Variable(torch.eye(w, w).repeat(b, h, 1, 1), requires_grad=False).cuda()
        # loss_cycle = F.l1_loss(M_left_right_left * V_left_to_right.permute(0, 2, 1, 3), Identity * V_left_to_right.permute(0, 2, 1, 3)) + \
        #                 F.l1_loss(M_right_left_right * V_right_to_left.permute(0, 2, 1, 3), Identity * V_right_to_left.permute(0, 2, 1, 3))

        # ### loss_photometric
        # LR_right_warped = torch.bmm(M_right_to_left.contiguous().view(b*h,w,w), LR_right.permute(0,2,3,1).contiguous().view(b*h, w, c))
        # LR_right_warped = LR_right_warped.view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
        # LR_left_warped = torch.bmm(M_left_to_right.contiguous().view(b * h, w, w), LR_left.permute(0, 2, 3, 1).contiguous().view(b * h, w, c))
        # LR_left_warped = LR_left_warped.view(b, h, w, c).contiguous().permute(0, 3, 1, 2)

        # loss_photo = F.l1_loss(LR_left * V_left_to_right, LR_right_warped * V_left_to_right) + \
        #                 F.l1_loss(LR_right * V_right_to_left, LR_left_warped * V_right_to_left)

        # ### losses
        # loss = loss_SR + 0.005 * (loss_photo + loss_smooth + loss_cycle)

        ############################################################
        # (4) Update networks
        # ##########################################################

        optim.zero_grad()
        loss.backward()
        optim.step()

        losses_Recon.update(loss.data, HR_left.size(0))
        losses_Recon_masked.update(loss_SR.data, HR_left.size(0))
        
        batch_time.update(time.time() - end)
        end = time.time()
        if i % print_freq == 0:
            logger.info('E : [{0}][{1}/{2}]\t'
                        'T {batch_time.val:.3f}\n'
                        'Recon {R.val:.4f} ({R.avg:.4f})\t'
                        'loss_SR {Rm.val:.4f} ({Rm.avg:.4f})\t'
                        'G {G.val:.4f} ({G.avg:.4f})\t'
                        'D {D.val:.4f} ({D.avg:.4f})\t'.format(
                epoch, i, len(Train_loader), batch_time=batch_time,
                R = losses_Recon, Rm = losses_Recon_masked, G = losses_G, D=losses_D))

    Net.eval()
    print("One train session is finished.")
    ###
    # Evaluate PSNR, SSIM, LPIPS with validation set
    ###

    # save_dir = os.path.join(output_dir, 'epoch_{:04d}'.format(epoch+1),'Val')
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir, exist_ok=True)
    # save_dir = os.path.join(output_dir, 'epoch_{:04d}'.format(epoch+1),'Val')
    # for i, (HR_left, HR_right, LR_left, LR_right, name) in enumerate(Val_loader):
    #     HR_left = Variable(HR_left).float().cuda()
    #     HR_right = Variable(HR_right[0]).float().cuda()
    #     LR_left = Variable(LR_left[0]).float().cuda()
    #     LR_right = Variable(LR_right[0]).float().cuda()
    #     # name = name[0]
        
    #     with torch.no_grad():
    #         SR_left, SR_right = Net(LR_left, LR_right, is_training=0)
    #         SR_left = torch.clamp(SR_left, min=0, max=1)
    #         SR_right = torch.clamp(SR_right, min=0, max=1)

    #         score_psnr_L  = criterion_eval(SR_left, HR_left, 'PSNR')
    #         score_ssim_L  = criterion_eval(SR_left, HR_left, 'SSIM')
    #         score_LPIPS_L = criterion_eval(SR_left, HR_left, 'LPIPS')
    #         score_psnr.update(score_psnr_L, LR_left.size(0))
    #         score_ssim.update(score_ssim_L, LR_left.size(0))
    #         score_lpip.update(score_LPIPS_L, LR_left.size(0))
    #         score_psnr_R  = criterion_eval(SR_right, HR_right, 'PSNR')
    #         score_ssim_R  = criterion_eval(SR_right, HR_right, 'SSIM')
    #         score_LPIPS_R = criterion_eval(SR_right, HR_right, 'LPIPS')
    #         score_psnr.update(score_psnr_R, LR_right.size(0))
    #         score_ssim.update(score_ssim_R, LR_right.size(0))
    #         score_lpip.update(score_LPIPS_R, LR_right.size(0))
            
    #         # resultname = os.path.join(save_dir, name[:-4] +'.jpg')
    #         # save_image(output[0], resultname, quality=100)
    #         # SR_left = F.interpolate(SR_left, size=(h, w), mode='bilinear')
            
    #         # SR_left = SR_left.cpu().data.numpy() * 255.0
    #         # SR_right = SR_right.cpu().data.numpy() * 255.0
    #         # HR_left = HR_left.cpu().data.numpy() * 255.0
    #         # HR_right = HR_right.cpu().data.numpy() * 255.0
    #         # LR_left = LR_left.cpu().data.numpy() * 255.0
    #         # LR_right = LR_right.cpu().data.numpy() * 255.0
    #         # input_name = os.path.join(save_dir, name[:-4] +'_input.jpg')
    #         # HRL_name = os.path.join(save_dir, name[:-4] +'_HRLeft.jpg')
    #         # HRR_name = os.path.join(save_dir, name[:-4] +'_HRRight.jpg')
    #         # LRL_name = os.path.join(save_dir, name[:-4] +'_LRLeft.jpg')
    #         # LRR_name = os.path.join(save_dir, name[:-4] +'_LRRight.jpg')
    #         # SRL_name = os.path.join(save_dir, name[:-4] +'_LRRight.jpg')

    #         # save_output_images(SR_left, '_SRLeft', name, save_dir)
    #         # save_output_images(SR_right, '_SRRight', name, save_dir)
    #         # save_output_images(HR_left, '_HRLeft', name, save_dir)
    #         # save_output_images(HR_right, '_HRRight', name, save_dir)
    #         # save_output_images(LR_left, '_LRLeft', name, save_dir)
    #         # save_output_images(LR_right, '_LRRight', name, save_dir)

    # if logger is not None:
    #     logger.info(' * PSNR  Score is {s.avg:.3f}'.format(s=score_psnr))
    #     logger.info(' * SSIM  Score is {s.avg:.4f}'.format(s=score_ssim))
    #     logger.info(' * LPIPS Score is {s.avg:.3f}'.format(s=score_lpip))

    return score_psnr.avg, score_ssim.avg, score_lpip.avg

def train_rest(args, saveDirName='.', logger=None):
    # Print the systems settings
    logger.info(' '.join(sys.argv))
    logger.info(args.memo)
    for k, v in args.__dict__.items():
        logger.info('{0}:\t{1}'.format(k, v))

    # Hyper-parameters
    batch_size = args.batch_size
    crop_size = args.crop_size
    lr = args.lr 
    scale = args.scale_factor
    best_score = 0
    
    # Define transform functions
    t_pair = [#transforms.RandomCrop(crop_size),
                # transforms.Resize(crop_size),
                transforms.RandomFlip(),
                transforms.ToTensor()] # transform function for paired training data
    t_unpair = [#transforms.RandomCrop_One(crop_size),
                # transforms.Resize_One(crop_size),
                transforms.RandomFlip_One(),
                transforms.ToTensor_One()] # transform function for unpaired training data
    v_pair = [transforms.Resize_pair(), transforms.ToTensor()] # transform function for paired validation data
    v_unpair = [transforms.ToTensor_One()] # transform function for unpaired validation training data

    # Define dataloaders
    Train_loader = torch.utils.data.DataLoader(
        RestList('Train', transforms.Compose(t_pair), transforms.Compose(t_unpair), batch=batch_size),
        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)
    Val_loader = torch.utils.data.DataLoader(
        RestList('Validation', transforms.Compose(v_pair), transforms.Compose(v_unpair), out_name=True),
        batch_size=2, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    Val_NTIRE_loader = torch.utils.data.DataLoader(
        RestList('NTIRE', transforms.Compose(t_pair), transforms.Compose(v_unpair), out_name=True),
        batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)    
    loaders = Train_loader, Val_loader, Val_NTIRE_loader
        
    cudnn.benchmark = True

    # Define networks
    Net = torch.nn.DataParallel(iPASSRNet(scale)).cuda()
    # Net = iPASSRNet(scale).cuda()
    # Net.apply(weights_init_xavier)
    # model_D = torch.nn.DataParallel(Discriminator()).cuda()
    models = Net#, model_D
    
    checkpoint = torch.load('Pretrained.tar')
    # print(checkpoint.keys())
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint['Net'].items():
        if k[0:6]=='module':
            name = k[0:] # remove `module.`
            # print('module....')
        else:
            name = k[0:]
        new_state_dict[name] = v
    Net.load_state_dict(new_state_dict)

    # Define loss functions
    criterion_GAN  = GANLoss().cuda()
    criterion_eval = Evaluation().cuda()
    criterions = criterion_GAN, criterion_eval

    # Define optimizers
    optimizer = torch.optim.Adam([paras for paras in Net.parameters() if paras.requires_grad == True], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.n_steps, gamma=args.gamma)
    # optim_D = torch.optim.Adam(model_D.parameters(), lr=lr, betas=(0, 0.99))
    optims = optimizer#, optim_D

    for epoch in range(args.epochs): # train and validation
        logger.info('Epoch : [{0}]'.format(epoch))
        
        val_psnr, val_ssim, val_lpips = Train(loaders, models, optims, criterions, epoch, scale, best_score, output_dir=saveDirName+'/val', eval_score=psnr, logger=logger)
        scheduler.step()
        
        ## save the neural network
        if best_score < val_psnr :
            best_score = val_psnr
            history_path = saveDirName + '/' + 'ckpt{:03d}_'.format(epoch + 1) + 'p_' + str(val_psnr)[:6] + '_s_' + str(val_ssim)[7:13] + '_l_' + str(val_lpips)[7:13] + '_Better.pth.tar'
        else : 
            history_path = saveDirName + '/' + 'ckpt{:03d}_'.format(epoch + 1) + 'p_' + str(val_psnr)[:6] + '_s_' + str(val_ssim)[7:13] + '_l_' + str(val_lpips)[7:13] + '.pth.tar'

        save_checkpoint({
            'epoch': epoch + 1,
            'Net': Net.state_dict(),
        }, True, filename=history_path)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)

def weights_init_Kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.kaiming_normal_(m.bias.data)

def weights_init_Kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.kaiming_uniform_(m.bias.data)