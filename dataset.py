import os
import numpy as np
import random

from PIL import Image
from glob import glob
#import rawpy
import data_transforms as transforms
from torchvision import transforms as torch_transforms
import torch


class RestList(torch.utils.data.Dataset):
    def __init__(self, phase, t_pair, t_unpair, batch=1, out_name=False):
        self.phase = phase
        self.batch = batch

        self.t_pair = t_pair
        self.t_unpair = t_unpair
        self.out_name = out_name

        self.image_list_FFHQ = None
        self.image_list_Places = None
        self.image_list_ImgNet = None
        self.image_list_Wiki = None

        self.mask_list = None
        self.val_list = None
        self._make_list()
        if self.val_list is not None :
            self.num_img  = len(self.val_list)
            self.num_mask = len(self.mask_list)


    def __getitem__(self, index):
        # np.random.seed()
        # random.seed()
        # idx = i2s(index)
        if self.phase == 'Train':
            # index_ = np.random.randint(len(self.HR_Left_list))
            HR_Left  = Image.open(self.HR_Left_list[index]).convert('RGB')
            HR_Right = Image.open(self.HR_Right_list[index]).convert('RGB')
            LR_Left  = Image.open(self.LR_Left_list[index]).convert('RGB')
            LR_Right = Image.open(self.LR_Right_list[index]).convert('RGB')
            
            data = list(self.t_pair(*[HR_Left, HR_Right,LR_Left,LR_Right]))
        elif self.phase == 'NTIRE':
            name = (self.LR_Left_list[index]).split('/')
            name = name[-1]
            name = name[0:4]
            LR_Left  = Image.open(self.LR_Left_list[index]).convert('RGB')
            LR_Right = Image.open(self.LR_Right_list[index]).convert('RGB')
            # data = list(self.t_unpair(*[LR_Left,LR_Right, name]))
            data = list([self.t_unpair(*[LR_Left]), self.t_unpair(*[LR_Right]), name])

        else :
            HR_Left  = Image.open(self.HR_Left_list[index]).convert('RGB')
            HR_Right = Image.open(self.HR_Right_list[index]).convert('RGB')
            LR_Left  = Image.open(self.LR_Left_list[index]).convert('RGB')
            LR_Right = Image.open(self.LR_Right_list[index]).convert('RGB')

            name = (self.HR_Left_list[index]).split('/')
            name = name[-1]

            data = list(self.t_unpair(*[HR_Left]))
            data.append(self.t_unpair(*[HR_Right]))
            data.append(self.t_unpair(*[LR_Left]))
            data.append(self.t_unpair(*[LR_Right]))
            data.append(name)

        return tuple(data)


    def __len__(self):
        return len(self.LR_Left_list)

    def _make_list(self):
        if self.phase == 'Train':
            self.HR_Left_list = sorted(glob('../../01_Data/Train_patches/HR_Left/*.png'))
            print('Num of training HR left images : ' + str(len(self.HR_Left_list)))
            self.HR_Right_list = sorted(glob('../../01_Data/Train_patches/HR_Right/*.png'))
            print('Num of training HR right images : ' + str(len(self.HR_Right_list)))
            self.LR_Left_list = sorted(glob('../../01_Data/Train_patches/LR_Left/*.png'))
            print('Num of training LR left images : ' + str(len(self.LR_Left_list)))
            self.LR_Right_list = sorted(glob('../../01_Data/Train_patches/LR_Right/*.png'))
            print('Num of training LR right images : ' + str(len(self.LR_Right_list)))

        elif self.phase == 'Validation' :
            self.HR_Left_list = sorted(glob('../../01_Data/Validation/HR_Left/*.png'))
            print('Num of training HR left images : ' + str(len(self.HR_Left_list)))
            self.HR_Right_list = sorted(glob('../../01_Data/Validation/HR_Right/*.png'))
            print('Num of training HR right images : ' + str(len(self.HR_Right_list)))
            self.LR_Left_list = sorted(glob('../../01_Data/Validation/LR_Left/*.png'))
            print('Num of training LR left images : ' + str(len(self.LR_Left_list)))
            self.LR_Right_list = sorted(glob('../../01_Data/Validation/LR_Right/*.png'))
            print('Num of training LR right images : ' + str(len(self.LR_Right_list)))

        elif self.phase == 'NTIRE' :
            if os.path.exists('../../01_Data/Test'):
                root  = '../../01_Data/Test'
                print('test session start')
                self.LR_Left_list = sorted(glob(root + '/*_L.png'))
                print('Num of testing LR left images : ' + str(len(self.LR_Left_list)))
                self.LR_Right_list = sorted(glob(root + '/*_R.png'))
                print('Num of testing LR right images : ' + str(len(self.LR_Right_list)))
            else:
                root = '../../01_Data/Validation'
                self.LR_Left_list = sorted(glob(root + '/LR_Left/*.png'))
                print('Num of training LR left images : ' + str(len(self.LR_Left_list)))
                self.LR_Right_list = sorted(glob(root + '/LR_Right/*.png'))
                print('Num of training LR right images : ' + str(len(self.LR_Right_list)))

def i2s(n):
    temp = str(n)
    l = temp.__len__()
    st = []
    for i in range(6-l):
        st.append('0')
    st.append(temp)
    nt = ''.join(st)
    return nt