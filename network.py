import time
import datetime
import torch
import math
from torch import nn, optim
from torch.nn import functional as F
from modules import ConvLayer, ResBlk, CBAM
import numpy as np
from skimage import morphology
import core

class iPASSRNet(nn.Module):
    def __init__(self, upscale_factor):
        super(iPASSRNet, self).__init__()
        self.upscale_factor = upscale_factor
        self.init_feature = nn.Conv2d(3, 64, 3, 1, 1, bias=True)
        self.deep_feature = RDGfilter(G0=64, C=4, G=24, n_RDB=4)
        self.pam = PAM(64)
        self.fusion = nn.Sequential(
            RDB(G0=128, C=4, G=32),
            CALayer(128),
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=True))
        self.reconstruct = RDG(G0=64, C=4, G=24, n_RDB=4)
        self.upscale1 = nn.Sequential(
            nn.Conv2d(64, 64 * 4, 1, 1, 0, bias=True),
            nn.PixelShuffle(2))
        self.inter_RDG = RDG(G0=64, C=4, G=24, n_RDB=4)
        self.upscale2 = nn.Sequential(
            nn.Conv2d(64, 64 * 4, 1, 1, 0, bias=True),
            nn.PixelShuffle(2))
        self.fin_RDG= RDG(G0=64, C=4, G=24, n_RDB=4)
        self.fin_conv = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

    def forward(self, x_left, x_right, is_training):
        x_left_upscale = (core.imresize(x_left, scale = 4, antialiasing = True)).clamp(min=0, max=1)
        x_right_upscale = (core.imresize(x_right, scale = 4, antialiasing = True)).clamp(min=0, max=1)
        buffer_left = self.init_feature(x_left)
        buffer_right = self.init_feature(x_right)
        buffer_left, catfea_left = self.deep_feature(buffer_left)
        buffer_right, catfea_right = self.deep_feature(buffer_right)

        if is_training == 1:
            buffer_leftT, buffer_rightT, (M_right_to_left, M_left_to_right), (V_left, V_right)\
                = self.pam(buffer_left, buffer_right, catfea_left, catfea_right, is_training)
        if is_training == 0:
            buffer_leftT, buffer_rightT \
                = self.pam(buffer_left, buffer_right, catfea_left, catfea_right, is_training)

        buffer_leftF = self.fusion(torch.cat([buffer_left, buffer_leftT], dim=1))
        buffer_rightF = self.fusion(torch.cat([buffer_right, buffer_rightT], dim=1))
        buffer_leftF, _ = self.reconstruct(buffer_leftF)
        buffer_rightF, _ = self.reconstruct(buffer_rightF)
        out_left, _ = self.inter_RDG(self.upscale1(buffer_leftF))
        out_left, _ = self.fin_RDG(self.upscale2(out_left))
        out_left = self.fin_conv(out_left) + x_left_upscale
        out_right, _ = self.inter_RDG(self.upscale1(buffer_rightF))
        out_right, _ = self.fin_RDG(self.upscale2(out_right))
        out_right = self.fin_conv(out_right) + x_right_upscale

        if is_training == 1:
            return out_left, out_right, (M_right_to_left, M_left_to_right), (V_left, V_right)
        if is_training == 0:
            return out_left, out_right


class one_conv(nn.Module):
    def __init__(self, G0, G):
        super(one_conv, self).__init__()
        self.conv = nn.Conv2d(G0, G, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):
        output = self.relu(self.conv(x))
        return torch.cat((x, output), dim=1)


class RDB(nn.Module):
    def __init__(self, G0, C, G):
        super(RDB, self).__init__()
        convs = []
        for i in range(C):
            convs.append(one_conv(G0+i*G, G))
        self.conv = nn.Sequential(*convs)
        self.LFF = nn.Conv2d(G0+C*G, G0, kernel_size=1, stride=1, padding=0, bias=True)
    def forward(self, x):
        out = self.conv(x)
        lff = self.LFF(out)
        return lff + x


class RDG(nn.Module):
    def __init__(self, G0, C, G, n_RDB):
        super(RDG, self).__init__()
        self.n_RDB = n_RDB
        RDBs = []
        for i in range(n_RDB):
            RDBs.append(RDB(G0, C, G))
        self.RDB = nn.Sequential(*RDBs)
        self.conv = nn.Conv2d(G0*n_RDB, G0, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        buffer = x
        temp = []
        for i in range(self.n_RDB):
            buffer = self.RDB[i](buffer)
            temp.append(buffer)
        buffer_cat = torch.cat(temp, dim=1)
        out = self.conv(buffer_cat)
        return out, buffer_cat

class RDGfilter(nn.Module):
    def __init__(self, G0, C, G, n_RDB):
        super(RDGfilter, self).__init__()
        self.n_RDB = n_RDB
        RDBs = []
        for i in range(n_RDB):
            RDBs.append(RDB(G0, C, G))
        self.RDB = nn.Sequential(*RDBs)
        self.conv = nn.Conv2d(G0*n_RDB, G0, kernel_size=1, stride=1, padding=0, bias=True)
        # self.ca = CALayer(G0*n_RDB)
        self.filter = ResB(G0*n_RDB, 1)

    def forward(self, x):
        buffer = x
        temp = []
        for i in range(self.n_RDB):
            buffer = self.RDB[i](buffer)
            temp.append(buffer)
        buffer_cat = torch.cat(temp, dim=1)
        buffer_cat = self.filter(buffer_cat)
        out = self.conv(buffer_cat)
        return out, buffer_cat

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel//16, 1, padding=0, bias=True),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(channel//16, channel, 1, padding=0, bias=True),
                nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class ResB(nn.Module):
    def __init__(self, channels, group=4):
        super(ResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=group, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, groups=group, bias=True),
        )
    def __call__(self,x):
        out = self.body(x)
        return out + x


class PAM(nn.Module):
    def __init__(self, channels):
        super(PAM, self).__init__()
        self.bq = nn.Conv2d(4*channels, channels, 1, 1, 0, groups=4, bias=True)
        self.bs = nn.Conv2d(4*channels, channels, 1, 1, 0, groups=4, bias=True)
        self.softmax = nn.Softmax(-1)
        self.rb = ResB(4 * channels)
        self.bn = nn.BatchNorm2d(4 * channels)

    def __call__(self, x_left, x_right, catfea_left, catfea_right, is_training):
        b, c0, h0, w0 = x_left.shape
        Q = self.bq(self.rb(self.bn(catfea_left)))
        b, c, h, w = Q.shape
        Q = Q - torch.mean(Q, 3).unsqueeze(3).repeat(1, 1, 1, w)
        K = self.bs(self.rb(self.bn(catfea_right)))
        K = K - torch.mean(K, 3).unsqueeze(3).repeat(1, 1, 1, w)

        score = torch.bmm(Q.permute(0, 2, 3, 1).contiguous().view(-1, w, c),                    # (B*H) * Wl * C
                          K.permute(0, 2, 1, 3).contiguous().view(-1, c, w))                    # (B*H) * C * Wr
        M_right_to_left = self.softmax(score)                                                   # (B*H) * Wl * Wr
        M_left_to_right = self.softmax(score.permute(0, 2, 1))                                  # (B*H) * Wr * Wl

        M_right_to_left_relaxed = M_Relax(M_right_to_left, num_pixels=2)
        V_left = torch.bmm(M_right_to_left_relaxed.contiguous().view(-1, w).unsqueeze(1),
                           M_left_to_right.permute(0, 2, 1).contiguous().view(-1, w).unsqueeze(2)
                           ).detach().contiguous().view(b, 1, h, w)  # (B*H*Wr) * Wl * 1
        M_left_to_right_relaxed = M_Relax(M_left_to_right, num_pixels=2)
        V_right = torch.bmm(M_left_to_right_relaxed.contiguous().view(-1, w).unsqueeze(1),  # (B*H*Wl) * 1 * Wr
                            M_right_to_left.permute(0, 2, 1).contiguous().view(-1, w).unsqueeze(2)
                                  ).detach().contiguous().view(b, 1, h, w)   # (B*H*Wr) * Wl * 1

        V_left_tanh = torch.tanh(5 * V_left)
        V_right_tanh = torch.tanh(5 * V_right)

        x_leftT = torch.bmm(M_right_to_left, x_right.permute(0, 2, 3, 1).contiguous().view(-1, w0, c0)
                            ).contiguous().view(b, h0, w0, c0).permute(0, 3, 1, 2)                           #  B, C0, H0, W0
        x_rightT = torch.bmm(M_left_to_right, x_left.permute(0, 2, 3, 1).contiguous().view(-1, w0, c0)
                            ).contiguous().view(b, h0, w0, c0).permute(0, 3, 1, 2)                              #  B, C0, H0, W0
        out_left = x_left * (1 - V_left_tanh.repeat(1, c0, 1, 1)) + x_leftT * V_left_tanh.repeat(1, c0, 1, 1)
        out_right = x_right * (1 - V_right_tanh.repeat(1, c0, 1, 1)) +  x_rightT * V_right_tanh.repeat(1, c0, 1, 1)

        if is_training == 1:
            return out_left, out_right, \
                   (M_right_to_left.contiguous().view(b, h, w, w), M_left_to_right.contiguous().view(b, h, w, w)),\
                   (V_left_tanh, V_right_tanh)
        if is_training == 0:
            return out_left, out_right


def M_Relax(M, num_pixels):
    _, u, v = M.shape
    M_list = []
    M_list.append(M.unsqueeze(1))
    for i in range(num_pixels):
        pad = nn.ZeroPad2d(padding=(0, 0, i+1, 0))
        pad_M = pad(M[:, :-1-i, :])
        M_list.append(pad_M.unsqueeze(1))
    for i in range(num_pixels):
        pad = nn.ZeroPad2d(padding=(0, 0, 0, i+1))
        pad_M = pad(M[:, i+1:, :])
        M_list.append(pad_M.unsqueeze(1))
    M_relaxed = torch.sum(torch.cat(M_list, 1), dim=1)
    return M_relaxed
        
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvLayer(3, 64, kernel_size=3, stride=2)
        self.conv2 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.conv3 = ConvLayer(128, 256, kernel_size=3, stride=2)
        self.conv4 = ConvLayer(256, 512, kernel_size=3, stride=2)

        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        self.conv1x1 = nn.Conv2d(512, 1, kernel_size=1, stride = 1, padding=0)

    def forward(self, x):        
        feat1 = F.leaky_relu(self.conv1(x), negative_slope=0.2, inplace=True)
        feat2 = F.leaky_relu(self.bn2(self.conv2(feat1)), negative_slope=0.2, inplace=True)
        feat3 = F.leaky_relu(self.bn3(self.conv3(feat2)), negative_slope=0.2, inplace=True)
        feat4 = F.leaky_relu(self.bn4(self.conv4(feat3)), negative_slope=0.2, inplace=True)
        prob = self.conv1x1(feat4)
        return prob