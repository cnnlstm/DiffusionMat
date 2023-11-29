# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from networks_swtu.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
from networks_swtu.swin_transformer_unet_skip_expand_decoder_sys_tem import SwinTransformerSysTem


logger = logging.getLogger(__name__)

class SwinUnetTem(nn.Module):
    def __init__(self, zero_head=False):
        super(SwinUnetTem, self).__init__()
        #self.num_classes = num_classes
        self.zero_head = zero_head
        # self.config = config

        # self.swin_unet = SwinTransformerSys(img_size=config.DATA.IMG_SIZE,
        #                         patch_size=config.MODEL.SWIN.PATCH_SIZE,
        #                         in_chans=config.MODEL.SWIN.IN_CHANS,
        #                         num_classes=self.num_classes,
        #                         embed_dim=config.MODEL.SWIN.EMBED_DIM,
        #                         depths=config.MODEL.SWIN.DEPTHS,
        #                         num_heads=config.MODEL.SWIN.NUM_HEADS,
        #                         window_size=config.MODEL.SWIN.WINDOW_SIZE,
        #                         mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
        #                         qkv_bias=config.MODEL.SWIN.QKV_BIAS,
        #                         qk_scale=config.MODEL.SWIN.QK_SCALE,
        #                         drop_rate=config.MODEL.DROP_RATE,
        #                         drop_path_rate=config.MODEL.DROP_PATH_RATE,
        #                         ape=config.MODEL.SWIN.APE,
        #                         patch_norm=config.MODEL.SWIN.PATCH_NORM,
        #                         use_checkpoint=config.TRAIN.USE_CHECKPOINT)
        # tdim = ch * 4
        # self.time_embedding = TimeEmbedding(T, ch, tdim)

        self.swin_unet = SwinTransformerSysTem(T=1000,ch=64,
                                patch_size=4,
                                in_chans=4,
                                num_classes=1,
                                embed_dim=96,
                                depths=[2, 2, 6, 2],
                                num_heads=[3, 6, 12, 24],
                                window_size=7,
                                mlp_ratio=4.,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.1,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False)


    def forward(self, x, t):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.swin_unet(x, t)
        return logits


class SwinUnet(nn.Module):
    def __init__(self, zero_head=False):
        super(SwinUnet, self).__init__()
        #self.num_classes = num_classes
        self.zero_head = zero_head
        

        self.swin_unet = SwinTransformerSys(#T=1000,ch=64,
                                patch_size=4,
                                in_chans=3,
                                num_classes=32,
                                embed_dim=96,
                                depths=[2, 2, 6, 2],
                                num_heads=[3, 6, 12, 24],
                                window_size=7,
                                mlp_ratio=4.,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.1,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False)


    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.swin_unet(x)
        return logits


if __name__ == '__main__':

    img = torch.ones([2, 4, 256, 512]).cuda()
    

    unet = SwinUnetTem().cuda()

    out = unet(img,100)

    print('Done')
