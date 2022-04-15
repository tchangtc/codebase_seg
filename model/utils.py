import numpy as np
import torch 
import torch.nn as nn

def get_model(args, pretrain=False):
    if args.dimension == '2d':
        if args.model == 'utnetv2':
            from .dim2.utnetv2 import UTNetV2
            if pretrain:
                raise ValueError('No pretrain model available')
            return UTNetV2(args.in_ch, args.classes, args.base_ch, conv_block=args.conv_block, conv_num=args.conv_num, trans_num=args.trans_num, num_heads=args.num_heads, fusion_depth=args.fusion_depth, fusion_dim=args.fusion_dim, fusion_heads=args.fusion_heads, map_size=args.map_size, proj_type=args.proj_type, act=args.act, norm=args.norm, expansion=args.expansion, attn_drop=args.attn_drop, proj_drop=args.proj_drop)


    elif args.dimension == '3d':
        if args.model == 'vnet':
            from .dim3.vnet import VNet
            if pretrain:
                raise ValueError('No pretrain model available')
            return VNet(args.in_ch, args.classes, scale=args.downsample_scale, base_ch=args.base_ch)
        
        elif args.model == 'unet':
            from .dim3.unet import UNet
            return UNet(args.in_ch, args.base_ch, num_classes=args.classes, scale=args.down_scale, norm=args.norm, kernel_size=args.kernel_size, block=args.block) 

    else:
        raise ValueError('Invalid dimension, should be \'2d\' or \'3d\'')
