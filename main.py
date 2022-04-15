import os
import sys
import numpy as np
from utils import get_parser, init_network, train_net
# import torch 
# import torch.nn as nn
# from model.utils import get_model
# from training.dataset.utils import get_dataset
# from torch.utils import data
# from torch.utils.tensorboard import SummaryWriter
# from training.utils import update_ema_variables, exp_lr_scheduler_with_warmup, log_evaluation_result, get_optimizer
# from training.losses import DiceLoss
# from training.validation import validation
# import argparse
# import yaml
# import time

if __name__ == '__main__':

    args = get_parser()
    # print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    args.log_path = args.log_path + '%s/' % args.dataset

    Dice_list = []
    HD_list = []
    ASD_list = []

    for i in range(args.k_fold):
        net, ema_net = init_network(args)

        print(net)

        net.cuda()

        if args.ema:
            ema_net.cuda()
        # print("--*27")
        print("--"*27 + "Training of Fold: {}".format(i + 1) +"--"*27)
        best_Dice, best_HD, best_ASD = train_net(net, args, ema_net, fold_idx=i)

        Dice_list.append(best_Dice)
        HD_list.append(best_HD)
        ASD_list.append(best_ASD)
    
    if not os.path.exists('./exp/exp_%s' % args.dataset):
        os.mkdir('./exp/exp_%s' % args.dataset)

    # print(args.k_fold)
    
    with open('./exp/exp_%s/%s.txt' % (args.dataset, args.unique_name), 'w') as f:
        f.write('Dice       HD      ASD\n')
        for i in range(args.k_fold):
            f.write(str(Dice_list[i]) + str(HD_list[i]) + str(ASD_list[i]) + '\n')
        
        total_Dice = np.vstack(Dice_list)
        total_HD = np.vstack(HD_list)
        total_ASD = np.vstack(ASD_list)

        f.write('avg Dice: ' + str(np.mean(total_Dice, axis=0)) + ' std Dice: ' + str(np.std(total_Dice, axis=0)) + ' mean: ' + str(total_Dice.mean()) + ' std: ' + str(np.mean(total_Dice, axis=1).std()) + '\n')
        f.write('avg HD: ' + str(np.mean(total_HD, axis=0)) + ' std HD: ' + str(np.std(total_HD, axis=0)) + ' mean: ' + str(total_HD.mean()) + ' std: ' + str(np.mean(total_HD, axis=1).std()) + '\n')
        f.write('avg ASD: ' + str(np.mean(total_ASD, axis=0)) + ' std ASD: ' + str(np.std(total_ASD, axis=0)) + ' mean: ' + str(total_ASD.mean()) + ' std: '+ str(np.mean(total_ASD, axis=1).std()) + '\n')    
    
    print('Done!')

    sys.exit(0)