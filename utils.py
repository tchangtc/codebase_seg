import argparse
import time
from tqdm import tqdm
import yaml
import os
import torch
import torch.nn as nn
import numpy as np
from model.utils import get_model
from training.dataset.utils import get_dataset
from training.validation import validation
from training.utils import update_ema_variables, exp_lr_scheduler_with_warmup, log_evaluation_result, get_optimizer
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from training.losses import DiceLoss


def get_parser():

    parser = argparse.ArgumentParser(description='2D & 3D Trans-Conv Segmentation in Medical with PyTorch')
    
    parser.add_argument('--dataset', type=str, default='knee', help='dataset name')
    
    parser.add_argument('--model', type=str, default='unet', help='model name')

    parser.add_argument('--dimension', type=str, default='3d', help='2d or 3d model')

    parser.add_argument('--pretrain', action='store_true', help='if use pretrained weight for init')

    parser.add_argument('--batch_size', default=2, type=int, help='batch size')

    parser.add_argument('--load', type=str, default=False, help='load pretrained model')

    parser.add_argument('--cp_path', type=str, default='./checkpoint/', help='checkpoint path')

    parser.add_argument('--log_path', type=str, default='./log/', help='log path')

    parser.add_argument('--unique_name', type=str, default='test_demo', help='unique experiment name')

    parser.add_argument('--gpu', type=str, default='0')

    parser.add_argument('--resume', type=str, default=False, help='continue interrupted trained or not')

    parser.add_argument('--ckpt', type=str, default='./continue/models/checkpoint/ckpt_1_1.pth', help='recent saved .pth')

    args = parser.parse_args()

    config_path = 'config/%s/%s_%s.yaml' % (args.dataset, args.model, args.dimension)

    if not os.path.exists(config_path):
        raise ValueError("The Specified Configuration doesn't Exist: %s" % config_path)
    
    print('Loading Configurations from %s Successfully!' % config_path)

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    
    for key, value in config.items():
        setattr(args, key, value)
    
    return args


def init_network(args):

    net = get_model(args, pretrain=args.pretrain)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model Loaded from {}'.format(args.load))

    if args.ema:
        ema_net = get_model(args, pretrain=args.pretrain)
        for p in ema_net.parameters():
            p.requires_grad_(False)
    
    else:
        ema_net = None
    
    return net, ema_net


def train_net(net, args, ema_net=None, fold_idx=0):
    data_path = args.data_root

    trainset = get_dataset(args, mode='train', fold_idx=fold_idx)
    trainLoader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    testset = get_dataset(args, mode='test', fold_idx=fold_idx)
    testLoader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

    writer = SummaryWriter(args.log_path + args.unique_name + '_%d' % fold_idx)

    optimizer = get_optimizer(args, net)

    # criterion = nn.CrossEntropyLoss(weight=torch.tensor(args.weight).cuda())
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(args.weight).cuda())
    criterion_dl = DiceLoss()

    best_Dice = np.zeros(args.classes)
    best_HD = np.ones(args.classes) * 1000
    best_ASD = np.ones(args.classes) * 1000

    iter_count = 0

    start_epoch = 0

    if args.resume == True:
        path_checkpoint = args.ckpt  # 断点路径
        checkpoint = torch.load(path_checkpoint)  # 加载断点

        net.load_state_dict(checkpoint['net'])  # 加载模型可学习参数

        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch
    else:
        print('Not resume!')

    for epoch in tqdm(range(start_epoch, args.epochs)):

        print('Starting epoch {}/{}'.format(epoch + 1, args.epochs))
        epoch_loss = 0

        exp_scheduler = exp_lr_scheduler_with_warmup(optimizer, init_lr=args.base_lr, epoch=epoch, warmup_epoch=5, max_epoch=args.epochs)
        print('Current lr: ', exp_scheduler)

        # tic = time.time()
        iter_num_per_epoch = 0
        for i, (img, label) in enumerate(trainLoader, 0):
            '''
            # uncomment this for visualize the input images and labels for debug
            for idx in range(img.shape[0]):
                plt.subplot(1,2,1)
                plt.imshow(img[idx, 0, 40, :, :].numpy())
                plt.subplot(1,2,2)
                plt.imshow(label[idx, 0, 40, :, :].numpy())

                plt.show()
            '''
            img = img.cuda()
            label = label.cuda()

            net.train()

            optimizer.zero_grad()   # clean previous gratitude

            result = net(img)

            loss = 0

            # print("reslut.shape is {}".format(result.shape))
            # print("label.shape is {}".format(label.shape))

            if isinstance(result, tuple) or isinstance(result, list):
                for j in range(len(result)):
                    loss += args.weight[j] * (criterion(result[j], label.squeeze(1)) + criterion_dl(result[j], label))
            else:
                loss = criterion(result, label.squeeze(1)) + criterion_dl(result, label)
            
            loss.backward()     # back propagation, calculate the present gratitude
            optimizer.step()    # update net parameters according to the gratitude
            iter_count += 1

            if args.ema:
                update_ema_variables(net, ema_net, args.ema_alpha, iter_count)
            
            epoch_loss += loss.item()
            # batch_time = time.time() - tic
            # tic = time.time()
            # print('%d batch loss: %.5f, batch_time: %.5f' % (i, loss.item(), batch_time))

            if args.dimension == '3d':
                iter_num_per_epoch += 1
                if iter_num_per_epoch > args.iter_per_epoch:
                    break
        
        checkpoint = {
            "net": net.state_dict(),
            'optimizer':optimizer.state_dict(),
            "epoch": epoch
            }
        
        # checkpoint_for_pred = {
        #     "net": net,
        #     # 'optimizer':optimizer.state_dict(),
        #     # "epoch": epoch
        #     }
        
        if not os.path.isdir("./continue/models/checkpoint"):
            os.mkdir("./continue/models/checkpoint")
        torch.save(checkpoint, './continue/models/checkpoint/ckpt_%s_%s.pth' %(str(fold_idx+1),str(epoch+1)))
        torch.save(net, './continue/models/checkpoint/ckpt_%s_%s_for_pred.pth' %(str(fold_idx+1),str(epoch+1)))

        print('[epoch %d] epoch loss: %.5f' % (epoch + 1, epoch_loss/(i+1)))
        torch.cuda.empty_cache()

        writer.add_scalar('Train/Loss', epoch_loss/(i + 1), epoch + 1)
        writer.add_scalar('LR', exp_scheduler, epoch + 1)

        if not os.path.isdir('%s%s' % (args.cp_path, args.dataset)):
            os.mkdir('%s%s' % (args.cp_path, args.dataset))
        
        if not os.path.isdir('%s%s/%s/' % (args.cp_path, args.dataset, args.unique_name)):
            os.mkdir('%s%s/%s/' % (args.cp_path, args.dataset, args.unique_name))
        
        if args.ema:
            net_for_eval = ema_net
        else:
            net_for_eval = net
        
        if(epoch + 1) % args.val_frequency == 0:
            dice_list_test, ASD_list_test, HD_list_test = validation(net_for_eval, testLoader, args, fold_idx)
            log_evaluation_result(writer, dice_list_test, ASD_list_test, HD_list_test, 'test', epoch, args)

            if dice_list_test.mean() >= best_Dice.mean():
                best_Dice = dice_list_test
                best_ASD = ASD_list_test
                best_HD = HD_list_test

                torch.save(net_for_eval.state_dict(), '%s%s/%s/%d_best.pth' % (args.cp_path, args.dataset, args.unique_name, fold_idx))
                torch.save(net_for_eval.state_dict(), '%s%s/%s/%d_for_pred_best.pth' % (args.cp_path, args.dataset, args.unique_name, fold_idx))
            print('Save Done!')
            print('dice: %.5f/best dice %.5f' % (dice_list_test.mean(), best_Dice.mean()))
    
    return best_Dice, best_HD, best_ASD