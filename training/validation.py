import torch
from inference.utils import get_inference
import numpy as np
from metric.utils import calculate_dice, calculate_distance
import SimpleITK as sitk

data_dir = './predict-unet-ski-2classes_100/'

def validation(net, dataloader, args, fold_idx):
    net.eval()

    dice_list = np.zeros(args.classes - 1)
    ASD_list = np.zeros(args.classes - 1)
    HD_list = np.zeros(args.classes - 1)

    inference = get_inference(args)

    counter = 0
    with torch.no_grad():
        for i, (images, labels, spacing) in enumerate(dataloader):
            # Here, spacing is used for distance metrics calculation


            ##########################################################################################################################################
            """
            Original images and corresponding labels
            """ 
            # label_pred1 = label_pred.cpu().numpy()
            # label_pred1 = label_pred1.astype(np.uint8)
            images_origin = images.cpu().numpy().astype(np.float32)
            images_origin = sitk.GetImageFromArray(images_origin)
            save_img_origin = sitk.WriteImage(images_origin, data_dir + 'images/{}/{}.nii.gz'.format(fold_idx,i+1))

            labels_origin = labels.cpu().numpy().astype(np.uint8)
            labels_origin = sitk.GetImageFromArray(labels_origin)
            save_lab_origin = sitk.WriteImage(labels_origin, data_dir + 'labels/{}/{}_gt.nii.gz'.format(fold_idx,i+1))
            ##########################################################################################################################################
            
            # print("The shape of original image is {}".format(images.shape))

            # print("The shape of original label is {}".format(labels.shape))

            ##########################################################################################################################################

            inputs, labels = images.float().cuda(), labels.long().cuda()

            if args.dimension == '2d':
                inputs = inputs.permute(1, 0, 2, 3)
            
            pred = inference(net, inputs, args)
            
            _, label_pred = torch.max(pred, dim=1)

            if args.dimension == '2d':
                labels = labels.squeeze(0)
            else:
                label_pred = label_pred.squeeze(0)
                labels = labels.squeeze(0).squeeze(0)
            
            print("#" * 27 + "The {} example in TestLoader".format(i + 1) + "#" * 27)

            # print("The shape of image is {}".format(images.shape))
            # print("The shape of label is {}".format(labels.shape))
            # print("The shape of predct label is {}".format(label_pred.shape))

            #####################################################################
            # assert labels.shape == label_pred.shape
            # assert images.shape == labels.shape
            # print(labels.shape)
            # print(label_pred.shape)

            # img = img.astype(np.float32)
            # lab = lab.astype(np.uint8)
            
            # label_pred1 = label_pred.cpu().numpy()
            # label_pred1 = label_pred1.astype(np.uint8)
            # save = sitk.GetImageFromArray(label_pred1)
            # save = sitk.WriteImage(save, data_dir + '{}.nii.gz'.format(i))



            labels_pred = label_pred.cpu().numpy().astype(np.uint8)
            labels_pred = sitk.GetImageFromArray(labels_pred)
            save_lab_pred = sitk.WriteImage(labels_pred, data_dir + 'pred/labels/{}/{}_gt.nii.gz'.format(fold_idx,i+1))
            # labels_origin = labels.cpu().numpy().astype(np.uint8)
            # labels_origin = sitk.GetImageFromArray(labels_origin)
            # save_lab_origin = sitk.WriteImage(labels_origin, data_dir + 'labels/{}_gt.nii.gz'.format(i))
            
            #####################################################################

            tmp_ASD_list, tmp_HD_list = calculate_distance(label_pred, labels, spacing[0], args.classes)
            ASD_list += np.clip(np.nan_to_num(tmp_ASD_list, nan=500), 0, 500)
            HD_list += np.clip(np.nan_to_num(tmp_HD_list, nan=500), 0, 500)

            dice, _, _ = calculate_dice(label_pred.view(-1, 1), labels.view(-1, 1), args.classes)

            dice_list += dice.cpu().numpy()[1:]

            counter += 1
    
    dice_list /= counter
    ASD_list /= counter
    HD_list /= counter

    return dice_list, ASD_list, HD_list


