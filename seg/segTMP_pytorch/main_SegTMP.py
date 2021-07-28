#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 13:05:05 2020

@author: sophiabano
"""

import os
import torch
import numpy as np
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import argparse
import cv2


from torch.utils.data import DataLoader

from utilsSegSB import get_training_augmentation, visualize
from dataloaders.dataloaders import DatasetTMP, DatasetTMP_test
from utilsSegSB import get_preprocessing, get_validation_augmentation

torch.manual_seed(0)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs to train for')
parser.add_argument('--mode', type=str, default="train", help='Select "train" or "test" or "test_unlab') 
parser.add_argument('--architecture', type=str, default="DeepLabV3Plus", help='FPN or DeepLabV3')
parser.add_argument('--encoder_backbone', type=str, default="mobilenet_v2", help='dpn68, resnest, se_resnext50_32x4d')
#parser.add_argument('--act_type', type=str, default="softmax2d", help='sigmoid or softmax2d')
#parser.add_argument('--loss_type', type=str, default="ce", help='dice or ce, dicetopk')

#parser.add_argument('--continue_training', type=str2bool, default=False, help='Whether to continue training from a checkpoint')

args = parser.parse_args()


DATA_DIR = '../../Data/dataset277+phantom_rev2/'
DATASET = 'TMP'
CLASSES = ['umbo' ,'malleus', 'tympanicmembrane']

MODE = args.mode
EPOCHS = args.num_epochs
n_epochs_stop = 50

# https://github.com/qubvel/segmentation_models.pytorch
ARCH = args.architecture #'DeepLabV3' # 'DeepLabV3' #'FPN'
ENCODER = args.encoder_backbone # 'se_resnext50_32x4d' #'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'

ACTIVATION = 'sigmoid' # sigmoid or None for logits or 'softmax2d' for multicalss segmentation
DEVICE = 'cpu'

# For CamVid dataset
if DATASET == 'TMP':
    CLASSES = ['umbo' ,'malleus', 'tympanicmembrane']
    
    x_train_dir = os.path.join(DATA_DIR, 'train/images')
    y_train_dir = os.path.join(DATA_DIR, 'train/masks')
    
    x_valid_dir = os.path.join(DATA_DIR, 'val/images')
    y_valid_dir = os.path.join(DATA_DIR, 'val/masks')
    
    x_test_dir = os.path.join(DATA_DIR, 'test/images')
    y_test_dir = os.path.join(DATA_DIR, 'test/masks')


"""
#### Visualize resulted augmented images and masks
augmented_dataset = DatasetTMP(
    x_train_dir, 
    y_train_dir, 
    augmentation=get_training_augmentation(), 
    classes=CLASSES,
)

valid_dataset = DatasetTMP(
    x_valid_dir, 
    y_valid_dir, 
    augmentation=get_validation_augmentation(), 
    classes=CLASSES,
)

# same image with different random transforms
for i in range(10):
   image, mask = valid_dataset[i]
   visualize(image=image, mask=mask*255)

"""

# create segmentation model with pretrained encoder
if ARCH == 'FPN':
    model = smp.FPN(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=len(CLASSES), 
        activation=ACTIVATION,
    )
elif ARCH == 'DeepLabV3':
    model = smp.DeepLabV3(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=len(CLASSES), 
        activation=ACTIVATION,
    )
elif ARCH == 'DeepLabV3Plus':
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=len(CLASSES), 
        activation=ACTIVATION,
    )
print(model)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

# IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index
# loss = smp.utils.losses.DiceLoss()
loss = smp.utils.losses.BCELoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]


if MODE == 'train':

    # For TMP dataset
    train_dataset = DatasetTMP(
        x_train_dir, 
        y_train_dir, 
        augmentation=get_training_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )
    
    valid_dataset = DatasetTMP(
        x_valid_dir, 
        y_valid_dir, 
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )
    
    """
    # check training and validation images and masks
    for i in range(3):
        image, mask = train_dataset[1]
        image = np.transpose(image,[1,2,0])
        visualize(image=image, mask=np.squeeze(mask))
        
    for i in range(3):
        image, mask = valid_dataset[1]
        image = np.transpose(image,[1,2,0])
        visualize(image=image, mask=np.squeeze(mask))
     """   

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=12)
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=4)
        
    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=0.0001),
    ])
    
    # create epoch runners 
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )
    
    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )
    
    # train model for 'EPOCHS'
    max_score = 0
    
    for i in range(0, EPOCHS):
        
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        
        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, './checkpoints/best_'+ARCH+'_'+ENCODER + '_' + ACTIVATION +'_dataset277.pth')
            print('Model saved!')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        # Check early stopping condition
        if epochs_no_improve == n_epochs_stop:
            print('Early stopping!' )
            early_stop = True
            break
        else:
            continue
        break
        if early_stop:
            print("Stopped")
            break
            
        if i == 50:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')
               
if MODE == 'test':
    # load best saved checkpoint
    # best_model = torch.load('./checkpoints/best_'+ ARCH + '_'+ENCODER + '_' + ACTIVATION +'.pth')
   
    best_model = torch.load('./checkpoints/best_DeepLabV3Plus_mobilenet_v2_sigmoid_dataset277_P80_PM60.pth')

    test_dataset = DatasetTMP(
        x_test_dir, 
        y_test_dir, 
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )
    
    # test dataset without transformations for image visualization
    test_dataset_vis = DatasetTMP(
        x_test_dir, y_test_dir, 
        classes=CLASSES,
    )
    
    test_dataloader = DataLoader(test_dataset)
    
    # evaluate model on test set
    test_epoch = smp.utils.train.ValidEpoch(
        model=best_model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
    )
    
    logs = test_epoch.run(test_dataloader)
    
    num_images = os.listdir(x_test_dir) 
    for i in range(len(num_images)):
        #n = np.random.choice(len(test_dataset))
        
        image_vis = test_dataset_vis[i][0].astype('uint8')
        image, gt_mask = test_dataset[i]
        
        gt_mask = gt_mask.squeeze()
        
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        #pr_mask = (pr_mask.squeeze().cpu().numpy())      
        
        
        gt_mask= (np.transpose(gt_mask,[1,2,0])).astype(np.uint8)
        pr_mask= (np.transpose(pr_mask,[1,2,0])).astype(np.uint8) 
        
        
        visualize(
            image=image_vis, 
            ground_truth_mask=gt_mask*255, 
            predicted_mask=pr_mask*255
        )
        
if MODE == 'test_unlab':
    best_model = torch.load('./checkpoints/best_DeepLabV3Plus_mobilenet_v2_sigmoid_dataset277_P80_PM60.pth')
        
    x_test_dir = '../../../datasets/TMP/dataset277_P80_PM60/Case3_1_seq_unlab/images'
    output_path = '../../../datasets/TMP/dataset277_P80_PM60/Case3_1_seq_unlab/output'
    #x_test_dir = input_path

    # create test dataset
    test_dataset = DatasetTMP_test(
        x_test_dir, 
        #y_test_dir, 
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )
    
    # test dataset without transformations for image visualization
    test_dataset_vis = DatasetTMP_test(
        x_test_dir, 
        #y_test_dir, 
        classes=CLASSES,
    )   

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(output_path +' created')
    else:
        print(output_path +' exists')
        
    """
    # For testing using dataloader    
    test_dataloader = DataLoader(test_dataset)
    
    # evaluate model on test set
    test_epoch = smp.utils.train.ValidEpoch(
        model=best_model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
    )
    
    #logs = test_epoch.run(test_dataloader)
    """
    with torch.no_grad():
        num_images = os.listdir(x_test_dir) 
        for i in range(len(num_images)):
            pr_mask = []
            #n = np.random.choice(len(test_dataset))
            
            image_vis = test_dataset_vis[i][0].astype('uint8')
            image, gt_mask, ids = test_dataset[i]
            
            # gt_mask = gt_mask.squeeze()
            
            x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
            pr_mask1 = best_model.predict(x_tensor)
            #pr_mask = (pr_mask.squeeze().cpu().numpy().round())
            pr_mask = pr_mask1.squeeze().cpu().numpy()

            pr_mask= (np.transpose(pr_mask,[1,2,0]))#.astype(np.uint8)             
            
            # gt_mask= (np.transpose(gt_mask,[1,2,0])).astype(np.uint8)
            
            pr_maskR = pr_mask[:,:,0]
            pr_maskR[pr_maskR > 0.5] = 1
            pr_maskR[pr_maskR <= 0.5] = 0

            pr_maskG = pr_mask[:,:,1]
            pr_maskG[pr_maskG > 0.5] = 1
            pr_maskG[pr_maskG <= 0.5] = 0
            
            pr_maskB = pr_mask[:,:,2]
            pr_maskB[pr_maskB > 0.5] = 1
            pr_maskB[pr_maskB <= 0.5] = 0
            
            pr_mask = (np.concatenate((np.expand_dims(pr_maskR, axis =2),np.expand_dims(pr_maskG, axis = 2),np.expand_dims(pr_maskB, axis = 2)),axis=2)).astype(np.uint8) 
            

            visualize(
                image=image_vis, 
                #ground_truth_mask=gt_mask*255, 
                predicted_mask=pr_mask*255
            )
            plt.savefig(output_path +'/'+ids)
            plt.show()
            
            pr_mask = cv2.cvtColor(pr_mask, cv2.COLOR_RGB2BGR)
            result=cv2.imwrite(output_path +'/'+ids , pr_mask*255)
            #result=cv2.imwrite(output_path +'/'+ids , pr_mask*255)
            # if result==True:
            #    print(output_path+'/' +ids +' output mask saved')
            #else:
            #    print('Error in saving file')