#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 16:57:01 2020

@author: sophiabano
"""
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
from dataloaders.dataloaders import DatasetTMP_test,white_balance
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
parser.add_argument('--mode', type=str, default="test_unlab", help='Select "train" or "test" or "test_unlab') 
parser.add_argument('--architecture', type=str, default="DeepLabV3Plus", help='FPN or DeepLabV3')
parser.add_argument('--encoder_backbone', type=str, default="mobilenet_v2", help='dpn68, resnest, se_resnext50_32x4d')

args = parser.parse_args()


DATA_DIR = '../../../datasets/TMP/dataset277_P80/'
DATASET = 'TMP'
CLASSES = ['umbo' ,'malleus', 'tympanicmembrane']

MODE = args.mode

# https://github.com/qubvel/segmentation_models.pytorch
ARCH = args.architecture #'DeepLabV3' # 'DeepLabV3' #'FPN'
ENCODER = args.encoder_backbone 
ENCODER_WEIGHTS = 'imagenet'

ACTIVATION = 'sigmoid' # sigmoid or None for logits or 'softmax2d' for multicalss segmentation
DEVICE = 'cpu' #'cuda'

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

if MODE == 'test_unlab':
    best_model = torch.load('./checkpoints/best_DeepLabV3Plus_mobilenet_v2_sigmoid_dataset277_P80_PM60.pth',map_location=torch.device(DEVICE))
        
    # Input sequence path
    x_test_dir = '../../../datasets/TMP/dataset277_P80_PM60/Case3_1_seq_unlab/images'
    # Output path where the inference results (mask) will be stored
    output_path = '../../../datasets/TMP/dataset277_P80_PM60/Case3_1_seq_unlab/output'
    image_size = (320, 320)

    # create output folder
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(output_path +' created')
    else:
        print(output_path +' exists')
        
    
    with torch.no_grad():
        num_images = np.sort(os.listdir(x_test_dir))
        for i in range(len(num_images)):

            image1 = cv2.imread(x_test_dir+'/'+num_images[i])
            #image = white_balance(image)
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image1, (image_size[0] , image_size[1]), interpolation = cv2.INTER_LANCZOS4)
            
            # preprocessing for preparing the input for the network
            image = preprocessing_fn(image)
            
            image = (np.transpose(image,[2,0,1])).astype(np.float32)  
            x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
            pr_mask1 = best_model.predict(x_tensor)
            pr_mask = pr_mask1.squeeze().cpu().numpy()

            pr_mask= (np.transpose(pr_mask,[1,2,0])) #.astype(np.uint8)             
            
            # Applying small threshold to malleus and umbo 
            pr_maskR = pr_mask[:,:,0]
            pr_maskR[pr_maskR > 0.3] = 1
            pr_maskR[pr_maskR <= 0.3] = 0

            pr_maskG = pr_mask[:,:,1]
            pr_maskG[pr_maskG > 0.3] = 1
            pr_maskG[pr_maskG <= 0.3] = 0
            
            # Applying athreshold of 0.5 to TYMP
            pr_maskB = pr_mask[:,:,2]
            pr_maskB[pr_maskB > 0.5] = 1
            pr_maskB[pr_maskB <= 0.5] = 0
            
            pr_mask = (np.concatenate((np.expand_dims(pr_maskR, axis =2),np.expand_dims(pr_maskG, axis = 2),np.expand_dims(pr_maskB, axis = 2)),axis=2)).astype(np.uint8)         

            visualize(
                image=image1, 
                #ground_truth_mask=gt_mask*255, 
                predicted_mask=pr_mask*255
            )
            plt.savefig(output_path +'/'+num_images[i])
            plt.show()
