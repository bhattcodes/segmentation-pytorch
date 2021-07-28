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
import time

from torch.utils.data import DataLoader

from utilsSegSB import get_training_augmentation, visualize
from dataloaders.dataloaders import DatasetTMP_test,white_balance
from utilsSegSB import get_preprocessing, get_validation_augmentation

import roslibpy
from targetDetector import TargetDetector

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


DATA_DIR = '../../../../datasets/TMP/dataset277_P80/'
DATASET = 'TMP'
CLASSES = ['umbo' ,'malleus', 'tympanicmembrane']

MODE = args.mode

# https://github.com/qubvel/segmentation_models.pytorch
ARCH = args.architecture #'DeepLabV3' # 'DeepLabV3' #'FPN'
ENCODER = args.encoder_backbone 
ENCODER_WEIGHTS = 'imagenet'

ACTIVATION = 'sigmoid' # sigmoid or None for logits or 'softmax2d' for multicalss segmentation
DEVICE = 'cuda' #'cuda'

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
# print(model)




preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


vid = cv2.VideoCapture(2) 




print("Running")

if MODE == 'test_unlab':
    # best_model = torch.load('./checkpoints/best_DeepLabV3Plus_mobilenet_v2_sigmoid_dataset277_P80.pth',map_location=torch.device(DEVICE))
    # best_model = torch.load('./checkpoints/best_DeepLabV3Plus_mobilenet_v2_sigmoid_dataset277_P80_all80.pth',map_location=torch.device(DEVICE))
    best_model = torch.load('./checkpoints/best_DeepLabV3Plus_mobilenet_v2_sigmoid_dataset277+phantom_rev2.pth',map_location=torch.device(DEVICE))

    # Initialize ros publisher
    client = roslibpy.Ros(host='localhost', port=9090)
    client.run()
    pub = roslibpy.Topic(client, '/sr_vis_target', 'geometry_msgs/Point32')


    image_size = (320, 320)
    t = np.zeros([300,1])
    pt = np.zeros([300,2])

    i = 0
    init = True
    t_wc = 0

    with torch.no_grad():
        print("Loop")
        while client.is_connected:
            t0 = int(round(time.time() * 1000))
            ret, image1 = vid.read()
            # print(image1.shape)

            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image1, (image_size[0] , image_size[1]), interpolation = cv2.INTER_LANCZOS4)
            img_org = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
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
            mask = pr_mask * 255
            
            if init:
                td = TargetDetector(mask)
                init = False
            cv2.imshow("Original", img_org)
            cv2.imshow("Segmentation",td.overlayMask(img_org, cv2.cvtColor(mask, cv2.COLOR_RGB2BGR), 0.6))



            cv2.waitKey(1)
            try:
                err, target = td.processAll(mask)
                # td.loadMask(mask)
                # td.getMalleusAndUmbo()
                # td.getQuadrants()
                # cv2.imshow("Axes", td.overlayMask(img_org, cv2.cvtColor(td.maskUmbo, cv2.COLOR_RGB2BGR), 0.5))

                # cv2.imshow("Axes", td.overlayMask(td.overlayMask(img_org, cv2.cvtColor(mask, cv2.COLOR_RGB2BGR), 0.6), cv2.cvtColor(td.maskLine, cv2.COLOR_RGB2BGR), 0.5))
                # img = cv2.drawContours(img_org, td.quadCnts, -1, (255,255,255), 3)
                # cv2.imshow("Quads", img)
                # cv2.waitKey(1)
                if err is None:
                    img2 = cv2.drawMarker(img_org, target, (125, 0, 125), cv2.MARKER_CROSS, 25, 3)
                    img2 = cv2.drawMarker(img2, (int(td.rows/2), int(td.cols/2)), (255, 255, 255), cv2.MARKER_CROSS, 500, 1)
                    img2 = cv2.line(img2,(int(td.rows/2), int(td.cols/2)),target,255,1)
                    cv2.imshow("Target", img2)

                    # cv2.imshow("Image line mask",td.overlayMask(cv2.drawMarker(img_org, target, (0, 0, 255), cv2.MARKER_CROSS, 25, 3), cv2.cvtColor(td.maskLine, cv2.COLOR_GRAY2RGB), 0.6))                
                    pub.publish(roslibpy.Message({'x': int(td.cols/2) - int(target[0]),'y': int(td.rows/2) - int(target[1])}))
                    cv2.waitKey(1)                
                else:
                    pub.publish(roslibpy.Message({'x': -1, 'y': -1}))
                    continue
            except:
                print("broke")
                pass

            

            # dt = int(round(time.time() * 1000)) - t0
            # if dt > t_wc:
            #     t_wc = dt
            # while  dt < t_wc:
            #     dt = int(round(time.time() * 1000)) - t0





pub.unadvertise()

client.terminate()



