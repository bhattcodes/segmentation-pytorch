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
print(model)
















global rows, cols


def morphImage(img, n, diam):
    img_org = img
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(diam,diam))
    
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    return img, img_org


def loadMask(img_file):
    global rows, cols
    # img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    img = img_file
    rows,cols = img.shape[:2]


    Umbo = img[:,:,0]
    Malleus = img[:,:,1] 
    Tymp = img[:,:,2]    

    # cv2.imshow("Umbo", cv2.cvtColor(Umbo, cv2.COLOR_RGB2BGR))
    # cv2.imshow("Malleus", cv2.cvtColor(Malleus, cv2.COLOR_RGB2BGR))
    # cv2.imshow("Tymp", cv2.cvtColor(Tymp, cv2.COLOR_RGB2BGR))



    img = cv2.cvtColor(np.uint8(img*0.2), cv2.COLOR_BGR2RGB)    # Convert to BGR if using Matplotlib

    return img, Umbo, Malleus, Tymp


def getMalleusAndUmbo(img, Malleus):


    # Find Malleus centerline
    # ------------------------------------------

    contours,_ = cv2.findContours(Malleus,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
    cnt = cntsSorted[-1]


    [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)

    lefty = int((-x*vy/vx) + y)
    righty = int(((cols-x)*vy/vx)+y)
    ctrMalleus =(int(x), int(y))
    
    slopeMalleus = vy/vx
    slopeUmbo = -1/slopeMalleus

    lineMask = np.zeros((rows,cols), np.uint8)
    lineMask = cv2.line(lineMask,(cols-1,righty),(0,lefty),255,2)


    # Find Umbo center
    # ------------------------------------------

    contours,_ = cv2.findContours(Umbo, 1, 2)
    cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
    cnt = cntsSorted[-1]

    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    ctrUmbo = (cx, cy)
    img = cv2.drawContours(img, cnt, -1, (255,0,0), 2)


    # Perpendicular line through Umbo center
    # ------------------------------------------

    lefty = int((cx*vx/vy) + cy)
    righty = int(((cols-cx)*(-vx)/vy)+cy)


    # Get lineMask to divide Tymp into quadrants
    # ------------------------------------------

    lineMask = cv2.line(lineMask,(cols-1,righty),(0,lefty),255,2)

    return ctrMalleus, ctrUmbo, slopeMalleus, slopeUmbo, lineMask


def getQuadrants(img, Tymp, lineMask):


    # Subtract lineMask from Tymp and find resulting quadrant contours
    # ------------------------------------------

    sub = Tymp - lineMask
    _,sub_thresh = cv2.threshold(sub,127,255,cv2.THRESH_BINARY)
    quadCnts,_ = cv2.findContours(sub_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cntsSorted = sorted(quadCnts, key=lambda x: cv2.contourArea(x))

    if len(quadCnts) < 4:
        return -1, -1

    quadCnts = cntsSorted[-4:]

    img = cv2.drawContours(img, quadCnts, -1, (255,255,255), 1)
    
    return img, quadCnts



def getTarget(img, quadCnts, ctrMalleus, ctrUmbo, slopeMalleus, slopeUmbo):

    
    # Draw principle axes of coordinate system
    # ------------------------------------------
    # dx = 50
    # ctrMalleusSlope = (ctrMalleus[0] + dx, ctrMalleus[1] + int(slopeMalleus * dx))
    # ctrUmboSlope = (ctrUmbo[0] + dx, ctrUmbo[1] + int(slopeUmbo * dx))        
    # img = cv2.line(img, ctrMalleus, ctrMalleusSlope, (255,0,0), 3)
    # img = cv2.line(img, ctrUmbo, ctrUmboSlope, (255,0,255), 3)


    # Calculate intersection between axes
    # ------------------------------------------
    # 1: Umbo
    # 2: Malleus

    m1 = slopeUmbo
    m2 = slopeMalleus

    b1 = ctrUmbo[1]
    b2 = ctrMalleus[1]

    x1 = ctrUmbo[0]
    x2 = ctrMalleus[0]

    x_intersect = (-m1*x1 + b1 - b2 + m2*x2)/(m2 - m1)
    y_intersect = slopeMalleus * (x_intersect - ctrMalleus[0]) + ctrMalleus[1]

    ctrInt = (int(x_intersect), int(y_intersect))
    
    # img = cv2.drawMarker(img, ctrInt, (0, 0, 255), cv2.MARKER_CROSS)


    # Calculate reference vector (Malleus -> Intersection) and global offset angle
    # ------------------------------------------

    offset = np.empty([2])
    offset[0] = ctrInt[0] -  ctrMalleus[0]
    offset[1] = ctrInt[1] -  ctrMalleus[1]

    offset_angle = np.arctan2(offset[1], offset[0])

    ctrPts = np.empty([4,2])
    angles = np.empty([4])
    

    for i, cnt in enumerate(quadCnts):        

        # Obtain quadrant centers
        # ------------------------------------------

        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        ctrPts[i, :] = np.array([cx, cy])
        # img = cv2.drawMarker(img, (cx, cy), (0, 127, 127), cv2.MARKER_STAR, 5)


        # Calculate angle of quadrant vector wrt. reference vector
        # ------------------------------------------        

        angles[i] = np.arctan2(cy - ctrInt[1], cx - ctrInt[0])
        angles[i] -= offset_angle


        # Map to +/- pi
        # ------------------------------------------   

        if angles[i] < -np.pi:
            angles[i] += angles[i] + 2*np.pi
        elif angles[i] > np.pi:
            angles[i] -= 2*np.pi
        else:
            pass

    # Find angle corresponding to quadrant I
    # ------------------------------------------      

    minVal = np.min(angles[angles >= 0])
    idx = np.where(angles == minVal)
    img = cv2.drawContours(img, quadCnts[int(idx[0])], -1, (0,i*50,0), 2)

    target = (int(ctrPts[int(idx[0]), 0]), int(ctrPts[int(idx[0]), 1]))
    img = cv2.drawMarker(img, target, (0, 255, 255), cv2.MARKER_CROSS, 7)


    return img, target, quadCnts[int(idx[0])]


























preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


vid = cv2.VideoCapture(0) 


if MODE == 'test_unlab':
    best_model = torch.load('./checkpoints/best_DeepLabV3Plus_mobilenet_v2_sigmoid_dataset277_P80.pth',map_location=torch.device(DEVICE))
        
    image_size = (320, 320)
    t = np.zeros([300,1])
    pt = np.zeros([300,2])

    i = 0
    init = True
    with torch.no_grad():
        while i < 300:
            ret, image1 = vid.read()
            #image = white_balance(image)


            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image1, (image_size[0] , image_size[1]), interpolation = cv2.INTER_LANCZOS4)
            img_org = image
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
            img, Umbo, Malleus, Tymp = loadMask(mask)
            # Umbo, _ = morphImage(Umbo, 5, 20)
            # Malleus, _ = morphImage(Malleus, 5, 20)
            # Tymp, _ = morphImage(Tymp, 5, 40)    


            cv2.imshow("Image", cv2.cvtColor(image1, cv2.COLOR_RGB2BGR))
            cv2.imshow("Mask", cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)


            try:
                ctrMalleus, ctrUmbo, slopeMalleus, slopeUmbo, lineMask = getMalleusAndUmbo(img, Malleus)
                cv2.imshow('Ctrs', Tymp - lineMask)

            except:
                continue

            img_mask, quadCnts = getQuadrants(img, Tymp, lineMask)

            if quadCnts == -1:
                print("Quads not found")
                continue

            try:
                img_target, target, cnt = getTarget(img_org, quadCnts, ctrMalleus, ctrUmbo, slopeMalleus, slopeUmbo)
                target_inf = target
                millis = int(round(time.time() * 1000))
                print(str(millis) + "___" + str(target) + "___" + str(i))
                t[i] = millis
                pt[i,0] = target[0]
                pt[i,1] = target[1]
                i+=1

                # cv2.imshow('Mask', pred_mask)
                cv2.imshow('Target', cv2.cvtColor(img_target, cv2.COLOR_RGB2BGR))
                if init:
                    # cv2.imwrite('Image0.png', cv2.cvtColor(image1, cv2.COLOR_RGB2BGR))
                    # cv2.imwrite('Mask0.png', cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)) 
                    # cv2.imwrite('Quads0.png', Tymp - lineMask)
                    # cv2.imwrite('Target0.png', cv2.cvtColor(img_target, cv2.COLOR_RGB2BGR)) 
                    init = False
                cv2.waitKey(1)

            except:
                print("No target found")

                continue

        # cv2.imwrite('Image.png', cv2.cvtColor(image1, cv2.COLOR_RGB2BGR))
        # cv2.imwrite('Mask.png', cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)) 
        # cv2.imwrite('Quads.png', Tymp - lineMask)
        # cv2.imwrite('Target.png', cv2.cvtColor(img_target, cv2.COLOR_RGB2BGR)) 
        # np.savetxt('tracking.csv', np.hstack([t, pt]), delimiter=",")
        # cv2.waitKey(0)
        # s = input("Put")
        # for i in range(30):
        #     ret, image1 = vid.read()
        
        # cv2.imwrite('Ref.png', image1)



