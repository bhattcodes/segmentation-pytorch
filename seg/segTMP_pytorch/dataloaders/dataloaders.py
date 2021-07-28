#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 14:10:36 2020

@author: sophiabano
"""
from torch.utils.data import Dataset as BaseDataset
import numpy as np
import os
import cv2 
import random
import matplotlib.pyplot as plt

##############################################################################  

def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

##############################################################################    
# Data loader 
class DatasetTMP_test(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    CLASSES = ['umbo' ,'malleus', 'tympanicmembrane']
    
    def __init__(
            self, 
            images_dir, 
            # masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        #self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        
        # read data
        image = cv2.imread(self.images_fps[i])
        #image = white_balance(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        ids = self.ids[i]
        
        #mask_str = self.masks_fps[i]
        
        #mask_str = mask_str.replace('.png','_L.png')
        #mask = cv2.imread(mask_str)
        #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        #mask = (mask/255).astype(np.uint8)
        
        mask = np.random.randint(1, size=(image.shape[0], image.shape[1], image.shape[2]), dtype=np.uint8)

        if np.shape(image)[0] == 500 and np.shape(image)[1] == 500:
            image_size = (320, 320)
            image = cv2.resize(image, (image_size[0] , image_size[1]), interpolation = cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (image_size[0] , image_size[1]), interpolation = cv2.INTER_NEAREST)
        elif np.shape(image)[0] == 480 and np.shape(image)[1] == 640:
            image_size = (416, 320)
            image = cv2.resize(image, (image_size[0] , image_size[1]), interpolation = cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (image_size[0] , image_size[1]), interpolation = cv2.INTER_NEAREST)
        elif np.shape(image)[0] == 200 and np.shape(image)[1] == 200:
           image_size = (320, 320)
           image = cv2.resize(image, (image_size[0] , image_size[1]), interpolation = cv2.INTER_LINEAR)
           mask = cv2.resize(mask, (image_size[0] , image_size[1]), interpolation = cv2.INTER_NEAREST)
            
        #if np.shape(image)[0] == 500 and np.shape(image)[1] == 500:
        #    image_size = (320, 320)
        #    image = cv2.resize(image, (image_size[0] , image_size[1]), interpolation = cv2.INTER_AREA)
        #    mask = cv2.resize(mask, (image_size[0] , image_size[1]), interpolation = cv2.INTER_NEAREST)
        #elif np.shape(image)[0] == 480 and np.shape(image)[1] == 640:
        #    image_size = (416, 320)
        #    image = cv2.resize(image, (image_size[0] , image_size[1]), interpolation = cv2.INTER_AREA)
        #    mask = cv2.resize(mask, (image_size[0] , image_size[1]), interpolation = cv2.INTER_NEAREST)
   
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask, ids
        
    def __len__(self):
        return len(self.ids)
    
##############################################################################    
# Data loader 
class DatasetTMP(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    CLASSES = ['umbo' ,'malleus', 'tympanicmembrane']
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        ip_img_siz = np.shape(image)
        if random.randint(0,1): 
            image = cv2.resize(image, (int(ip_img_siz[1]/2.5), int(ip_img_siz[0]/2.5)), interpolation = cv2.INTER_LINEAR)
            image = cv2.resize(image, (ip_img_siz[1], ip_img_siz[0]), interpolation = cv2.INTER_LINEAR)
        mask_str = self.masks_fps[i]
        
        #mask_str = mask_str.replace('.png','_L.png')
        mask = cv2.imread(mask_str)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
        
        if np.shape(image)[0] == 500 and np.shape(image)[1] == 500:
            image_size = (320, 320)
            image = cv2.resize(image, (image_size[0] , image_size[1]), interpolation = cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (image_size[0] , image_size[1]), interpolation = cv2.INTER_NEAREST)
        elif np.shape(image)[0] == 480 and np.shape(image)[1] == 640:
            image_size = (416, 320)
            image = cv2.resize(image, (image_size[0] , image_size[1]), interpolation = cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (image_size[0] , image_size[1]), interpolation = cv2.INTER_NEAREST)
        elif np.shape(image)[0] == 200 and np.shape(image)[1] == 200:
            image_size = (320, 320)
            image = cv2.resize(image, (image_size[0] , image_size[1]), interpolation = cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (image_size[0] , image_size[1]), interpolation = cv2.INTER_NEAREST)
   
        mask = (mask/255).astype(np.uint8)
        # extract certain classes from mask (e.g. cars)
        #masks = [(mask == v) for v in self.class_values]
        #mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)