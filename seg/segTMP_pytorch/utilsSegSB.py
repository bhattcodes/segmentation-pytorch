#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 13:53:19 2020

@author: sophiabano
"""
from torch.utils.data import Dataset as BaseDataset
from matplotlib import pyplot as plt
import albumentations as albu
import numpy as np
import os
import cv2

# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    #plt.show()
    
##############################################################################    
# data augmentation
# Since our dataset is very small we will apply a large number of different augmentations:
# horizontal flip
# affine transforms
# perspective transforms
# brightness/contrast/colors manipulations
# image bluring and sharpening
# gaussian noise
# random crops
def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=180, shift_limit=0.01, p=0.5, border_mode=0),

        albu.PadIfNeeded(min_height=192, min_width=192, always_apply=True, border_mode=0),
        albu.RandomCrop(height=192, width=192, always_apply=True),

        #albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),
        
        #albu.IAASuperpixels(p=0.2),
        albu.RGBShift(r_shift_limit=[0,60], g_shift_limit=[0,60], b_shift_limit=0, p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(limit=0.4,p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.5,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=7, p=1),
                albu.MotionBlur(blur_limit=7, p=1),
            ],
            p=0.5,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(limit=0.4,p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.5,
        ),

    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        #albu.PadIfNeeded(320, 320)
        #albu.Resize(320, 320)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)