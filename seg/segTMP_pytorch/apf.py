import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline
import sys

class Apf:
    def __init__(self, q, weights):
        # weights    =   [target, tymp, malleus, umbo]
        self.weights = weights

        if q is not None:
            self.q = q
        else:
            q = []

        return


    def getReps(self, masks):
        for i in range(masks.shape[-1]):
            m = masks[:,:,i]            



            for i in range(self.q[0]-1, self.q[0]+1):
                for j in range(self.q[1]-1, self.q[1]+1):
                    d = self.findClosestPoint(m, self.q)
                    img[i,j] += k * np.sqrt((j - loc[1])**2 + (i - loc[0])**2)



        return

    def findClosestPoint(self, img, pt):
        d_min = sys.maxsize
        for i in range(img.shape[0]):            
            for j in range(img.shape[1]):
                d = np.sqrt((j - pt[1])**2 + (i - pt[0])**2)

                if img[i][j] not 0 and d < d_min:
                    d_min = d

        return d_min

    def getAtt(self, target):
        return

    def getPot(self, masks, target):
        return




if __name__ == "__main__":

