import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline


class TargetDetector:
    def __init__(self, mask):

        self.rows,self.cols = mask.shape[:2]
        print("Rows " + str(self.rows) + " Cols " + str(self.cols))
        self.maskUmbo = np.zeros([self.rows, self.cols], np.uint8)
        self.maskMalleus = np.zeros([self.rows, self.cols], np.uint8)
        self.maskTymp = np.zeros([self.rows, self.cols], np.uint8)
        self.mask = np.zeros([self.rows, self.cols, 3], np.uint8)
        self.maskLine = np.zeros([self.rows, self.cols, 3], np.uint8)
        

        return


    def loadMask(self, mask):

        try:
            self.maskUmbo = mask[:,:,0]
            self.maskMalleus = mask[:,:,1] 
            self.maskTymp = mask[:,:,2]    

            self.mask = mask
        except:
            return "Error in mask assignment"
        return None

    def getMalleusAndUmbo(self):

        Malleus = self.maskMalleus
        Umbo = self.maskUmbo

        # Clear line mask
        self.maskLine = np.zeros((self.rows,self.cols), np.uint8)



        # # Find Tymp cnt
        # # ------------------------------------------
        # try:
        #     contours,_ = cv2.findContours(Tymp,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        #     cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
        #     self.cntTymp = cntsSorted[-1]
        # except:
        #     return "Tymp not found"


        # Find Malleus centerline
        # ------------------------------------------
        try:
            contours,_ = cv2.findContours(Malleus,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
            self.cntMalleus = cv2.convexHull(cntsSorted[-1])
        except:
            return "Malleus not found"
        # XXXXXXXXXXXXXXXX Add sanity check


        [vx,vy,x,y] = cv2.fitLine(self.cntMalleus, cv2.DIST_L2,0,0.01,0.01)

        try:
            lefty = int((-x*vy/vx) + y)
            righty = int(((self.cols-x)*vy/vx)+y)
        except:
            return "Division by 0"

        # if lefty >= self.rows or lefty < 0:
        #     print(lefty)
        #     return "Line Malleus out of scope"
        # if righty >= self.rows or righty < 0:
        #     print(rigthy)
        #     return "Line Malleus out of scope"

        self.ctrMalleus =(int(x), int(y))
        
        self.slopeMalleus = vy/vx
        self.slopeUmbo = -1.0/self.slopeMalleus

        self.maskLine = cv2.line(self.maskLine,(self.cols-1,righty),(0,lefty),255,2)


        # Find Umbo center
        # ------------------------------------------
        try:
            contours,_ = cv2.findContours(Umbo, 1, 2)
            cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
            self.cntUmbo = cntsSorted[-1]
        except:
            return "Umbo not found"

        
        M = cv2.moments(self.cntUmbo)

        if M['m00'] == 0:
            return "Division by 0"
        else:            
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

        self.ctrUmbo = (cx, cy)
        # img = cv2.drawContours(img, self.cntUmbo, -1, (255,0,0), 2)


        # Perpendicular line through Umbo center
        # ------------------------------------------
        try:
            lefty = int((cx*vx/vy) + cy)
            righty = int(((self.cols-cx)*(-vx)/vy)+cy)
        except:
            return "Division by 0"


        # if lefty >= self.rows or lefty < 0:
        #     return "Line out of scope"
        # if righty >= self.rows or righty < 0:
        #     return "Line out of scope"

        # Get lineMask to divide Tymp into quadrants
        # ------------------------------------------

        self.maskLine = cv2.line(self.maskLine,(self.cols-1,righty),(0,lefty),255,2)

        return None


    def getQuadrants(self):


        # Subtract lineMask from Tymp and find resulting quadrant contours
        # ------------------------------------------

        sub = self.maskTymp - self.maskLine


        _,sub_thresh = cv2.threshold(sub,127,255,cv2.THRESH_BINARY)
        quadCnts,_ = cv2.findContours(sub_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cntsSorted = sorted(quadCnts, key=lambda x: cv2.contourArea(x))

        if len(quadCnts) < 4:
            return "Less than 4 quadrants found"

        self.quadCnts = cntsSorted[-4:]

        # img = cv2.drawContours(img, quadCnts, -1, (255,255,255), 1)
        
        return None


    def getTarget(self):

        
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

        m1 = self.slopeUmbo
        m2 = self.slopeMalleus

        b1 = self.ctrUmbo[1]
        b2 = self.ctrMalleus[1]

        x1 = self.ctrUmbo[0]
        x2 = self.ctrMalleus[0]

        x_intersect = (-m1*x1 + b1 - b2 + m2*x2)/(m2 - m1)
        y_intersect = m2 * (x_intersect - self.ctrMalleus[0]) + self.ctrMalleus[1]

        ctrInt = (int(x_intersect), int(y_intersect))
        
        # img = cv2.drawMarker(img, ctrInt, (0, 0, 255), cv2.MARKER_CROSS)


        # Calculate reference vector (Malleus -> Intersection) and global offset angle
        # ------------------------------------------

        offset = np.empty([2])
        offset[0] = ctrInt[0] -  self.ctrMalleus[0]
        offset[1] = ctrInt[1] -  self.ctrMalleus[1]

        offset_angle = np.arctan2(offset[1], offset[0])

        ctrPts = np.empty([4,2])
        angles = np.empty([4])
        

        for i, cnt in enumerate(self.quadCnts):        

            # Obtain quadrant centers
            # ------------------------------------------

            M = cv2.moments(cnt)
            try:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
            except:
                return "Quads division through 0"

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
        try:
            minVal = np.min(angles[angles >= 0])
            idx = np.where(angles == minVal)
            target = (int(ctrPts[int(idx[0]), 0]), int(ctrPts[int(idx[0]), 1]))
            # target = (self.rows/2 - int(ctrPts[int(idx[0]), 0]), self.cols/2 -  int(ctrPts[int(idx[0]), 1]))
        except:
            return "No positive angles found"    
        # img = cv2.drawContours(img, quadCnts[int(idx[0])], -1, (0,i*50,0), 2)
        # img = cv2.drawMarker(img, target, (0, 255, 255), cv2.MARKER_CROSS, 7)

        self.cntTarget = self.quadCnts[int(idx[0])]
        self.target = target

        return None

    def overlayMask(self, img, mask, alpha):
        beta = (1.0 - alpha)
        return np.uint8(alpha*(img)+beta*(mask))


    def processAll(self, mask):
        if (s := self.loadMask(mask)) is not None:
            print(s)
            return 1, None
        if (s := self.getMalleusAndUmbo()) is not None:
            print(s)
            return 2, None
        if (s := self.getQuadrants()) is not None:
            print(s)
            return 3, None
        if (s := self.getTarget()) is not None:
            print(s)
            return 4, None

        return None, self.target



if __name__ == "__main__":
    
    i = 9
    # for i in range(1):
    filename = "L7_"+str(i+1) +".png"
    folder = "../../../datasets/test2/"

    print(filename)

    img = cv2.imread(folder + "/images/" + filename)
    mask = cv2.imread(folder + "/masks/" + filename)
    # b,g,r = cv2.split(mask)
    # mask = np.

    cv2.imshow("mask",mask)

    td = TargetDetector(img)
    err, target = td.processAll(mask)

    if err is None:
        img2 = cv2.drawMarker(img, target, (0, 0, 255), cv2.MARKER_CROSS, 25, 3)

        img2 = td.overlayMask(img2, td.mask, 0.9)
        cv2.imshow("target",img2)

        img3 = cv2.drawContours(img, td.cntMalleus, -1, (0,200,200), 5, 8)
        img3 = cv2.drawContours(img3, td.cntUmbo, -1, (200,0,200), 5, 8)
        # img3 = cv2.drawContours(img3, td.cntTymp, -1, (0,200,200), 5, 8)

        # print(td.cntMalles)
        cv2.imshow("cnts",img3)

        # x = []
        # y = []
        # for p in td.cntMalleus:
        #     x.append(p[0][0])
        #     y.append(p[0][1])
        # points = [x,y]
        # print(points)
        # # Linear length along the line:
        # distance = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1 )) )
        # distance = np.insert(distance, 0, 0)/distance[-1]

        # print(points)
        # # Build a list of the spline function, one for each dimension:
        # for pts in points:
        #     print(pts)
        # splines = [UnivariateSpline(distance, pts, k=3, s=.2) for pts in points]

        # # Computed the spline for the asked distances:
        # alpha = np.linspace(0, 1, 75)
        # points_fitted = np.vstack( spl(alpha) for spl in splines ).T


        # xnew = np.arange(0,x[-1],0.1)
        cv2.waitKey(2000)

    else:
        print(err)

        # if (s := td.loadMask(mask)) is not None:
        #     print(s)
        #     continue
        # if (s := td.getMalleusAndUmbo()) is not None:
        #     print(s)
        #     continue
        # if (s := td.getQuadrants()) is not None:
        #     print(s)
        #     continue
        # if (s := td.getTarget()) is not None:
        #     print(s)
        #     continue





        # img1 = cv2.drawMarker(img, td.ctrUmbo, (0, 0, 255), cv2.MARKER_STAR, 50)
        # img2 = cv2.drawContours(img, td.quadCnts, -1, (255,255,255), 1)
        # img2 = cv2.drawMarker(img2, td.target, (0, 0, 255), cv2.MARKER_DIAMOND, 50)


        # # cv2.imshow("img",img)
        # cv2.imshow("mask",td.overlayMask(td.overlayMask(img1, td.mask), cv2.cvtColor(td.maskLine,cv2.COLOR_GRAY2RGB)))
        
        # cv2.imshow("line",td.maskLine)

    
