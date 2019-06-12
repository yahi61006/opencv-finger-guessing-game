import cv2
import numpy as np
import copy
import math
import random
from time import time

# parameters
cap_region_x_begin=0.5  # start point/total width
cap_region_y_end=0.8  # start point/total width
threshold = 60  #  BINARY threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0

# variables
isBgCaptured = 0   # bool, whether the background captured

def removeBG(frame):
    fgmask = bgModel.apply(frame,learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    frame_NoBG = cv2.bitwise_and(frame, frame, mask=fgmask)
    return frame_NoBG

def calculateFingers(res,drawing):  # -> finished bool, cnt: finger count
    hull = cv2.convexHull(res, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(res, hull)
        if type(defects) != type(None):  # avoid crashing.   (BUG not found)
            cnt = 0
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(res[s][0]) # point of the contour where the defect begins
                end = tuple(res[e][0])   # point of the contour where the defect ends
                far = tuple(res[f][0])   # the farthest from the convex hull point within the defect
                #算三角形邊長
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) #二維座標算兩點間的長度
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                    cnt += 1
                    cv2.circle(drawing, far, 8, [211, 84, 0], -1)
            return True, cnt
    return False, 0

# Camera
camera = cv2.VideoCapture(0)
camera.set(10,200)
start_time=int(time())
random_num=random.randint(1,5)
guess_num=0
score=0

while camera.isOpened():
    ret, frame = camera.read()
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
    frame = cv2.flip(frame, 1)  # flip the frame horizontally
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                 (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
    cv2.putText(frame,'Guess Number(1-5)',(0,30), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255,0,255),2)
    cv2.putText(frame,"press b to set backround",(0,150), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,200,0),2)
    cv2.putText(frame,str("Score "+str(score)),(0,400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0),3)

    #  Main operation
    if isBgCaptured == 1:  # this part wont run until background captured
        if(guess_num==0):
            cv2.putText(frame,'Please guess',(0,300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),3)
        else:
            cv2.putText(frame,'guess num:{}'.format(guess_num),(0,300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),3)
        if((int(time())-start_time)>3):
            print(random_num)
            if(guess_num==random_num):
                print("good")
                score+=5
                random_num=random.randint(1,5)
            else:
                print("wrong")
                score-=1
            start_time=int(time())

        img = removeBG(frame)
        img = img[0:int(cap_region_y_end * frame.shape[0]),
                    int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI

        # convert the image into binary image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)

        # get the coutours
        thresh1 = copy.deepcopy(thresh)
        contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        length = len(contours)
           
        if length > 0:
            res = contours[0]
            hull = cv2.convexHull(res)
            drawing = np.zeros(img.shape, np.uint8)
            #cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
            #cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)
            isFinishCal,cnt = calculateFingers(res,drawing)

        if(length==0):
            guess_num=0
        else:
            guess_num=cnt+1

        #cv2.imshow('output', drawing)  

    cv2.imshow('original', frame)

    k = cv2.waitKey(10)
    if k == 27:  # press ESC to exit
        camera.release()
        cv2.destroyAllWindows()
        break
    elif k == ord('b'):  # press 'b' to capture the background
        bgModel = cv2.createBackgroundSubtractorMOG2()
        isBgCaptured = 1
        print( 'Background Captured')
    elif k == ord('r'):  # press 'r' to reset the background
        bgModel = None
        isBgCaptured = 0
        print ('Reset BackGround')