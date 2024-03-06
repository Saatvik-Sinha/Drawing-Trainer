import cv2
import os
import numpy as np

import handtrackingmodule as htm
import poseestimationmodule as pem

folderpath = 'header'
headers = os.listdir(folderpath)
overlayList = []
for headerpath in headers:
    img = cv2.imread(folderpath + "/" + headerpath)
    overlayList.append(img)

#All button height 25-100
#Draw button width 53-222
#Exercise button width 272-441
#Start button width 618-787
#Stop button width 833-1002
#Reset button width 1050-1218

header = overlayList[0]
state = 'draw'
xp, yp = 0, 0

canvas = np.zeros((720, 1280, 3), np.uint8)

cam = cv2.VideoCapture(0)
cam.set(3, 1280)
cam.set(4, 720)

handdetector = htm.HandDetector(min_detection_confidence=0.55)
posedetector = pem.PoseEstimator()

count, dir = 0, 0
while True:
    success, img = cam.read()
    img = cv2.flip(img, 1)

    img = posedetector.findPose(img, draw=False)
    poseLMlist = posedetector.findPosition(img, draw=False)

    if state == 'start' and len(poseLMlist) != 0:
        angle = posedetector.findAngle(img, 12, 14, 16) #Left
        pct = int(np.interp(angle, [30, 120], [100, 0]))
        bar = int(np.interp(angle, [30, 120], [200, 700]))

        barcolor = (255, 0, 255)
        if pct == 100:
            barcolor = (0, 255, 0)
            if dir == 0:
                count += 0.5
                dir = 1
        if pct == 0:
            barcolor = (0, 255, 0)
            if dir == 1:
                count += 0.5
                dir = 0

        cv2.rectangle(img, (1100, 200), (1175, 700), (255, 0, 0), 3)
        cv2.rectangle(img, (1100, bar), (1175, 700), barcolor, cv2.FILLED)
        cv2.putText(img, str(pct)+"%", (1100, 175), cv2.FONT_HERSHEY_PLAIN, 4, barcolor, 4)

    if state == 'start' or state == 'exercise/stop':
        cv2.putText(img, str(int(count)), (25, 700), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

    img = handdetector.findHands(img, False)
    handLMlist = handdetector.findPosition(img, draw=False)
    if len(handLMlist) != 0:
        x1, y1, = handLMlist[8][1], handLMlist[8][2] #Index finger
        x2, y2, = handLMlist[12][1], handLMlist[12][2] #Middle finger

        fingers = handdetector.fingersUP()
        
        if (fingers[1] and fingers[2]): #Selection
            cv2.rectangle(img, (x1-25, y1-25), (x2+25, y2+25), (0, 255, 0), cv2.FILLED)

            if y1 < 100 and y1 > 25:
                if state == 'draw':
                    if 272 < x1 and x1 < 441:
                        header = overlayList[1]
                        state = 'exercise/stop'
                elif state == 'exercise/stop':
                    if 53 < x1 and x1 < 222:
                        header = overlayList[0]
                        state = 'draw'
                    elif 618 < x1 and x1 < 787:
                        header = overlayList[2]
                        state = 'start'
                else:                           
                    if 833 < x1 and x1 < 1002:
                        header = overlayList[1]
                        state = 'exercise/stop'

            if state == 'exercise/stop':
                if 1050 < x1 and x1 < 1218 and y1 < 100 and y1 > 25:
                    header = overlayList[3]
                    count = 0
                else:
                    header = overlayList[1]

        if state == 'draw':
            if fingers[1] and (not fingers[2]): #Drawing
                drawcolor = (0, 165, 255)
                cv2.circle(img, (x1, y1), 15, drawcolor, cv2.FILLED)

                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                
                cv2.line(img, (xp, yp), (x1, y1), drawcolor, 25)
                cv2.line(canvas, (xp, yp), (x1, y1), drawcolor, 25)

                xp, yp = x1, y1
            else:
                xp, yp = 0, 0
        elif state == 'exercise/stop':
            canvas = np.zeros((720, 1280, 3), np.uint8)

    processedCanvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, processedCanvas = cv2.threshold(processedCanvas, 50, 255, cv2.THRESH_BINARY_INV)
    processedCanvas = cv2.cvtColor(processedCanvas, cv2.COLOR_GRAY2BGR)

    img = cv2.bitwise_and(img, processedCanvas)
    img = cv2.bitwise_or(img, canvas)

    img[0:125, 0:1280] = header

    cv2.imshow("Drawing Trainer", img)
    cv2.waitKey(1)