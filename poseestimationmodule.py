import cv2
import mediapipe as mp
import time
import math

class PoseEstimator():
    def __init__(self, mode=False, model_complexity=1, smooth=True, detectionConf=0.5, trackingConf=0.5):
        self.mpDraw = mp.solutions.drawing_utils
        
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=mode, model_complexity=model_complexity, smooth_landmarks=smooth, min_detection_confidence=detectionConf, min_tracking_confidence=trackingConf)

    def findPose(self, img, draw=True):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(rgb_img)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        
        return img

    def findPosition(self, img, draw=True):
        self.landmarklist = []

        if self.results.pose_landmarks:
            for idx, landmark in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                self.landmarklist.append([idx, x, y])

                if draw:
                    cv2.circle(img, (x, y), 5, (255, 0, 0), cv2.FILLED)

        return self.landmarklist
    
    def findAngle(self, img, p1, p2, p3, draw=True, text=False):
        x1, y1 = self.landmarklist[p1][1:]
        x2, y2 = self.landmarklist[p2][1:]
        x3, y3 = self.landmarklist[p3][1:]

        angle = abs(int(math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2, x1-x2))))
        if angle > 180:
            angle = 360 - angle

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.line(img, (x2, y2), (x3, y3), (0, 255, 0), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)

            if text:
                cv2.putText(img, str(angle), (x2-150, y2+50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)

        return angle

def main():
    cam = cv2.VideoCapture(0)
    detector = PoseEstimator()

    ptime = 0
    while True:
        success, img = cam.read()

        img = detector.findPose(img)
        landmarklist = detector.findPosition(img, draw=False)

        if len(landmarklist) > 15:
            cv2.circle(img, (landmarklist[15][1], landmarklist[15][2]), 15, (0, 0, 255), cv2.FILLED)

        ctime = time.time()
        fps = int(1 / (ctime-ptime))
        ptime = ctime

        cv2.putText(img, str(fps), (70, 50), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()