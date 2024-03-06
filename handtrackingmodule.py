import cv2
import mediapipe as mp
import time

class HandDetector():
    def __init__(self, static_image_mode=False, max_num_hands=2, mode_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):


        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode, max_num_hands, mode_complexity, min_detection_confidence, min_tracking_confidence)

        self.mpDraw = mp.solutions.drawing_utils

        self.tipids = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb_img)

        if self.results.multi_hand_landmarks:
            for landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, landmarks, self.mpHands.HAND_CONNECTIONS)
                    
        return img
    
    def findPosition(self, img, handidx=0, draw=True):
        self.landmarklist = []

        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[handidx]
            for idx, landmark in enumerate(hand.landmark):
                h, w, c = img.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                self.landmarklist.append([idx, x, y])

                if draw:
                    cv2.circle(img, (x, y), 15, (255, 0, 255), cv2.FILLED)

        return self.landmarklist
    
    def fingersUP(self):
        fingers = []

        #Thumb
        if self.landmarklist[self.tipids[0]][1] < self.landmarklist[self.tipids[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        #4 Fingers
        for idx in range(1, len(self.tipids)):
            if self.landmarklist[self.tipids[idx]][2] < self.landmarklist[self.tipids[idx]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

def main():
    cam = cv2.VideoCapture(0)

    detector = HandDetector()

    ptime = 0
    while True:
        success, img = cam.read()
        img = detector.findHands(img)
        landmarklist = detector.findPosition(img)
        if len(landmarklist) != 0:
            print(landmarklist[8])

        ctime = time.time()
        fps = int(1/(ctime-ptime))
        ptime = ctime

        cv2.putText(img, str(fps), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()