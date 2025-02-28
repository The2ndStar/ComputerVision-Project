import cv2
import mediapipe as mp
import time

class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5, complexity=1):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.complexity = complexity
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,  
            max_num_hands=self.maxHands,  
            model_complexity=self.complexity,  
            min_detection_confidence=self.detectionCon,  
            min_tracking_confidence=self.trackCon          
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        
        bbox = None

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                x_min, y_min = float('inf'), float('inf')
                x_max, y_max = 0, 0

                for landmark in handLms.landmark:
                    x, y = int(landmark.x * img.shape[1]), int(landmark.y * img.shape[0])

                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)

                bbox = {'bbox': (x_min, y_min, x_max - x_min, y_max - y_min)}
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS,self.mpDraw.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=3), 
                           self.mpDraw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3))
                   

        return img, bbox

    def findPosition(self, img, handNo=0 ,draw =True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            
            for id, lm in enumerate(myHand.landmark):
                h,w,c = img.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                lmList.append([id , cx, cy])
            
            
        return lmList

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector(complexity=0)  

    while True:
        success, img = cap.read()
        img, bboxes = detector.findHands(img)
        lmList = detector.findPosition(img)

    
        if lmList:  
            print(lmList[1])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)
      


if __name__ == "__main__":
    main()


