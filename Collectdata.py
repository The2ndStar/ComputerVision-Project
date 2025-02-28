import cv2
import math
import os
import time
import numpy as np
import csv
from handtracking import HandDetector  

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1, complexity=0)  

offset = 20
imgSize = 300
counter = 0
save_interval = 0.05
last_save = time.time()

folder = "data_new/Z_new"
if not os.path.exists(folder):
    os.makedirs(folder)

csv_filename = f"{folder}/all_joints.csv"
while True:
    success, img = cap.read()
    if not success:
        break

    img, bboxes = detector.findHands(img)
    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
    lmList = []  

    if bboxes:
        x, y, w, h = bboxes['bbox']
        x, y, w, h = max(0, x - offset), max(0, y - offset), w + 2 * offset, h + 2 * offset
        imgCrop = img[y:y + h, x:x + w]

        imgCropShape = imgCrop.shape
        aspectRatio = h / w

        lmList = detector.findPosition(img, draw=False)

        if aspectRatio > 1:
            C = imgSize / h
            HC = math.ceil(C * w)
            imgReSize = cv2.resize(imgCrop, (HC, imgSize))
            imgReSizeHeight, imgReSizeWidth = imgReSize.shape[:2]
            hGap = math.ceil((imgSize - HC) / 2)
            imgWhite[0:imgReSizeHeight, hGap:HC + hGap] = imgReSize
        else:
            C = imgSize / w
            WC = math.ceil(C * h)
            imgReSize = cv2.resize(imgCrop, (WC, imgSize))
            imgReSizeHeight, imgReSizeWidth = imgReSize.shape[:2]
            wGap = math.ceil((imgSize - WC) / 2)
            imgWhite[0:imgReSizeHeight, wGap:WC + wGap] = imgReSize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    if lmList and time.time() - last_save > save_interval:
        counter += 1
        joint_data = [counter] 
        image_filename = f"Image_{counter}.jpg"
        image_path = os.path.join(folder, image_filename)

        for joint in lmList:
            joint_data.extend(joint[1:])  

        # Append to the CSV file
        with open(csv_filename, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(joint_data)

        cv2.imwrite(image_path, imgWhite)
        print(f"Saved: {image_filename}")

    
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
