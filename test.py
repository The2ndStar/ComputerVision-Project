import cv2
import math
import time
import numpy as np
import tensorflow as tf
from handtracking import HandDetector  

model = tf.keras.models.load_model("Model/keras_model.h5")
labels = []
with open("Model/labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1, complexity=0)  

offset = 20
imgSize = 300
counter = 0
save_interval = 0.05
last_save = time.time()

while True:
    success, img = cap.read()
    if not success:
        break
    
    # Find hands and get bounding box
    img, bboxes = detector.findHands(img)
    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255  # White canvas for resizing
    lmList = []  # Initialize list to store hand landmarks

    if bboxes:
        x, y, w, h = bboxes['bbox']
        x, y, w, h = max(0, x - offset), max(0, y - offset), w + 2 * offset, h + 2 * offset
        imgCrop = img[y:y + h, x:x + w]

        imgCropShape = imgCrop.shape
        aspectRatio = h / w

        lmList = detector.findPosition(img, draw=False)

        # Resize the cropped image to maintain aspect ratio
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

        imgWhite = cv2.resize(imgWhite, (224, 224))  

        imgWhiteNormalized = imgWhite / 255.0  
        imgInput = np.expand_dims(imgWhiteNormalized, axis=0)  

        predictions = model.predict(imgInput)
        predicted_class_index = np.argmax(predictions)
        predicted_label = labels[predicted_class_index]
        confidence = predictions[0][predicted_class_index]

        text = f"{predicted_label} ({confidence:.2f})"
        cv2.putText(img, text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (161,132,60), 2)
        cv2.rectangle(img, (x, y), (x + w, y + h), (219, 165, 255), 2)


        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()