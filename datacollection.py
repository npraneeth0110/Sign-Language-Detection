import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
counter = 0

folder = r"D:\Sign Language Detection\Data\Hello"
os.makedirs(folder, exist_ok=True)

while True:
    success, img = cap.read()
    if not success:
        continue

    img = cv2.resize(img, (1280, 720))
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
        x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size == 0:
            continue

        aspectRatio = h / w

        if aspectRatio > 1:
            # Handle portrait orientation
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap: wGap + wCal] = imgResize

        else:
            # Handle landscape orientation
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hGap + hCal, :] = imgResize

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', img)

    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        filename = os.path.join(folder, f'Image_{int(time.time())}.jpg')
        cv2.imwrite(filename, imgWhite)
        print(f"Saved {filename} ({counter})")

    elif key == 27:  # Escape key to exit
        break

cap.release()
cv2.destroyAllWindows()
