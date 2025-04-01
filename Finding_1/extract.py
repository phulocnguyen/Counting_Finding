import cv2
import numpy as np
import os

output_dir = "/Users/phulocnguyen/Documents/Workspace/Image Processing/Finding_1/items"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

img = cv2.imread('/Users/phulocnguyen/Documents/Workspace/Image Processing/Finding_1/objects.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
_, thresh = cv2.threshold(img_blur, 200, 255, cv2.THRESH_BINARY_INV)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

area_threshold = 1500  

for i, contour in enumerate(contours):
    if cv2.contourArea(contour) < 100:
        continue
    x, y, w, h = cv2.boundingRect(contour)
    if w * h < area_threshold:  
        continue
    item = img[y:y+h, x:x+w]
    cv2.imwrite(f"{output_dir}/item_{i+1}.png", item)
