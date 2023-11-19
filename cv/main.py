from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

image = cv2.imread('img.jpeg')
image2 = image.copy()
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)
edged = cv2.Canny(gray, 255, 10)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

# Find contours in the edged image
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

largest_area = 0
largest_rectangle = None

for contour in contours:
    # Approximate the contour as a polygon
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # If the polygon has four vertices, it's likely a rectangle
    if len(approx) == 4:
        area = cv2.contourArea(approx)
        if area > largest_area:
            largest_area = area
            largest_rectangle = approx
            
lines = cv2.HoughLinesP(edged, 1, np.pi/180, 100, minLineLength=500, maxLineGap=250)

# Extend the lines across the entire image
for line in lines:
    x1, y1, x2, y2 = line[0]
    if abs(y2 - y1) < abs(x2 - x1):
        cv2.line(image2, (0, y1), (image.shape[1], y2), (0, 0, 255), 2)
    else:
        cv2.line(image2, (x1, 0), (x2, image.shape[0]), (0, 0, 255), 2)
      
hsv_image = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)
edged = cv2.Canny(gray, 255, 10)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

# Find contours in the edged image
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

largest_area2 = 0
largest_rectangle2 = None

for contour in contours:
    # Approximate the contour as a polygon
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # If the polygon has four vertices, it's likely a rectangle
    if len(approx) == 4:
        cv2.drawContours(image2, [approx], 0, (0, 255, 0), 2)

cv2.imshow('Extended Lines', image2) 

if largest_rectangle is not None:
    x, y, w, h = cv2.boundingRect(largest_rectangle)
    print(f"Largest Rectangle: Width = {w}, Height = {h}")

    res_w,res_h = 21/w, 29.7/h # pixels per metric
    
    
    # Draw the largest rectangle on the original image
    cv2.drawContours(image, [largest_rectangle], 0, (0, 255, 0), 2)
    cv2.imshow('Largest Rectangle', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No rectangles found.")

