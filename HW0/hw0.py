"""
File: hw0.py
---------------
Course: CS 527
Instructor: Roy Shilkrot
Homework 1: Hello Vision World
Name: Trung Nguyen
SBU ID: 111752939
---------------
This file contains the code to:
1. Read an image from a file ('teton1.jpg') and display it to the screen
2. Add to, subtract from, multiply or divide each pixel with a scalar, display the result
3. Resize the image uniformly by 1/2
"""

import numpy as np
import cv2
#from matplotlib import pyplot as plt

#Load input image and display to the screen:
img = cv2.imread('teton1.jpg')
cv2.imshow('Original image', img)

#Create an mask of the same size of the input with all 1 elements:
unit_image = np.ones(img.shape)

added_scalar = 100
subtracted_scalar = 100
multiple_scalar = 2
divide_scalar = 2

#Arithmetic operations and display:
img_add = cv2.add(img, np.uint8(added_scalar * unit_image))
img_sub = cv2.subtract(img, np.uint8(subtracted_scalar * unit_image))
img_multiple = cv2.multiply(img, np.uint8(multiple_scalar * unit_image))
img_divide = cv2.divide(img, np.uint8(divide_scalar * unit_image))
cv2.imshow('Image added by 100', img_add)
cv2.imshow('Substracted image', img_sub)
cv2.imshow('Image multiplied by 2', img_multiple)
cv2.imshow('Image divided by 2', img_divide)

#Scaling image size by 1/2 and display:
img_scaled = cv2.resize(img, None, fx=0.5, fy=0.5)
cv2.imshow('Image resized by 1/2 uniformly', img_scaled)

#Press any key to close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
