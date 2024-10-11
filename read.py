import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np

MIN_PIXEL_WIDTH = 5
MIN_PIXEL_HEIGHT = 8
MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 1.0
MIN_PIXEL_AREA = 80

# Create one-hot encoded arrays for each digit/letter
labels = {chr(i): np.array([1 if j == i else 0 for j in range(32)]) for i in range(32)}

model = keras.models.load_model("my_model.h5")

def plate_inv(img):
    listOfPossiblePlates = []  # This will be the return value
    height, width, numChannels = img.shape

    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    imgHue, imgSaturation, imgValue = cv2.split(imgHSV)

    imgGrayscale = imgValue
    height, width = imgGrayscale.shape

    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement)

    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
    imgMaxContrastGrayscale = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, (5, 5), 0)
    imgThreshScene = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)

    cv2.imshow("Thresholded Image", imgThreshScene)
    return imgThreshScene

def find_character(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    data = img_binary.reshape((-1, 24, 24, 1))

    accuracy = 0.9
    output_char = ''

    for char, label in labels.items():
        result = model.evaluate(data, label.reshape(1, 32), verbose=0)
        if result[1] > accuracy:
            accuracy = result[1]
            output_char = char
        if accuracy > 0.9:
            return output_char, accuracy

    return output_char, accuracy
