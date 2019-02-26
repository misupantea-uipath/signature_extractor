###############################
#
#   (c) Vlad Zat 2017
#   Student No: C14714071
#   Course: DT228
#   Date: 06-10-2017
#
#	Title: Signature Extractor

#   Get Signature from Page:
#   * convert image to grayscale
#   * get edges of the signature
#   * close the result
#   * find contours
#   * find the contour with the biggest bounding rectangle area
#   * add padding to the bounding rectangle
#   * generate a new image that only contains the largest bounding rectangle

# Extra Notes:
# Filtering is not necessary because writting doesn't have a texture to impede edge detection
# Filtering in this case only makes the text harder to read

import numpy as np
import cv2


class Rect:
    def __init__(self, x=0, y=0, width=0, height=0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.area = width * height


    def addPadding(self, imgSize, padding):
        self.x -= padding
        self.y -= padding
        self.width += 2 * padding
        self.height += 2 * padding
        if self.x < 0:
            self.x = 0
        if self.y < 0:
            self.y = 0
        if self.x + self.width > imgSize[0]:
            self.width = imgSize[0] - self.x
        if self.y + self.height > imgSize[1]:
            self.height = imgSize[1] - self.y


def getSignatureFromPage(candidateImage):
    imageSize = np.shape(candidateImage)

    greyscaleImage = cv2.cvtColor(candidateImage, cv2.COLOR_BGR2GRAY)

    # The values for edge detection can be approximated using Otsu's Algorithm
    # Reference - http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.402.5899&rep=rep1&type=pdf
    threshold, _ = cv2.threshold(src=greyscaleImage, thresh=0, maxval=255, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cannyedImage = cv2.Canny(image=greyscaleImage, threshold1=0.5*threshold, threshold2=threshold)

    # Close the image to fill blank spots so blocks of text that are close together (like the signature) are easier to detect
    # Signature usually are wider and shorter so the strcturing elements used for closing will have this ratio
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(30, 1))
    cannyedImage = cv2.morphologyEx(src=cannyedImage, op=cv2.MORPH_CLOSE, kernel=kernel)

    # findContours is a distructive function so the image pased is only a copy
    _, contours, _ = cv2.findContours(image=cannyedImage.copy(), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

    maxRectangle = Rect(0, 0, 0, 0)
    maxCorners = 0
    for contour in contours:
        epsilon = cv2.arcLength(contour, True)
        corners = cv2.approxPolyDP(contour, 0.01 * epsilon, True)
        x, y, w, h = cv2.boundingRect(points=contour)
        # Maybe add w > h ?
        # if currentArea > maxRect.getArea():
        if len(corners) > maxCorners:
            maxCorners = len(corners)
            maxRectangle = Rect(x, y, w, h)

    # Increase the bounding box to get a better view of the signature
    maxRectangle.addPadding(imgSize=imageSize, padding=10)

    return candidateImage[maxRectangle.y: maxRectangle.y + maxRectangle.height, maxRectangle.x: maxRectangle.x + maxRectangle.width]
    

def display(image, title='', maxDesiredDimension=1280, minDesiredDimension=720):
    height, width, channels = image.shape

    maximum = max(height, width)
    minimum = min(height, width)

    dsize = (maxDesiredDimension, minDesiredDimension) if height == max else (minDesiredDimension, maxDesiredDimension)
    resized = cv2.resize(image, dsize) 
    cv2.imshow(title, resized)


def run():
    candidateImage = cv2.imread('release.jpeg')
    display(candidateImage, 'Candidate Image')

    candidateSignature = getSignatureFromPage(candidateImage)
    display(candidateSignature, 'Candidate Signature')


run()
cv2.waitKey(0)