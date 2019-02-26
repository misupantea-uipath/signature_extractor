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


def getPageFromImage(candidateImage):
    greyscaleImage = cv2.cvtColor(candidateImage, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Full candidate image as greyscale', greyscaleImage)
    clonedGreyscaleImage = greyscaleImage.copy()

    threshold, _ = cv2.threshold(src=clonedGreyscaleImage, thresh=0, maxval=255, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow('Full candidate image thresholded', threshold)
    cannyedImage = cv2.Canny(image=clonedGreyscaleImage, threshold1=0.5*threshold, threshold2=threshold)
    cv2.imshow('Full candidate image cannyed', cannyedImage)

    _, contours, _ = cv2.findContours(image=cannyedImage.copy(), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

    # There is no page in the image
    if len(contours) == 0:
        print 'No Page Found'
        return candidateImage

    biggestRectangle = Rect(0, 0, 0, 0)
    for contour in contours:
        # Detect edges
        # Reference - http://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html
        epsilon = cv2.arcLength(contour, True)
        corners = cv2.approxPolyDP(contour, 0.1 * epsilon, True)
        x, y, width, height = cv2.boundingRect(points=contour)
        currentArea = width * height

        # check if length of approx is 4
        if len(corners) == 4 and currentArea > biggestRectangle.area:
            biggestRectangle = Rect(x, y, width, height)
            print 'Is contour convex: ' + str(cv2.isContourConvex(contour))

    contoursInPage = 0
    for contour in contours:
        x, y, _, _ = cv2.boundingRect(points=contour)
        inbetweenX = x > biggestRectangle.x and x < biggestRectangle.x + biggestRectangle.width
        inbetweenY = y > biggestRectangle.y and y < biggestRectangle.y + biggestRectangle.height
        if (inbetweenX) and (inbetweenY):
            contoursInPage += 1

    minimumContoursInPage = 5
    if contoursInPage <= minimumContoursInPage:
        print 'No Page Found'
        return candidateImage

    return candidateImage[biggestRectangle.y: biggestRectangle.y + biggestRectangle.height, biggestRectangle.x: biggestRectangle.x + biggestRectangle.width]


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


def getSignature(candidateImage):
    imageSize = np.shape(candidateImage)

    greyscaleImage = cv2.cvtColor(candidateImage, cv2.COLOR_BGR2GRAY)

    blockSize = 21
    constant = 10
    if blockSize > imageSize[0]:
        if imageSize[0] % 2 == 0:
            blockSize = imageSize[0] - 1
        else:
            blockSize = imageSize[0]

    if blockSize > imageSize[1]:
        if imageSize[0] % 2 == 0:
            blockSize = imageSize[1] - 1
        else:
            blockSize = imageSize[1]

    mask = cv2.adaptiveThreshold(greyscaleImage, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=blockSize, C=constant)
    negatedMask = cv2.bitwise_not(mask)

    return cv2.bitwise_and(candidateImage, candidateImage, mask=negatedMask)


imageFolderPath = 'images/'
candidateImage = cv2.imread(imageFolderPath + 'mysig3.jpg')
cv2.imshow('Full candidate image', candidateImage)
key = cv2.waitKey(0)

candidatePage = getPageFromImage(candidateImage)
candidateSignature = getSignatureFromPage(candidatePage)
signature = getSignature(candidateSignature)

cv2.imshow('Signature', signature)
key = cv2.waitKey(0)
cv2.destroyAllWindows()