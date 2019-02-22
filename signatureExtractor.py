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
    def __init__(self, x = 0, y = 0, widht = 0, height = 0):
        self.x = x
        self.y = y
        self.width = widht
        self.height = height
        self.area = 0

    def setArea(self, area):
        self.area = area
    def getArea(self):
        return self.area

    def set(self, x, y, width, height):
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

imageFolderPath = 'images/'
candidateImage = cv2.imread(imageFolderPath + 'mysig3.jpg')

def getPageFromImage(candidateImage):
    imageSize = np.shape(candidateImage)

    greyscaleImage = cv2.cvtColor(candidateImage, cv2.COLOR_BGR2GRAY)
    clonedGreyscaleImage = greyscaleImage.copy()

    threshold, _ = cv2.threshold(src = clonedGreyscaleImage, thresh = 0, maxval = 255, type = cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cannyedImage = cv2.Canny(image = clonedGreyscaleImage, threshold1 = 0.5 * threshold, threshold2 = threshold)

    _, contours, _ = cv2.findContours(image = cannyedImage.copy(), mode = cv2.RETR_TREE, method = cv2.CHAIN_APPROX_SIMPLE)

    # There is no page in the image
    if len(contours) == 0:
        print 'No Page Found'
        return candidateImage

    biggestRectangle = Rect(0, 0, 0, 0)
    coordinates = []
    for contour in contours:
        # Detect edges
        # Reference - http://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html
        epsilon = cv2.arcLength(contour, True)
        corners = cv2.approxPolyDP(contour, 0.1 * epsilon, True)
        x, y, width, height = cv2.boundingRect(points = contour)
        currentArea = width * height

        # check if length of approx is 4
        if len(corners) == 4 and currentArea > biggestRectangle.getArea():
            biggestRectangle.set(x, y, width, height)
            print 'Is contour convex: ' + str(cv2.isContourConvex(contour))

    contoursInPage = 0
    for contour in contours:
        x, y, _, _ = cv2.boundingRect(points = contour)
        if (x > biggestRectangle.x and x < biggestRectangle.x + biggestRectangle.width) and (y > biggestRectangle.y and y < biggestRectangle.y + biggestRectangle.height):
                contoursInPage += 1

    minimumContoursInPage = 5
    if contoursInPage <= minimumContoursInPage:
        print 'No Page Found'
        return candidateImage

    return candidateImage[biggestRectangle.y : biggestRectangle.y + biggestRectangle.height, biggestRectangle.x : biggestRectangle.x + biggestRectangle.width]


def getSignatureFromPage(candidateImage):
    imageSize = np.shape(candidateImage)

    greyscaleImage = cv2.cvtColor(candidateImage, cv2.COLOR_BGR2GRAY)

    # The values for edge detection can be approximated using Otsu's Algorithm
    # Reference - http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.402.5899&rep=rep1&type=pdf
    threshold, _ = cv2.threshold(src = greyscaleImage, thresh = 0, maxval = 255, type = cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cannyedImage = cv2.Canny(image = greyscaleImage, threshold1 = 0.5 * threshold, threshold2 = threshold)

    # Close the image to fill blank spots so blocks of text that are close together (like the signature) are easier to detect
    # Signature usually are wider and shorter so the strcturing elements used for closing will have this ratio
    kernel = cv2.getStructuringElement(shape = cv2.MORPH_RECT, ksize = (30, 1))
    cannyedImage = cv2.morphologyEx(src = cannyedImage, op = cv2.MORPH_CLOSE, kernel = kernel)

    # findContours is a distructive function so the image pased is only a copy
    _, contours, _ = cv2.findContours(image = cannyedImage.copy(), mode = cv2.RETR_TREE, method = cv2.CHAIN_APPROX_SIMPLE)

    maxRect = Rect(0, 0, 0, 0)
    maxCorners = 0
    for contour in contours:
        epsilon = cv2.arcLength(contour, True)
        corners = cv2.approxPolyDP(contour, 0.01 * epsilon, True)
        x, y, w, h = cv2.boundingRect(points = contour)
        currentArea = w * h
        # Maybe add w > h ?
        # if currentArea > maxRect.getArea():
        if len(corners) > maxCorners:
            maxCorners = len(corners)
            maxRect.set(x, y, w, h)

    # Increase the bounding box to get a better view of the signature
    maxRect.addPadding(imgSize = imageSize, padding = 10)

    return candidateImage[maxRect.y : maxRect.y + maxRect.height, maxRect.x : maxRect.x + maxRect.width]

def getSignature(img):
    imgSize = np.shape(img)

    gImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # minBlockSize = 3
    # maxBlockSize = 101
    # minC = 3
    # maxC = 101
    #
    # bestContourNo = 1000000
    # bestBlockSize = 0
    # bestC = 0
    #
    # for c in range(minC, maxC, 2):
    #     for bs in range(minBlockSize, maxBlockSize, 2):
    #         mask = cv2.adaptiveThreshold(gImg, maxValue = 255, adaptiveMethod = cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType = cv2.THRESH_BINARY, blockSize = bs, C = c)
    #         rmask = cv2.bitwise_not(mask)
    #         _, contours, _ = cv2.findContours(image = rmask.copy(), mode = cv2.RETR_TREE, method = cv2.CHAIN_APPROX_SIMPLE)
    #         if len(contours) > 15 and len(contours) < bestContourNo:
    #             bestContourNo = len(contours)
    #             bestBlockSize = bs
    #             bestC = c

    # blockSize = 21, C = 10

    # TODO throw error if blockSize is bigger than image
    blockSize = 21
    C = 10
    if blockSize > imgSize[0]:
        if imgSize[0] % 2 == 0:
            blockSize = imgSize[0] - 1
        else:
            blockSize = imgSize[0]

    if blockSize > imgSize[1]:
        if imgSize[0] % 2 == 0:
            blockSize = imgSize[1] - 1
        else:
            blockSize = imgSize[1]

    mask = cv2.adaptiveThreshold(gImg, maxValue = 255, adaptiveMethod = cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType = cv2.THRESH_BINARY, blockSize = blockSize, C = C)
    rmask = cv2.bitwise_not(mask)

    return cv2.bitwise_and(candidateImage, candidateImage, mask=rmask)

candidateImage = getPageFromImage(candidateImage = candidateImage)
candidateImage = getSignatureFromPage(candidateImage = candidateImage)
candidateImage = getSignature(img = candidateImage)

cv2.imshow('Signature', candidateImage)
key = cv2.waitKey(0)