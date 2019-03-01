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
    maxRectangle.addPadding(imgSize=candidateImage.shape, padding=10)

    return candidateImage[maxRectangle.y: maxRectangle.y + maxRectangle.height, maxRectangle.x: maxRectangle.x + maxRectangle.width]


def displayImage(image, title='', maxDesiredDimension=1280, minDesiredDimension=720):
    height, width, channels = image.shape

    maximum = max(height, width)
    minimum = min(height, width)

    dsize = (maxDesiredDimension, minDesiredDimension) if height == max else (minDesiredDimension, maxDesiredDimension)
    resized = cv2.resize(image, dsize) 
    cv2.imshow(title, resized)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
    
def run(imagePath, display):
    candidateImage = cv2.imread(imagePath)
    display = str2bool(display)
    
    if(display):
        displayImage(candidateImage, 'Candidate Image')

    candidateSignature = getSignatureFromPage(candidateImage)
    if(display):
        displayImage(candidateSignature, 'Candidate Signature')
        
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

# run('C:\\Users\\Mihai Pantea\\Desktop\\OpenCV - Signature detection\\images\\release-signed.jpg', 'yes')
# run('C:\\Users\\Mihai Pantea\\Desktop\\OpenCV - Signature detection\\images\\contract-signed.jpg', 'n')