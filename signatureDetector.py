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


def getContourFromImage(candidateImage):
    greyscaleImage = cv2.cvtColor(candidateImage, cv2.COLOR_BGR2GRAY)
    clonedGreyscaleImage = greyscaleImage.copy()

    threshold, _ = cv2.threshold(src=clonedGreyscaleImage, thresh=0, maxval=255, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cannyedImage = cv2.Canny(image=clonedGreyscaleImage, threshold1=0.5*threshold, threshold2=threshold)

    _, contours, _ = cv2.findContours(image=cannyedImage.copy(), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

    # There is no page in the image
    if len(contours) == 0:
        print('No Page Found')
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
            print('Is contour convex: ' + str(cv2.isContourConvex(contour)))

    contoursInPage = 0
    for contour in contours:
        x, y, _, _ = cv2.boundingRect(points=contour)
        inbetweenX = x > biggestRectangle.x and x < biggestRectangle.x + biggestRectangle.width
        inbetweenY = y > biggestRectangle.y and y < biggestRectangle.y + biggestRectangle.height
        if (inbetweenX) and (inbetweenY):
            contoursInPage += 1

    minimumContoursInPage = 5
    if contoursInPage <= minimumContoursInPage:
        print('No Page Found')
        return candidateImage

    return candidateImage[biggestRectangle.y: biggestRectangle.y + biggestRectangle.height, biggestRectangle.x: biggestRectangle.x + biggestRectangle.width]


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

    candidateContours = getContourFromImage(candidateImage)
    if(display and candidateContours is not candidateImage):
        displayImage(candidateContours, 'Candidate Contours')

    cv2.waitKey(2000)
    cv2.destroyAllWindows()

# run('C:\\Users\\Mihai Pantea\\Desktop\\OpenCV - Signature detection\\images\\release-signed.jpg', 'yes')
# run('C:\\Users\\Mihai Pantea\\Desktop\\OpenCV - Signature detection\\images\\contract-signed.jpg')
run('C:\\Users\\Mihai Pantea\\Desktop\\OpenCV - Signature detection\\images\\contract-unsigned.jpg', 'n')