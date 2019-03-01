import numpy as np
import cv2
import os
import ntpath

class Rect:
    def __init__(self, x=0, y=0, width=0, height=0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.area = width * height


    def pad(self, imageSize, padding):
        self.x -= padding
        self.y -= padding
        self.width += 2 * padding
        self.height += 2 * padding
        if self.x < 0:
            self.x = 0
        if self.y < 0:
            self.y = 0
        if self.x + self.width > imageSize[0]:
            self.width = imageSize[0] - self.x
        if self.y + self.height > imageSize[1]:
            self.height = imageSize[1] - self.y


def get_contours(candidateImage):
    greyscaleImage = cv2.cvtColor(candidateImage, cv2.COLOR_BGR2GRAY)
    clonedGreyscaleImage = greyscaleImage.copy()

    threshold, _ = cv2.threshold(src=clonedGreyscaleImage, thresh=0, maxval=255, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cannyedImage = cv2.Canny(image=clonedGreyscaleImage, threshold1=0.5*threshold, threshold2=threshold)

    _, contours, _ = cv2.findContours(image=cannyedImage.copy(), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

    # There is no page in the image
    if len(contours) == 0:
        print('No Page Found')
        return (False, null)

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
        return (False, None)

    return (True, candidateImage[biggestRectangle.y: biggestRectangle.y + biggestRectangle.height, biggestRectangle.x: biggestRectangle.x + biggestRectangle.width])

def get_signature(candidateImage):
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
    maxRectangle.pad(imageSize=candidateImage.shape, padding=10)

    return candidateImage[maxRectangle.y: maxRectangle.y + maxRectangle.height, maxRectangle.x: maxRectangle.x + maxRectangle.width]


def display_image(image, title='', maxDesiredDimension=1280, minDesiredDimension=720):
    height, width, channels = image.shape

    maximum = max(height, width)
    minimum = min(height, width)

    dsize = (maxDesiredDimension, minDesiredDimension) if height == max else (minDesiredDimension, maxDesiredDimension)
    resized = cv2.resize(image, dsize) 
    cv2.imshow(title, resized)

def parse_bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def clear_folder_contents(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def run(image_path, signature_folder, display):
    image = cv2.imread(image_path)
    display = parse_bool(display)

    #clear_folder_contents(signature_folder)
    
    if(display):
        display_image(image, 'Candidate Image')

    has_contours, contours = get_contours(image)
    if(has_contours):
        signature = get_signature(image)
        image_path_parts = os.path.splitext(image_path)
        signature_path = signature_folder + path_leaf(image_path_parts[0]) + '-signature' + image_path_parts[1]
        cv2.imwrite(signature_path, signature)
        if(display):
            display_image(signature, 'Candidate Signature')

    cv2.waitKey(5000)
    cv2.destroyAllWindows()

#run('C:\\Users\\Mihai Pantea\\Desktop\\OpenCV - Signature detection\\images\\release-signed.jpg', 'C:\\Users\\Mihai Pantea\\Desktop\\OpenCV - Signature detection\\signatures\\', 'y')
#run('C:\\Users\\Mihai Pantea\\Desktop\\OpenCV - Signature detection\\images\\contract-signed.jpg', 'C:\\Users\\Mihai Pantea\\Desktop\\OpenCV - Signature detection\\signatures\\', 'y')
#run('C:\\Users\\Mihai Pantea\\Desktop\\OpenCV - Signature detection\\images\\contract-unsigned.jpg', 'C:\\Users\\Mihai Pantea\\Desktop\\OpenCV - Signature detection\\signatures\\', 'y')