import cv2
import sys
from imutils import face_utils
import imutils as imutils
import numpy
import dlib
import numpy as np

# import Image
# import ImageDraw
import glob


# print('Im Alive')
# faceCascade = cv2.CascadeClassifier(cascPath)
# print(sys.path)
# positiveImages = glob.glob("Pozytywne/*")
# negativeImages = glob.glob("Negatywne/*")
# print(glob.__file__)
# for nope in negativeImages :
#    print('x')


class FixedValues:
    inputImage = cv2.imread("Pozytywne/pionier.jpg")


####################################


inputImage = FixedValues().inputImage
grayInputImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

rects = detector(grayInputImage, 1)

# loop over the face detections
for (i, rect) in enumerate(rects):
    # determine the facial landmarks for the face region, then
    # convert the landmark (x, y)-coordinates to a NumPy array
    shape = predictor(grayInputImage, rect)
    shape = face_utils.shape_to_np(shape)

    # loop over the face parts individually
    for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
        # clone the original image so we can draw on it, then
        # display the name of the face part on the image
        clone = inputImage.copy()
        cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)

        # loop over the subset of facial landmarks, drawing the
        # specific face part
        for (x, y) in shape[i:j]:
            cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

        # extract the ROI of the face region as a separate image
        (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
        roi = inputImage[y:y + h, x:x + w]
        roi = imutils.resize(roi, 250, cv2.INTER_CUBIC)

        # show the particular face part
        cv2.imshow("ROI", roi)
        cv2.imshow("Image", clone)
        cv2.waitKey(0)

    # visualize all facial landmarks with a transparent overlay
    output = face_utils.visualize_facial_landmarks(inputImage, shape)
    cv2.imshow("Image", output)
    cv2.waitKey(0)

faceDet = cv2.CascadeClassifier("HaarCascadeConfigs/haarcascade_frontalface_default.xml")
faceDet_two = cv2.CascadeClassifier("HaarCascadeConfigs/haarcascade_frontalface_alt2.xml")
faceDet_three = cv2.CascadeClassifier("HaarCascadeConfigs/haarcascade_frontalface_alt.xml")
faceDet_four = cv2.CascadeClassifier("HaarCascadeConfigs/haarcascade_frontalface_alt_tree.xml")


def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)
