import glob
import pathlib

import cmath
import dlib
import cv2
import imutils
import numpy as np
import shutil
import datetime
# OPEN CV
import os
from imutils import face_utils
from mtcnn.mtcnn import MTCNN
from time import sleep
import argparse

from skimage import io

###STAŁE
predictor_path = "landmark/sp.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
net = cv2.dnn.readNetFromCaffe("landmark/deploy.prototxt.txt", "landmark/res10_300x300_ssd_iter_140000.caffemodel")
mmod_path = "landmark/mmod_human_face_detector.dat"
cnnFaceDetector = dlib.cnn_face_detection_model_v1("landmark/mmod_human_face_detector.dat")

# Core/landmark/vgg_face_caffe/vgg_face_caffe/VGG_FACE_deploy.prototxt
# net = cv2.dnn.readNetFromCaffe("landmark/PAM_frontal_AlexNet/PAM_frontal_deploy.prototxt.txt", "landmark/PAM_frontal_AlexNet/snap__iter_100000.caffemodel")



faceFolderPath = "Pozytywne/*"
badFaceFolderPath = "Negatywne/"
positiveLister = glob.glob(faceFolderPath)
# HaarCascade prepare data
haarFaceCascade = cv2.CascadeClassifier('HaarCascadeConfigs/haarcascade_frontalface_default.xml')
lbpCascade = cv2.CascadeClassifier('HaarCascadeConfigs/lbpcascade_frontalface_improved.xml')
chinHeightROI = 0.23
confidenceOfDetection = 0.5
imageSizeToResize = 150

haarGoodPath = "WynikiAnalizy\\Haar Cascade\\Dobre\\"
haarBadPath = "WynikiAnalizy\\Haar Cascade\\Zle\\"
lbpGoodPath = "WynikiAnalizy\\Haar Cascade\\Dobre\\"
lbpBadPath = "WynikiAnalizy\\LBP\\Zle\\"
dlibGoodPath = "WynikiAnalizy\\Dlib\\Dobre\\"
dlibBadPath = "WynikiAnalizy\\Dlib\\Zle\\"

personDefPath = "WynikiAnalizy\\ProbkiBadawcze\\"
researchDefPath = "WynikiAnalizy\\"
### ZMIENNE
printDetails = True

goodResult = 0
badResult = 0

goodDeepLearning = 0
badDeepLearning = 0
###

### Sprawdzenie czy istnieje plik do logów
getTime = str(datetime.datetime.now().ctime())
if not (pathlib.Path("LogFile_Etap2.txt").is_file()):
    # os.mknod("/LogFile.txt",0)
    file = open("LogFile_Etap2.txt", 'a')
    file.writelines(
        "\n##################################################################### " + "\nTest : " + getTime + "\n\n")
else:
    file = open("LogFile_Etap2.txt", 'a')
    file.writelines(
        "\n##################################################################### " + "\nTest : " + getTime + "\n\n")


# def removeAllResults(value):
#     if value == 0:
#         shutil.rmtree('WynikiAnalizy\\Haar Cascade\\Zle\\')
#         os.mkdir('WynikiAnalizy\\Haar Cascade\\Zle\\')
#         shutil.rmtree('WynikiAnalizy\\Haar Cascade\\Dobre\\')
#         os.mkdir('WynikiAnalizy\\Haar Cascade\\Dobre\\')
#     elif value == 10:
#         for i in range(1, 11, 1):
#             shutil.rmtree("WynikiAnalizy\\ProbkiBadawcze\\Osoba" + str(i) + "\\Haar\\Dobre")
#             shutil.rmtree("WynikiAnalizy\\ProbkiBadawcze\\Osoba" + str(i) + "\\Haar\\Zle")
#             os.mkdir("WynikiAnalizy\\ProbkiBadawcze\\Osoba" + str(i) + "\\Haar\\Dobre")
#             os.mkdir("WynikiAnalizy\\ProbkiBadawcze\\Osoba" + str(i) + "\\Haar\\Zle")
#
#     elif value == 11:
#         for i in range(1, 11, 1):
#             shutil.rmtree("WynikiAnalizy\\ProbkiBadawcze\\Osoba" + str(i) + "\\LBP\\Dobre")
#             shutil.rmtree("WynikiAnalizy\\ProbkiBadawcze\\Osoba" + str(i) + "\\LBP\\Zle")
#
#             os.mkdir("WynikiAnalizy\\ProbkiBadawcze\\Osoba" + str(i) + "\\LBP\\Dobre")
#             os.mkdir("WynikiAnalizy\\ProbkiBadawcze\\Osoba" + str(i) + "\\LBP\\Zle")
#
#     elif value == 12:
#         for i in range(1, 11, 1):
#             shutil.rmtree("WynikiAnalizy\\ProbkiBadawcze\\Osoba" + str(i) + "\\HOG\\Dobre")
#             shutil.rmtree("WynikiAnalizy\\ProbkiBadawcze\\Osoba" + str(i) + "\\HOG\\Zle")
#             os.mkdir("WynikiAnalizy\\ProbkiBadawcze\\Osoba" + str(i) + "\\HOG\\Dobre")
#             os.mkdir("WynikiAnalizy\\ProbkiBadawcze\\Osoba" + str(i) + "\\HOG\\Zle")
#     elif value == 13:
#         for i in range(1, 11, 1):
#             shutil.rmtree("WynikiAnalizy\\ProbkiBadawcze\\Osoba" + str(i) + "\\DeepLearning\\Dobre")
#             shutil.rmtree("WynikiAnalizy\\ProbkiBadawcze\\Osoba" + str(i) + "\\DeepLearning\\Zle")
#             os.mkdir("WynikiAnalizy\\ProbkiBadawcze\\Osoba" + str(i) + "\\DeepLearning\\Dobre")
#             os.mkdir("WynikiAnalizy\\ProbkiBadawcze\\Osoba" + str(i) + "\\DeepLearning\\Zle")
#     elif value == 14:
#         for i in range(1, 11, 1):
#             shutil.rmtree("WynikiAnalizy\\ProbkiBadawcze\\Osoba" + str(i) + "\\TensorFlow\\Dobre")
#             shutil.rmtree("WynikiAnalizy\\ProbkiBadawcze\\Osoba" + str(i) + "\\TensorFlow\\Zle")
#             os.mkdir("WynikiAnalizy\\ProbkiBadawcze\\Osoba" + str(i) + "\\TensorFlow\\Dobre")
#             os.mkdir("WynikiAnalizy\\ProbkiBadawcze\\Osoba" + str(i) + "\\TensorFlow\\Zle")
#
#
#
#     elif value == 1:
#         shutil.rmtree('WynikiAnalizy\\LBP\\Zle\\')
#         os.mkdir('WynikiAnalizy\\LBP\\Zle\\')
#         shutil.rmtree('WynikiAnalizy\\LBP\\Dobre\\')
#         os.mkdir('WynikiAnalizy\\LBP\\Dobre\\')
#     elif value == 2:
#         shutil.rmtree('WynikiAnalizy\\Dlib\\Zle\\')
#         os.mkdir('WynikiAnalizy\\Dlib\\Zle\\')
#         shutil.rmtree('WynikiAnalizy\\Dlib\\Dobre\\')
#         os.mkdir('WynikiAnalizy\\Dlib\\Dobre\\')
#     elif value == 3:
#         print("Single Shot Detector ")
#     elif value == 4:
#         print("FaceNet")
#     elif value == 10:
#         print("CLEAR ALL PERSON FOLDERS !")
#         for i in range(1, 11, 1):
#             shutil.rmtree("WynikiAnalizy\\ProbkiBadawcze\\Osoba" + str(i) + "\\Haar\\Dobre")
#             shutil.rmtree("WynikiAnalizy\\ProbkiBadawcze\\Osoba" + str(i) + "\\Haar\\Zle")
#             shutil.rmtree("WynikiAnalizy\\ProbkiBadawcze\\Osoba" + str(i) + "\\HOG\\Dobre")
#             shutil.rmtree("WynikiAnalizy\\ProbkiBadawcze\\Osoba" + str(i) + "\\HOG\\Zle")
#             shutil.rmtree("WynikiAnalizy\\ProbkiBadawcze\\Osoba" + str(i) + "\\LBP\\Dobre")
#             shutil.rmtree("WynikiAnalizy\\ProbkiBadawcze\\Osoba" + str(i) + "\\LBP\\Zle")
#
#             os.mkdir("WynikiAnalizy\\ProbkiBadawcze\\Osoba" + str(i) + "\\Haar\\Dobre")
#             os.mkdir("WynikiAnalizy\\ProbkiBadawcze\\Osoba" + str(i) + "\\Haar\\Zle")
#             os.mkdir("WynikiAnalizy\\ProbkiBadawcze\\Osoba" + str(i) + "\\HOG\\Dobre")
#             os.mkdir("WynikiAnalizy\\ProbkiBadawcze\\Osoba" + str(i) + "\\HOG\\Zle")
#             os.mkdir("WynikiAnalizy\\ProbkiBadawcze\\Osoba" + str(i) + "\\LBP\\Dobre")
#             os.mkdir("WynikiAnalizy\\ProbkiBadawcze\\Osoba" + str(i) + "\\LBP\\Zle")


def haarCascadeFaceDetector(inputFilePath, scaleFactor, neighbours, goodPath, badPath):
    # if printDetails:
    #     file.writelines(
    #         getTime + "\t" + "Haar Cascade: neighbours:\t" + str(neighbours) + "\tscaleFactor:\t" + str(
    #             scaleFactor) + "\t")
    #     file.writelines("haarFaceCascade" + "\n")
    #     file.writelines("scaleFactor: " + str(scaleFactor) + "\nneighbours: " + str(neighbours) + "\n\n")

    inputFile = cv2.imread(inputFilePath)
    width, height = inputFile.shape[:2]
    print("width: " + str(width) + " height: " + str(height) + "\n")
    # inputFile = imutils.resize(inputFile, 500)
    grayImage = cv2.cvtColor(inputFile, cv2.COLOR_BGR2GRAY)
    detectedFace = haarFaceCascade.detectMultiScale(grayImage, scaleFactor, neighbours)
    global goodResult, badResult

    if len(detectedFace) != 1:
        cv2.imwrite(badPath + pathlib.Path(inputFilePath).name, inputFile)
        print(len(detectedFace))
        badResult += 1
    else:
        goodResult += 1
        for x, y, w, h in detectedFace:
            # Pokazanie że wykrywa twarz - można pominąć
            cv2.rectangle(inputFile, (x, y), (x + w, y + int(h + (h * 0.2))), (255, 0, 0), 2)
            smart_h = int(h * chinHeightROI)
            if smart_h > height:
                roi_color = inputFile[y:y + (height - 1), x:x + w]
            else:
                roi_color = inputFile[y:y + h + int(smart_h), x:x + w]

            roi_gray = grayImage[y:y + h, x:x + w]
            # croppedImage = cv2.clone
            # cv2.imwrite('WynikiAnalizy\\Haar Cascade\\Dobre\\' + pathlib.Path(inputFilePath).name, roi_color)
            cv2.imwrite(goodPath + pathlib.Path(inputFilePath).name, inputFile)

        # cv2.imshow("image",roi_color)
        # cv2.waitKey(0)

    # if (detectedFace):
    # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # for (x,y,w,h) in detectedFace:
    #     cv2.rectangle(dSave,(x,y),(x+w,y+h),(255,0,0),2)

    #
    # cv2.imshow("Image", inputFile)
    # cv2.waitKey(0)


def lbpCascadeDetector(inputFilePath, scaleFactor, neighbours, goodPath, badPath):
    # if printDetails:
    #     file.writelines(
    #         getTime + "\t" + "LBP: neighbours:\t" + str(neighbours) + "\tscaleFactor:\t" + str(scaleFactor) + "\t")
    # #     file.writelines("lbpCascadeDetector" + "\n\n")
    #     file.writelines("scaleFactor: " + str(scaleFactor) + "\nneighbours: " + str(neighbours) + "\n\n")

    global goodResult, badResult
    inputFile = cv2.imread(inputFilePath)

    width, height = inputFile.shape[:2]
    # inputFile = imutils.resize(inputFile, 500)
    grayImage = cv2.cvtColor(inputFile, cv2.COLOR_BGR2GRAY)
    detectedFace = lbpCascade.detectMultiScale(grayImage, scaleFactor, neighbours)

    if len(detectedFace) != 1:
        cv2.imwrite(badPath + pathlib.Path(inputFilePath).name, inputFile)
        badResult += 1
    else:
        goodResult += 1
        for x, y, w, h in detectedFace:
            smart_h = int(h * chinHeightROI)
            if smart_h > height:
                roi_color = inputFile[y:y + (height - 1), x:x + w]
            else:
                roi_color = inputFile[y:y + h + int(smart_h), x:x + w]
            # Pokazanie że wykrywa twarz - można pominąć

            cv2.rectangle(inputFile, (x, y), (x + w, y + int(h + (h * 0.2))), (255, 0, 0), 2)
            roi_gray = grayImage[y:y + h, x:x + w]
            # croppedImage = cv2.clone
            # cv2.imwrite('WynikiAnalizy\\LBP\\Dobre\\' + pathlib.Path(inputFilePath).name, roi_color)
            cv2.imwrite(goodPath + pathlib.Path(inputFilePath).name, inputFile)

        # cv2.imshow("image",roi_color)
        # cv2.waitKey(0)

    # if (detectedFace):
    # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # for (x,y,w,h) in detectedFace:
    #     cv2.rectangle(dSave,(x,y),(x+w,y+h),(255,0,0),2)

    #
    # cv2.imshow("Image", inputFile)
    # cv2.waitKey(0)


def dlibFaceDetector(inputFilePath, goodPath, badPath):
    if printDetails:
        file.writelines(
            getTime + "\t" + "Histogram of Oriented Gradients: (neighbours:\t")
    global badResult, goodResult
    inputFile = cv2.imread(inputFilePath)
    # ( Width [0], Height [1]
    # inputFile = imutils.resize(inputFile, 500)
    grayImage = cv2.cvtColor(inputFile, cv2.COLOR_BGR2GRAY)
    width, height = inputFile.shape[:2]
    print("width: " + str(width) + " height: " + str(height) + "\n")
    rects = detector(grayImage, 1)
    if len(rects) != 1:
        cv2.imwrite(badPath + pathlib.Path(inputFilePath).name, inputFile)
        badResult += 1
    else:
        goodResult += 1
        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(grayImage, rect)
            shape = face_utils.shape_to_np(shape)

            # Pokazanie że wykrywa twarz - można pominąć
            # cv2.rectangle(inputFile, (x, y), (x + w, y + int(h+(h*0.2))), (255, 0, 0), 2)
            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box

            # udowodnienie że twarz wykrywa
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            if x < 0:
                x = 0
            elif x > width:
                x = width - 1

            if (y < 0):
                y = 0
            elif y > height:
                y = height - 1

            if w < 0:
                w = 0
            elif w > width:
                w = width - 1

            if (h < 0):
                h = 0
            elif h > height:
                h = height - 1

            cv2.rectangle(inputFile, (x, y), (x + w, y + h), (0, 255, 0), 2)

            smart_h = int(h * chinHeightROI)
            roi_color = inputFile[y:y + h, x:x + w]
            #
            # if smart_h > h:
            #     roi_color = inputFile[y:y + (height - 1), x:x + w]
            # else:
            #     roi_color = inputFile[y:y + h + int(smart_h), x:x + w]

            roi_gray = grayImage[y:y + height, x:x + w]
            # croppedImage = cv2.clone
            # cv2.imshow("Output", roi_color)
            # cv2.waitKey(0)
            # show the face number
            # cv2.putText(inputFile, "Face #{}".format(i + 1), (x - 10, y - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                cv2.circle(inputFile, (x, y), 1, (0, 0, 255), -1)

            # show the output image with the face detections + facial landmarks
            cv2.imwrite(goodPath + pathlib.Path(inputFilePath).name, inputFile)


def mtcnnDetector(inputFilePath, goodPath, badPath):
    # if printDetails:
    #     file.writelines(
    #         getTime + "\t" + "Histogram of Oriented Gradients: (neighbours:\t")
    global badResult, goodResult
    inputFile = cv2.imread(inputFilePath)

    detector = MTCNN()
    result = detector.detect_faces(inputFile)
    bounding_box = result[0]['box']

    keypoints = result[0]['keypoints']

    cv2.rectangle(inputFile,
                  (bounding_box[0], bounding_box[1]),
                  (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                  (0, 155, 255),
                  2)
    cv2.imshow("Output", inputFile)
    cv2.waitKey(0)


def caffeDeepLearningDetector(inputFilePath, globalConf, resizeSize, goodPath, badPath):
    # file.writelines(
    #     getTime + "\t" + "DEEPLEARNING_CAFFE: globalConf:\t" + str(globalConf) + "\tresizeSize:\t" + str(
    #         resizeSize) + "\t")
    global badResult, goodResult
    inputFile = cv2.imread(inputFilePath)
    (h, w) = inputFile.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(inputFile, (resizeSize, resizeSize)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    # inputFile = imutils.resize(inputFile, resizeSize)
    blob = cv2.dnn.blobFromImage(inputFile)
    net.setInput(blob)
    detections = net.forward()
    counter = 0
    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]
        if confidence > globalConf:
            counter += 1
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(inputFile, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            cv2.putText(inputFile, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            # cv2.imshow("Output", inputFile)
            # cv2.waitKey(0)
    if counter != 1:
        cv2.imwrite(badPath + pathlib.Path(inputFilePath).name, inputFile)
        print(counter)
        badResult += 1
    else:
        cv2.imwrite(goodPath + pathlib.Path(inputFilePath).name, inputFile)
        print(counter)
        goodResult += 1

def dlibDeepLearningDetector(inputFilePath, resizeSize, upsample, goodPath, badPath):
    # dlib.cuda.buffer
    # if printDetails:
    #     file.writelines(
    #         getTime + "\t" + "DEEPLEARNING_CNN: resizeSize:\t" + str(resizeSize) + "\tupsample:\t" + str(
    #             upsample) + "\t")
    # win = dlib.image_window()
    global badResult, goodResult
    inputFile = cv2.imread(inputFilePath)
    width, height = inputFile.shape[:2]
    if (resizeSize > 0):
        inputFile = cv2.resize(inputFile, (resizeSize, resizeSize))

    copyColor = inputFile

    dets = cnnFaceDetector(inputFile, upsample)
    print("Number of faces detected: {}".format(len(dets)))

    if (len(dets) != 1):
        cv2.imwrite(badPath + pathlib.Path(inputFilePath).name, copyColor)
        badResult += 1
    else:
        goodResult += 1

        for i, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(
                i, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence))

        # rects = dlib.rectangles()
        (x, y, w, h) = d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom()
        # if x < 0:
        #     x = 0
        # elif x > width:
        #     x = width - 1
        #
        # if (y < 0):
        #     y = 0
        # elif y > height:
        #     y = height - 1
        #
        # if w < 0:
        #     w = 0
        # elif w > width:
        #     w = width - 1
        #
        # if (h < 0):
        #     h = 0
        # elif h > height:
        #     h = height - 1
        cv2.rectangle(copyColor, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.imshow("Output", copyColor)
        # cv2.waitKey(0)
        cv2.imwrite(goodPath + pathlib.Path(inputFilePath).name, copyColor)

    # rects.extend([d.rect for d in dets])
    #
    # win.clear_overlay()
    # win.set_image(inputFile)
    # win.add_overlay(rects)
    # dlib.hit_enter_to_continue()


# def researchModeExecutor(startOption, clear, value, lister, goodPath, badPath):
#     global printDetails
#     global goodResult, badResult
#
#     if startOption == 0:
#         print("HaarCascade")
#         removeAllResults(0)
#         counter = 0
#         for image in lister:
#             print(image)
#             print("Iteracja: " + str(counter))
#             counter += 1
#             haarCascadeFaceDetector(image, value[0], value[1], goodPath, badPath)
#             # lbpCascadeDetector(image, 1.5, 5)
#             # dlibFaceDetector(image)
#             # deepLearningDetector(image, confidenceOfDetection, imageSizeToResize)
#             if printDetails:
#                 printDetails = False
#         printDetails = True
#         # file.writelines(getTime + "\t")
#         file.writelines("Results:\t")
#         file.writelines("Good:\t" + str(goodResult) + '\t')
#         file.writelines("Bad:\t" + str(badResult) + '\t')
#         file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
#         goodResult = 0
#         badResult = 0
#
#
#
#
#
#     elif startOption == 1:
#         print("LBP")
#         removeAllResults(1)
#         counter = 0
#         for image in lister:
#             print(image)
#             print("Iteracja: " + str(counter))
#             counter += 1
#             lbpCascadeDetector(image, value[0], value[1], goodPath, badPath)
#             # lbpCascadeDetector(image, 1.5, 5)
#             # dlibFaceDetector(image)
#             # deepLearningDetector(image, confidenceOfDetection, imageSizeToResize)
#             if printDetails:
#                 printDetails = False
#         printDetails = True
#         # file.writelines(getTime + "\t")
#         file.writelines("Results:\t")
#         file.writelines("Good:\t" + str(goodResult) + '\t')
#         file.writelines("Bad:\t" + str(badResult) + '\t')
#         file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
#         goodResult = 0
#         badResult = 0
#     elif startOption == 2:
#         print("Histogram of Oriented Gradients")
#         removeAllResults(2)
#         counter = 0
#         for image in lister:
#             print(image)
#             print("Iteracja: " + str(counter))
#             counter += 1
#             dlibFaceDetector(image, goodPath, badPath)
#             # lbpCascadeDetector(image, 1.5, 5)
#             # dlibFaceDetector(image)
#             # deepLearningDetector(image, confidenceOfDetection, imageSizeToResize)
#             if printDetails:
#                 printDetails = False
#         printDetails = True
#         # file.writelines(getTime + "\t")
#         file.writelines("Results:\t")
#         file.writelines("Good:\t" + str(goodResult) + '\t')
#         file.writelines("Bad:\t" + str(badResult) + '\t')
#         file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
#         goodResult = 0
#         badResult = 0
#     elif startOption == 3:
#         print("Single Shot Detector ")
#         print("Histogram of Oriented Gradients")
#         removeAllResults(3)
#         counter = 0
#         for image in lister:
#             print(image)
#             print("Iteracja: " + str(counter))
#             counter += 1
#             # dlibFaceDetector(image, goodPath, badPath)
#             # lbpCascadeDetector(image, 1.5, 5)
#             # dlibFaceDetector(image)
#             # deepLearningDetector(image, confidenceOfDetection, imageSizeToResize)
#             if printDetails:
#                 printDetails = False
#         printDetails = True
#         # file.writelines(getTime + "\t")
#         file.writelines("Results:\t")
#         file.writelines("Good:\t" + str(goodResult) + '\t')
#         file.writelines("Bad:\t" + str(badResult) + '\t')
#         file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
#         goodResult = 0
#         badResult = 0
#
#     elif startOption == 4:
#         print("FaceNet")
#

def f1ScoreComputer(truePositive, trueNegative, falsePositive, falseNegative, beta):
    print("working hard")
    accuracy = (truePositive + trueNegative) / (truePositive + trueNegative + falsePositive + falseNegative)
    precision = truePositive / (truePositive + falsePositive)
    recall = truePositive / (truePositive + falseNegative)
    fallout = falsePositive / (falsePositive + trueNegative)
    fMeasure = (1 + pow(beta, 2)) * ((precision * recall) / (pow(beta, 2) * (precision + recall)))
    return "\taccuracy:\t" + str(accuracy) + "\tprecision:\t" + str(precision) + "\trecall(TPR):\t" + str(
        recall) + "\tfallout(FPR):\t" + str(fallout) + "\tfMeasure:\t" + str(fMeasure)


getTimeFolderPersons = datetime.datetime.now()
getXTime = str(getTimeFolderPersons.strftime("%Y-%m-%d - %H-%M-%S"))


def researchOrderer(alghoritmName, mode, values, clear):
    global printDetails
    global goodResult, badResult
    falsePositive = 0
    truePositive = 0
    falseNegative = 0
    trueNegative = 0

    global positiveLister
    global getXTime
    getTimeFolderPersons = datetime.datetime.now()
    getXTime = str(getTimeFolderPersons.strftime("%Y-%m-%d - %H-%M-%S"))

    if (alghoritmName == "HAAR"):
        if (mode == "SICK"):
            file.writelines("Positive\t")
            print("HAAR: Sick People")

            pathCore = researchDefPath + "Haar Cascade\\" + getXTime + " Haar SF " + str(values[0]) + " NB " + str(
                values[1]) + "\\"
            pathCore = pathCore.replace(":", " ")

            os.mkdir(pathCore)
            pathGood = pathCore + "Dobre\\"
            pathBad = pathCore + "Zle\\"
            os.mkdir(pathGood)
            os.mkdir(pathBad)
            counter = 0
            for image in positiveLister:
                print(image)
                print("Iteracja: " + str(counter))
                counter += 1
                haarCascadeFaceDetector(image, values[0], values[1], pathGood, pathBad)
                if printDetails:
                    printDetails = False
            printDetails = True
            file.writelines("Results:\t")
            file.writelines("Good:\t" + str(goodResult) + '\t')
            file.writelines("Bad:\t" + str(badResult) + '\t')
            file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
            goodResult = 0
            badResult = 0
        elif (mode == "HEALTHY"):
            print("HAAR: Healthy People")
            # if clear == 0 :
            #     removeAllResults(00)
            pathCore = personDefPath + getXTime + " Haar SF " + str(values[0]) + " NB " + str(
                values[1]) + "\\"
            pathCore = pathCore.replace(":", " ")
            os.mkdir(pathCore)
            pathGood = pathCore + "Dobre\\"
            pathBad = pathCore + "Zle\\"
            pathGoodBad = pathCore + "Dobre_Nietrafione\\"
            pathBadBad = pathCore + "Zle_Nietrafione\\"

            os.mkdir(pathGood)
            os.mkdir(pathBad)
            os.mkdir(pathGoodBad)
            os.mkdir(pathBadBad)

            for i in range(1, 2, 1):
                # pathCore = personDefPath + str(i) + "\\" + getXTime + " Haar SF " + str(values[0]) + " NB " + str(
                #     values[1]) + "\\"

                # os.mkdir(pathCore)

                lister_good = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Dobre/*")
                # lister_moderate = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Srednie/*")
                lister_bad = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Zle/*")

                # file.writelines("Osoba " + str(i) + " " + "Dobre" + ":\t")
                counter = 0
                for image in lister_good:
                    print(image)
                    print("Iteracja: " + str(counter))
                    counter += 1
                    haarCascadeFaceDetector(image, values[0], values[1], pathGood, pathGoodBad)
                    if printDetails:
                        printDetails = False
                printDetails = True

                truePositive += goodResult
                falseNegative += badResult
                # file.writelines("Results:\t")
                # file.writelines("Good:\t" + str(goodResult) + '\t')
                # file.writelines("Good_failed:\t" + str(badResult) + '\t')
                # file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
                goodResult = 0
                badResult = 0

                # file.writelines("Osoba " + str(i) + " " + "Srednie" + ":\t")
                # counter = 0
                # for image in lister_moderate:
                #     print(image)
                #     print("Iteracja: " + str(counter))
                #     counter += 1
                #     haarCascadeFaceDetector(image, values[0], values[1], pathGood, pathBad)
                #     if printDetails:
                #         printDetails = False
                # printDetails = True
                # file.writelines("Results:\t")
                # file.writelines("Good:\t" + str(goodResult) + '\t')
                # file.writelines("Bad:\t" + str(badResult) + '\t')
                # file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
                # goodResult = 0
                # badResult = 0

                # file.writelines("Osoba " + str(i) + " " + "Zle" + ":\t")
                counter = 0
                for image in lister_bad:
                    print(image)
                    print("Iteracja: " + str(counter))
                    counter += 1
                    haarCascadeFaceDetector(image, values[0], values[1], pathBadBad, pathBad)
                    if printDetails:
                        printDetails = False
                printDetails = True

                falsePositive += badResult
                trueNegative += goodResult
                # file.writelines("Results:\t")
                # file.writelines("Bad:\t" + str(goodResult) + '\t')
                # file.writelines("Bad_failed:\t" + str(badResult) + '\t')
                # file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
                goodResult = 0
                badResult = 0

        file.writelines(
            getTime + "\tHaar Cascade: scaleFactor:_" + str(values[0]) + "_neighbours:_" + str(
                values[1]) + "\ttruePositive:\t" + str(truePositive) + "\tfalseNegative:\t" +
            str(falseNegative) + "\tfalsePositive:\t" + str(
                falsePositive) + "\ttrueNegative:\t" + str(
                trueNegative) + "\tTotal:\t" + str(
                truePositive + trueNegative + falsePositive + falseNegative) +
            f1ScoreComputer(truePositive, trueNegative, falsePositive, falseNegative, 1) + "\n")

    elif (alghoritmName == "LBP"):
        if (mode == "SICK"):
            file.writelines("Positive\t")
            print("LBP: Sick People")

            pathCore = researchDefPath + "LBP\\" + getXTime + " LBP SF " + str(values[0]) + " NB " + str(
                values[1]) + "\\"
            pathCore = pathCore.replace(":", " ")

            os.mkdir(pathCore)
            pathGood = pathCore + "Dobre\\"
            pathBad = pathCore + "Zle\\"
            os.mkdir(pathGood)
            os.mkdir(pathBad)
            counter = 0
            for image in positiveLister:
                print(image)
                print("Iteracja: " + str(counter))
                counter += 1
                lbpCascadeDetector(image, values[0], values[1], pathGood, pathBad)
                if printDetails:
                    printDetails = False
            printDetails = True
            file.writelines("Results:\t")
            file.writelines("Good:\t" + str(goodResult) + '\t')
            file.writelines("Bad:\t" + str(badResult) + '\t')
            file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
            goodResult = 0
            badResult = 0
        elif (mode == "HEALTHY"):
            print("LBP: Healthy People")
            # if clear == 0 :
            #     removeAllResults(00)
            pathCore = personDefPath + getXTime + " LBP SF " + str(values[0]) + " NB " + str(
                values[1]) + "\\"
            pathCore = pathCore.replace(":", " ")
            os.mkdir(pathCore)
            pathGood = pathCore + "Dobre\\"
            pathBad = pathCore + "Zle\\"
            pathGoodBad = pathCore + "Dobre_Nietrafione\\"
            pathBadBad = pathCore + "Zle_Nietrafione\\"

            os.mkdir(pathGood)
            os.mkdir(pathBad)
            os.mkdir(pathGoodBad)
            os.mkdir(pathBadBad)

            for i in range(1, 2, 1):
                # pathCore = personDefPath + str(i) + "\\" + getXTime + " Haar SF " + str(values[0]) + " NB " + str(
                #     values[1]) + "\\"

                # os.mkdir(pathCore)

                lister_good = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Dobre/*")
                # lister_moderate = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Srednie/*")
                lister_bad = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Zle/*")

                # file.writelines("Osoba " + str(i) + " " + "Dobre" + ":\t")
                counter = 0
                for image in lister_good:
                    print(image)
                    print("Iteracja: " + str(counter))
                    counter += 1
                    lbpCascadeDetector(image, values[0], values[1], pathGood, pathGoodBad)
                    if printDetails:
                        printDetails = False
                printDetails = True

                truePositive += goodResult
                falseNegative += badResult
                # file.writelines("Results:\t")
                # file.writelines("Good:\t" + str(goodResult) + '\t')
                # file.writelines("Good_failed:\t" + str(badResult) + '\t')
                # file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
                goodResult = 0
                badResult = 0

                # file.writelines("Osoba " + str(i) + " " + "Srednie" + ":\t")
                # counter = 0
                # for image in lister_moderate:
                #     print(image)
                #     print("Iteracja: " + str(counter))
                #     counter += 1
                #     haarCascadeFaceDetector(image, values[0], values[1], pathGood, pathBad)
                #     if printDetails:
                #         printDetails = False
                # printDetails = True
                # file.writelines("Results:\t")
                # file.writelines("Good:\t" + str(goodResult) + '\t')
                # file.writelines("Bad:\t" + str(badResult) + '\t')
                # file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
                # goodResult = 0
                # badResult = 0

                # file.writelines("Osoba " + str(i) + " " + "Zle" + ":\t")
                counter = 0
                for image in lister_bad:
                    print(image)
                    print("Iteracja: " + str(counter))
                    counter += 1
                    lbpCascadeDetector(image, values[0], values[1], pathBadBad, pathBad)
                    if printDetails:
                        printDetails = False
                printDetails = True

                falsePositive += badResult
                trueNegative += goodResult
                # file.writelines("Results:\t")
                # file.writelines("Bad:\t" + str(goodResult) + '\t')
                # file.writelines("Bad_failed:\t" + str(badResult) + '\t')
                # file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
                goodResult = 0
                badResult = 0

            file.writelines(
                getTime + "\tLBP :CNNscaleFactor:" + str(values[0]) + "_neighbours:_" + str(
                    values[1]) + "\ttruePositive:\t" + str(truePositive) + "\tfalseNegative:\t" +
                str(falseNegative) + "\tfalsePositive:\t" + str(
                    falsePositive) + "\ttrueNegative:\t" + str(
                    trueNegative) + "\tTotal:\t" + str(
                    truePositive + trueNegative + falsePositive + falseNegative) +
                f1ScoreComputer(truePositive, trueNegative, falsePositive, falseNegative, 1) + "\n")
    #  globalConf, resizeSize,

    # pathCore = personDefPath + str(i) + "\\" + getXTime + " DLCAFFE GLCONF" + str(
    #     values[0]) + " RSSIZE " + str(
    elif (alghoritmName == "DLCAFFE"):
        if (mode == "SICK"):
            file.writelines("Positive\t")
            print("DLCAFFE: Sick People")

            pathCore = researchDefPath + "DLCAFFE\\" + getXTime + " DLCAFFE GLCONF " + str(
                values[0]) + " RSSIZE " + str(
                values[1]) + "\\"
            pathCore = pathCore.replace(":", " ")

            os.mkdir(pathCore)
            pathGood = pathCore + "Dobre\\"
            pathBad = pathCore + "Zle\\"
            os.mkdir(pathGood)
            os.mkdir(pathBad)
            counter = 0
            for image in positiveLister:
                print(image)
                print("Iteracja: " + str(counter))
                counter += 1
                lbpCascadeDetector(image, values[0], values[1], pathGood, pathBad)
                if printDetails:
                    printDetails = False
            printDetails = True
            file.writelines("Results:\t")
            file.writelines("Good:\t" + str(goodResult) + '\t')
            file.writelines("Bad:\t" + str(badResult) + '\t')
            file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
            goodResult = 0
            badResult = 0
        elif (mode == "HEALTHY"):
            print("LBP: Healthy People")
            # if clear == 0 :
            #     removeAllResults(00)
            pathCore = personDefPath + getXTime + "DLCAFFE GLCONF " + str(values[0]) + " RSSIZE " + str(
                values[1]) + "\\"
            pathCore = pathCore.replace(":", " ")
            os.mkdir(pathCore)
            pathGood = pathCore + "Dobre\\"
            pathBad = pathCore + "Zle\\"
            pathGoodBad = pathCore + "Dobre_Nietrafione\\"
            pathBadBad = pathCore + "Zle_Nietrafione\\"

            os.mkdir(pathGood)
            os.mkdir(pathBad)
            os.mkdir(pathGoodBad)
            os.mkdir(pathBadBad)

            for i in range(1, 2, 1):
                # pathCore = personDefPath + str(i) + "\\" + getXTime + " Haar SF " + str(values[0]) + " NB " + str(
                #     values[1]) + "\\"

                # os.mkdir(pathCore)

                lister_good = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Dobre/*")
                # lister_moderate = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Srednie/*")
                lister_bad = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Zle/*")

                # file.writelines("Osoba " + str(i) + " " + "Dobre" + ":\t")
                counter = 0
                for image in lister_good:
                    print(image)
                    print("Iteracja: " + str(counter))
                    counter += 1
                    caffeDeepLearningDetector(image, values[0], values[1], pathGood, pathGoodBad)
                    if printDetails:
                        printDetails = False
                printDetails = True

                truePositive += goodResult
                falseNegative += badResult
                # file.writelines("Results:\t")
                # file.writelines("Good:\t" + str(goodResult) + '\t')
                # file.writelines("Good_failed:\t" + str(badResult) + '\t')
                # file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
                goodResult = 0
                badResult = 0

                # file.writelines("Osoba " + str(i) + " " + "Srednie" + ":\t")
                # counter = 0
                # for image in lister_moderate:
                #     print(image)
                #     print("Iteracja: " + str(counter))
                #     counter += 1
                #     haarCascadeFaceDetector(image, values[0], values[1], pathGood, pathBad)
                #     if printDetails:
                #         printDetails = False
                # printDetails = True
                # file.writelines("Results:\t")
                # file.writelines("Good:\t" + str(goodResult) + '\t')
                # file.writelines("Bad:\t" + str(badResult) + '\t')
                # file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
                # goodResult = 0
                # badResult = 0

                # file.writelines("Osoba " + str(i) + " " + "Zle" + ":\t")
                counter = 0
                for image in lister_bad:
                    print(image)
                    print("Iteracja: " + str(counter))
                    counter += 1
                    caffeDeepLearningDetector(image, values[0], values[1], pathBadBad, pathBad)
                    if printDetails:
                        printDetails = False
                printDetails = True

                falsePositive += badResult
                trueNegative += goodResult
                # file.writelines("Results:\t")
                # file.writelines("Bad:\t" + str(goodResult) + '\t')
                # file.writelines("Bad_failed:\t" + str(badResult) + '\t')
                # file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
                goodResult = 0
                badResult = 0
            # "DLCAFFE GLCONF " + str(values[0]) + " RSSIZE "
            file.writelines(
                getTime + "\tLBP :DLCAFFE:_GLCONF" + str(values[0]) + "_RSSIZE:_" + str(
                    values[1]) + "\ttruePositive:\t" + str(truePositive) + "\tfalseNegative:\t" +
                str(falseNegative) + "\tfalsePositive:\t" + str(
                    falsePositive) + "\ttrueNegative:\t" + str(
                    trueNegative) + "\tTotal:\t" + str(
                    truePositive + trueNegative + falsePositive + falseNegative) +
                f1ScoreComputer(truePositive, trueNegative, falsePositive, falseNegative, 1) + "\n")

    elif (alghoritmName == "CNNDLIB"):
        if (mode == "SICK"):
            file.writelines("Positive\t")
            print("CNNDLIB: Sick People")

            pathCore = researchDefPath + "CNNDLIB\\" + getXTime + " CNNDLIB RESIZE " + str(
                values[0]) + " UPSAMPLE " + str(
                values[1]) + "\\"
            pathCore = pathCore.replace(":", " ")

            os.mkdir(pathCore)
            pathGood = pathCore + "Dobre\\"
            pathBad = pathCore + "Zle\\"
            os.mkdir(pathGood)
            os.mkdir(pathBad)
            counter = 0
            for image in positiveLister:
                print(image)
                print("Iteracja: " + str(counter))
                counter += 1
                dlibDeepLearningDetector(image, values[0], values[1], pathGood, pathBad)
                if printDetails:
                    printDetails = False
            printDetails = True
            file.writelines("Results:\t")
            file.writelines("Good:\t" + str(goodResult) + '\t')
            file.writelines("Bad:\t" + str(badResult) + '\t')
            file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
            goodResult = 0
            badResult = 0
        elif (mode == "HEALTHY"):
            print("CNNDLIB: Healthy People")
            # if clear == 0 :
            #     removeAllResults(00)
            pathCore = personDefPath + getXTime + " CNNDLIB RESIZE" + str(
                values[0]) + " UPSAMPLE " + str(
                values[1]) + "\\"
            pathCore = pathCore.replace(":", " ")
            os.mkdir(pathCore)
            pathGood = pathCore + "Dobre\\"
            pathBad = pathCore + "Zle\\"
            pathGoodBad = pathCore + "Dobre_Nietrafione\\"
            pathBadBad = pathCore + "Zle_Nietrafione\\"
            os.mkdir(pathGood)
            os.mkdir(pathBad)
            os.mkdir(pathGoodBad)
            os.mkdir(pathBadBad)
            for i in range(1, 11, 1):

                lister_good = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Dobre/*")
                # lister_moderate = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Srednie/*")
                lister_bad = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Zle/*")

                # file.writelines("Osoba " + str(i) + " " + "Dobre" + ":\t")
                counter = 0
                for image in lister_good:
                    print(image)
                    print("Iteracja: " + str(counter))
                    counter += 1
                    dlibDeepLearningDetector(image, values[0], values[1], pathGood, pathGoodBad)
                    if printDetails:
                        printDetails = False
                printDetails = True
                # file.writelines("Results:\t")
                # file.writelines("Good:\t" + str(goodResult) + '\t')
                # file.writelines("Bad:\t" + str(badResult) + '\t')
                # file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
                truePositive += goodResult
                falseNegative += badResult
                goodResult = 0
                badResult = 0

                # file.writelines("Osoba " + str(i) + " " + "Srednie" + ":\t")
                # counter = 0
                # for image in lister_moderate:
                #     print(image)
                #     print("Iteracja: " + str(counter))
                #     counter += 1
                #     dlibDeepLearningDetector(image, values[0], values[1], pathGood, pathBad)
                #     if printDetails:
                #         printDetails = False
                # printDetails = True
                # file.writelines("Results:\t")
                # file.writelines("Good:\t" + str(goodResult) + '\t')
                # file.writelines("Bad:\t" + str(badResult) + '\t')
                # file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
                # goodResult = 0
                # badResult = 0

                # file.writelines("Osoba " + str(i) + " " + "Zle" + ":\t")
                counter = 0
                for image in lister_bad:
                    print(image)
                    print("Iteracja: " + str(counter))
                    counter += 1
                    dlibDeepLearningDetector(image, values[0], values[1], pathBadBad, pathBad)
                    if printDetails:
                        printDetails = False
                printDetails = True
                # file.writelines("Results:\t")
                # file.writelines("Good:\t" + str(goodResult) + '\t')
                # file.writelines("Bad:\t" + str(badResult) + '\t')
                # file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
                falsePositive += badResult
                trueNegative += goodResult
                goodResult = 0
                badResult = 0
            file.writelines(
                getTime + "\tDEEPLEARNING_CNN :resizeSize:" + str(values[0]) + "_upsample:_" + str(
                    values[1]) + "\ttruePositive:\t" + str(truePositive) + "\tfalseNegative:\t" +
                str(falseNegative) + "\tfalsePositive:\t" + str(
                    falsePositive) + "\ttrueNegative:\t" + str(
                    trueNegative) + "\tTotal:\t" + str(
                    truePositive + trueNegative + falsePositive + falseNegative) +
                f1ScoreComputer(truePositive, trueNegative, falsePositive, falseNegative, 1) + "\n")

    elif (alghoritmName == "HOG"):
        if (mode == "SICK"):
            file.writelines("Positive\t")
            print("HOG: Sick People")

            pathCore = researchDefPath + "HOG\\" + getXTime + " HOG " + "\\"
            pathCore = pathCore.replace(":", " ")

            os.mkdir(pathCore)
            pathGood = pathCore + "Dobre\\"
            pathBad = pathCore + "Zle\\"
            os.mkdir(pathGood)
            os.mkdir(pathBad)
            counter = 0
            for image in positiveLister:
                print(image)
                print("Iteracja: " + str(counter))
                counter += 1
                dlibFaceDetector(image, pathGood, pathBad)
                if printDetails:
                    printDetails = False
            printDetails = True
            file.writelines("Results:\t")
            file.writelines("Good:\t" + str(goodResult) + '\t')
            file.writelines("Bad:\t" + str(badResult) + '\t')
            file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
            goodResult = 0
            badResult = 0
        elif (mode == "HEALTHY"):
            print("HOG: Healthy People")
            # if clear == 0 :
            #     removeAllResults(00)
            pathCore = personDefPath + getXTime + " HOG " + "\\"
            pathCore = pathCore.replace(":", " ")
            os.mkdir(pathCore)
            pathGood = pathCore + "Dobre\\"
            pathBad = pathCore + "Zle\\"
            pathGoodBad = pathCore + "Dobre_Nietrafione\\"
            pathBadBad = pathCore + "Zle_Nietrafione\\"
            os.mkdir(pathGood)
            os.mkdir(pathBad)
            os.mkdir(pathGoodBad)
            os.mkdir(pathBadBad)
            for i in range(1, 2, 1):

                lister_good = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Dobre/*")
                # lister_moderate = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Srednie/*")
                lister_bad = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Zle/*")

                # file.writelines("Osoba " + str(i) + " " + "Dobre" + ":\t")
                counter = 0
                for image in lister_good:
                    print(image)
                    print("Iteracja: " + str(counter))
                    counter += 1
                    dlibFaceDetector(image, pathGood, pathGoodBad)
                    if printDetails:
                        printDetails = False
                printDetails = True
                # file.writelines("Results:\t")
                # file.writelines("Good:\t" + str(goodResult) + '\t')
                # file.writelines("Bad:\t" + str(badResult) + '\t')
                # file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
                truePositive += goodResult
                falseNegative += badResult
                goodResult = 0
                badResult = 0

                # file.writelines("Osoba " + str(i) + " " + "Srednie" + ":\t")
                counter = 0
                # for image in lister_moderate:
                #     print(image)
                #     print("Iteracja: " + str(counter))
                #     counter += 1
                #     dlibFaceDetector(image, pathGood, pathBad)
                #     if printDetails:
                #         printDetails = False
                # printDetails = True
                # file.writelines("Results:\t")
                # file.writelines("Good:\t" + str(goodResult) + '\t')
                # file.writelines("Bad:\t" + str(badResult) + '\t')
                # file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
                # goodResult = 0
                # badResult = 0
                #
                # file.writelines("Osoba " + str(i) + " " + "Zle" + ":\t")
                # counter = 0
                for image in lister_bad:
                    print(image)
                    print("Iteracja: " + str(counter))
                    counter += 1
                    dlibFaceDetector(image, pathBadBad, pathBad)
                    if printDetails:
                        printDetails = False
                printDetails = True
                # file.writelines("Results:\t")
                # file.writelines("Good:\t" + str(goodResult) + '\t')
                # file.writelines("Bad:\t" + str(badResult) + '\t')
                # file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
                falsePositive += badResult
                trueNegative += goodResult
                goodResult = 0
                badResult = 0
        file.writelines(
            getTime + "\tHOG: " + "\ttruePositive:\t" + str(truePositive) + "\tfalseNegative:\t" +
            str(falseNegative) + "\tfalsePositive:\t" + str(
                falsePositive) + "\ttrueNegative:\t" + str(
                trueNegative) + "\tTotal:\t" + str(
                truePositive + trueNegative + falsePositive + falseNegative) +
            f1ScoreComputer(truePositive, trueNegative, falsePositive, falseNegative, 1) + "\n")

    if (alghoritmName == "MTCNN"):
        if (mode == "SICK"):
            file.writelines("Positive\t")
            print("MTCNN: Sick People")

            pathCore = researchDefPath + "MTCNN\\" + getXTime + " Haar SF " + str(values[0]) + " NB " + str(
                values[1]) + "\\"
            pathCore = pathCore.replace(":", " ")

            os.mkdir(pathCore)
            pathGood = pathCore + "Dobre\\"
            pathBad = pathCore + "Zle\\"
            os.mkdir(pathGood)
            os.mkdir(pathBad)
            counter = 0
            for image in positiveLister:
                print(image)
                print("Iteracja: " + str(counter))
                counter += 1
                haarCascadeFaceDetector(image, values[0], values[1], pathGood, pathBad)
                if printDetails:
                    printDetails = False
            printDetails = True
            file.writelines("Results:\t")
            file.writelines("Good:\t" + str(goodResult) + '\t')
            file.writelines("Bad:\t" + str(badResult) + '\t')
            file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
            goodResult = 0
            badResult = 0
        elif (mode == "HEALTHY"):
            print("MTCNN: Healthy People")
            # if clear == 0 :
            #     removeAllResults(00)
            pathCore = personDefPath + getXTime + " MTCNN" + "\\"
            pathCore = pathCore.replace(":", " ")
            os.mkdir(pathCore)
            pathGood = pathCore + "Dobre\\"
            pathBad = pathCore + "Zle\\"
            pathGoodBad = pathCore + "Dobre_Nietrafione\\"
            pathBadBad = pathCore + "Zle_Nietrafione\\"

            os.mkdir(pathGood)
            os.mkdir(pathBad)
            os.mkdir(pathGoodBad)
            os.mkdir(pathBadBad)

            for i in range(1, 2, 1):
                # pathCore = personDefPath + str(i) + "\\" + getXTime + " Haar SF " + str(values[0]) + " NB " + str(
                #     values[1]) + "\\"

                # os.mkdir(pathCore)

                lister_good = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Dobre/*")
                # lister_moderate = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Srednie/*")
                lister_bad = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Zle/*")

                # file.writelines("Osoba " + str(i) + " " + "Dobre" + ":\t")
                counter = 0
                for image in lister_good:
                    print(image)
                    print("Iteracja: " + str(counter))
                    counter += 1
                    mtcnnDetector(image, pathGood, pathGoodBad)
                    if printDetails:
                        printDetails = False
                printDetails = True

                truePositive += goodResult
                falseNegative += badResult
                # file.writelines("Results:\t")
                # file.writelines("Good:\t" + str(goodResult) + '\t')
                # file.writelines("Good_failed:\t" + str(badResult) + '\t')
                # file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
                goodResult = 0
                badResult = 0

                # file.writelines("Osoba " + str(i) + " " + "Srednie" + ":\t")
                # counter = 0
                # for image in lister_moderate:
                #     print(image)
                #     print("Iteracja: " + str(counter))
                #     counter += 1
                #     haarCascadeFaceDetector(image, values[0], values[1], pathGood, pathBad)
                #     if printDetails:
                #         printDetails = False
                # printDetails = True
                # file.writelines("Results:\t")
                # file.writelines("Good:\t" + str(goodResult) + '\t')
                # file.writelines("Bad:\t" + str(badResult) + '\t')
                # file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
                # goodResult = 0
                # badResult = 0

                # file.writelines("Osoba " + str(i) + " " + "Zle" + ":\t")
                counter = 0
                for image in lister_bad:
                    print(image)
                    print("Iteracja: " + str(counter))
                    counter += 1
                    mtcnnDetector(image, pathGood, pathGoodBad)
                    if printDetails:
                        printDetails = False
                printDetails = True

                falsePositive += badResult
                trueNegative += goodResult
                # file.writelines("Results:\t")
                # file.writelines("Bad:\t" + str(goodResult) + '\t')
                # file.writelines("Bad_failed:\t" + str(badResult) + '\t')
                # file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
                goodResult = 0
                badResult = 0

        file.writelines(
            getTime + "\tHaar Cascade: scaleFactor:_" + str(values[0]) + "_neighbours:_" + str(
                values[1]) + "\ttruePositive:\t" + str(truePositive) + "\tfalseNegative:\t" +
            str(falseNegative) + "\tfalsePositive:\t" + str(
                falsePositive) + "\ttrueNegative:\t" + str(
                trueNegative) + "\tTotal:\t" + str(
                truePositive + trueNegative + falsePositive + falseNegative) +
            f1ScoreComputer(truePositive, trueNegative, falsePositive, falseNegative, 1) + "\n")


######################### B
######################### A
######################### D
######################### A
######################### N
######################### I
######################### A
# #
# # researchOrderer("CNNDLIB", "HEALTHY", [300, 1], 0)
# # researchOrderer("CNNDLIB", "HEALTHY", [200, 1], 0)
# # researchOrderer("CNNDLIB", "HEALTHY", [700, 1], 0)
# # researchOrderer("CNNDLIB", "HEALTHY", [800, 1], 0)
# # researchOrderer("CNNDLIB", "HEALTHY", [300, 2], 0)
# # researchOrderer("CNNDLIB", "HEALTHY", [200, 2], 0)
# # researchOrderer("CNNDLIB", "HEALTHY", [700, 2], 0)


# # file.writelines("Haar:\n")

# # researchOrderer("HAAR", "SICK", [2, 3], 0)
# # researchOrderer("HAAR", "SICK", [7, 8], 0)
# # researchOrderer("HAAR", "SICK", [3, 5], 0)
# # researchOrderer("HAAR", "SICK", [4, 10], 0)
# # researchOrderer("HAAR", "SICK", [5, 8], 0)
#
# researchOrderer("HAAR", "HEALTHY", [2, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [7, 8], 0)
# researchOrderer("HAAR", "HEALTHY", [3, 5], 0)
# researchOrderer("HAAR", "HEALTHY", [4, 10], 0)
# researchOrderer("HAAR", "HEALTHY", [5, 8], 0)
# # #
# #
# # file.writelines("LBP:\n")
# # researchOrderer("LBP", "SICK", [2, 3], 0)
# # researchOrderer("LBP", "SICK", [7, 8], 0)
# # researchOrderer("LBP", "SICK", [3, 5], 0)
# # researchOrderer("LBP", "SICK", [4, 10], 0)
# # researchOrderer("LBP", "SICK", [5, 8], 0)
# #
# researchOrderer("LBP", "HEALTHY", [2, 3], 0)
# researchOrderer("LBP", "HEALTHY", [7, 8], 0)
# researchOrderer("LBP", "HEALTHY", [3, 5], 0)
# researchOrderer("LBP", "HEALTHY", [4, 10], 0)
# researchOrderer("LBP", "HEALTHY", [5, 8], 0)
#
# # file.writelines("DEEPLEARNING_CAFFE:\n")
# researchOrderer("DLCAFFE", "HEALTHY", [0.5, 100], 0)
#
# researchOrderer("DLCAFFE", "HEALTHY", [0.5, 500], 0)
# researchOrderer("DLCAFFE", "HEALTHY", [0.5, 400], 0)
# researchOrderer("DLCAFFE", "HEALTHY", [0.5, 300], 0)
#
# researchOrderer("DLCAFFE", "HEALTHY", [0.5, 200], 0)



# researchOrderer("DLCAFFE", "SICK", [0.5, 300], 0)
#
# # researchModeExecutor(2, 2, 0, positiveLister, dlibGoodPath, dlibBadPath)
#
#
# researchOrderer("CNNDLIB", "SICK", [300, 1], 0)
# # researchOrderer("CNNDLIB", "SICK", [200, 1], 0)
# # researchOrderer("CNNDLIB", "SICK", [700, 1], 0)
# # researchOrderer("CNNDLIB", "SICK", [800, 1], 0)D
# researchOrderer("CNNDLIB", "HEALTHY", [0, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [0, 2], 0)

#
# # researchOrderer("CNNDLIB", "SICK", [300, 2], 0)
# # researchOrderer("CNNDLIB", "SICK", [200, 2], 0)
# # researchOrderer("CNNDLIB", "SICK", [700, 2], 0)
# # researchOrderer("CNNDLIB", "SICK", [800, 2], 0)
# # researchOrderer("CNNDLIB", "SICK", [0, 2], 0)
#
# print("druga seria")
# # # ZROBIONE
#
# # researchOrderer("CNNDLIB", "HEALTHY", [0, 2], 0)
# # researchOrderer("CNNDLIB", "HEALTHY", [800, 2], 0)
# # researchOrderer("CNNDLIB", "HEALTHY", [0, 1], 0)
#
# researchOrderer("HOG", "HEALTHY", 0, 0)
# researchOrderer("HOG", "SICK", 0, 0)

# researchOrderer("MTCNN","HEALTHY",0,0)


# researchOrderer("HAAR", "HEALTHY", [2, 4], 0)
# researchOrderer("HAAR", "HEALTHY", [7, 5], 0)
# researchOrderer("HAAR", "HEALTHY", [3, 5], 0)
# researchOrderer("HAAR", "HEALTHY", [4, 10], 0)
# researchOrderer("HAAR", "HEALTHY", [5, 8], 0)

# researchOrderer("LBP", "HEALTHY", [2, 4], 0)
# researchOrderer("LBP", "HEALTHY", [7, 8], 0)
# researchOrderer("LBP", "HEALTHY", [3, 5], 0)
# researchOrderer("LBP", "HEALTHY", [4, 10], 0)
# researchOrderer("LBP", "HEALTHY", [5, 8], 0)


# DODATKOWE BY WYRÓWNAC
researchOrderer("CNNDLIB", "HEALTHY", [250, 3], 0)
researchOrderer("CNNDLIB", "HEALTHY", [250, 3], 0)
researchOrderer("CNNDLIB", "HEALTHY", [350, 2], 0)

# researchOrderer("LBP", "HEALTHY", [4, 3], 0)
# researchOrderer("LBP", "HEALTHY", [3, 3], 0)
# researchOrderer("LBP", "HEALTHY", [5, 4], 0)
# researchOrderer("LBP", "HEALTHY", [5, 3], 0)
# researchOrderer("LBP", "HEALTHY", [6, 3], 0)
#
# researchOrderer("HAAR", "HEALTHY", [4, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [3, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [5, 4], 0)
# researchOrderer("HAAR", "HEALTHY", [5, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [6, 3], 0)
file.close()
