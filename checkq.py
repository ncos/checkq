#!/usr/bin/python

from matplotlib import pyplot as plt
import numpy as np
import cv2

def nothing(x):
    pass

def show_both(im1, im2):
    both = np.hstack((im1, im2))
    cv2.imshow('result', both)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_separate(im):
    for i, image in enumerate(im):
        cv2.imshow('image_'+str(i), image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def imshow(name, im):
    im_ = cv2.resize(im, None, fx=0.2, fy=0.2, interpolation = cv2.INTER_CUBIC)
    cv2.imshow(name, im_)

def get_text_th(img):
    cv2.namedWindow('result')
    cv2.createTrackbar('d1','result',1,255, nothing)
    cv2.createTrackbar('S1','result',1,15, nothing)
    cv2.createTrackbar('d2','result',1,15, nothing)
    cv2.createTrackbar('S2','result',1,255, nothing)
    img = cv2.blur(img, (3, 3))

    while (1):
        d2 = cv2.getTrackbarPos('d2','result')
        S2 = cv2.getTrackbarPos('S2','result')
        d1 = cv2.getTrackbarPos('d1','result')
        S1 = cv2.getTrackbarPos('S1','result')

        lapl = cv2.Laplacian(img, cv2.CV_32F)
        bilat = cv2.bilateralFilter(lapl, 6, 75, 75)
        imshow('result', bilat)

        ret, bth = cv2.threshold(bilat, 0.9, 1, cv2.THRESH_BINARY_INV)
        bth = np.mat(bth * 255, np.uint8)

        merge = cv2.bitwise_and(img, bth)
        imshow('m', merge)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break




# Load an color image in grayscale
img = cv2.imread('/root/Desktop/a.jpg', cv2.IMREAD_GRAYSCALE)
get_text_th(img)


