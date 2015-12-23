#!/usr/bin/python

import numpy as np
import cv2

class ImageDisplay:
    def __init__(self, windowname):
        self.windowname = windowname
        cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
    
    def show_blend(self, images, windowname):
        if len(images) == 0:
            print "WARNING: No images provided!"
            return

        for i, image in enumerate(images):
            if (len(image.shape) != 3) or (image.shape[2] != 3):
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                
            if (i == 0): 
                both = image
                continue
            
            both = np.hstack((both, image))
        cv2.imshow(windowname, both)

    def show_separate(self, images, windowname):
        if len(images) == 0:
            print "WARNING: No images provided!"
            return

        for i, image in enumerate(images):
            name = windowname
            if i != 0:
                name = windowname + '_' +  str(i)
                cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.imshow(name, image)

    def show(self, images, windowname = 'result', blend = True):
        if blend:
            self.show_blend(images, windowname)
        else:
            self.show_separate(images, windowname)

    def show_wait(self, images, blend = True):
        self.show(images, blend)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def spin(self, imager):
        imager.init_interface(self.windowname)

        while (1):
            images = imager.get()
            self.show_blend(images, self.windowname)

            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

class Normalize:
    def __init__(self, path):
        self.windowname = ''
        self.orig_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        self.orig_bgr = cv2.resize(self.orig_bgr, None, fx=0.8, fy=0.8, interpolation = cv2.INTER_CUBIC)
        self.orig_hsv = cv2.cvtColor(self.orig_bgr, cv2.COLOR_BGR2HSV)
        self.orig_gry = cv2.cvtColor(self.orig_bgr, cv2.COLOR_BGR2GRAY)

        self.orig_blr = cv2.blur(self.orig_gry, (3, 3))

    def init_interface(self, windowname):
        self.windowname = windowname
        cv2.createTrackbar('d1', self.windowname, 0, 15, self.nothing)
        cv2.createTrackbar('S1', self.windowname, 1, 15,  self.nothing)
        cv2.createTrackbar('d2', self.windowname, 0, 15,  self.nothing)
        cv2.createTrackbar('S2', self.windowname, 1, 15, self.nothing)
 
    def get(self):
        d1 = cv2.getTrackbarPos('d1', self.windowname)
        S1 = cv2.getTrackbarPos('S1', self.windowname)
        return [self.orig_gry, self.dilate(cv2.bitwise_not(self.orig_gry), d1*2+1, S1)]

    def dilate(self, ary, N, iterations):
        kernel = np.zeros((N,N), dtype=np.uint8)
        kernel[(N-1)/2,:] = 1
        dilated_image = cv2.dilate(ary, kernel, iterations=iterations)
        kernel = np.zeros((N,N), dtype=np.uint8)
        kernel[:,(N-1)/2] = 1
        dilated_image = cv2.dilate(dilated_image, kernel, iterations=iterations)
        return dilated_image


    def get_text_th(self, img):
        d2 = cv2.getTrackbarPos('d2', self.windowname)
        S2 = cv2.getTrackbarPos('S2', self.windowname)
        d1 = cv2.getTrackbarPos('d1', self.windowname)
        S1 = cv2.getTrackbarPos('S1', self.windowname)

        lapl = cv2.Laplacian(img, cv2.CV_32F)
        bilat = cv2.bilateralFilter(lapl, 6, 75, 75)

        ret, bth = cv2.threshold(bilat, 0.9, 1, cv2.THRESH_BINARY_INV)
        bth = np.mat(bth * 255, np.uint8)

        merge = cv2.bitwise_and(img, bth)
        return [bilat, merge]

    def nothing(self, x):
        pass


normalizer = Normalize('/root/Desktop/a.jpg')

img_disp = ImageDisplay('result')
img_disp.spin(normalizer)
