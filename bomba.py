#!/usr/bin/python

import cv2
import numpy as np

class ImageDisplay:
    def __init__(self, windowname):
        self.windowname = windowname
        cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
    
    def show_blend(self, images, windowname):
        if len(images) == 0:
            print "WARNING: No images provided!"
            return

        height0, width0 = images[0].shape[:2]

        for i, image in enumerate(images):
            if (len(image.shape) != 3) or (image.shape[2] != 3):
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            height, width = image.shape[:2]
            if (height != height0):
                image = cv2.resize(image, (int(float(width*height0)/float(height)), height0), fx=0, fy=0, interpolation = cv2.INTER_CUBIC)

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

class Process:
    def __init__(self, path):
        self.windowname = ''
        self.orig_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        self.orig_bgr = cv2.resize(self.orig_bgr, None, fx=1.0, fy=1.0, interpolation = cv2.INTER_CUBIC)
        self.orig_hsv = cv2.cvtColor(self.orig_bgr, cv2.COLOR_BGR2HSV)
        self.orig_gry = cv2.cvtColor(self.orig_bgr, cv2.COLOR_BGR2GRAY)

    def nothing(self):
        return

    def init_interface(self, windowname):
        self.windowname = windowname
        cv2.createTrackbar('l1', self.windowname, 10, 255, self.nothing)
        cv2.createTrackbar('l2', self.windowname, 5, 255, self.nothing)
        cv2.createTrackbar('l3', self.windowname, 0, 255, self.nothing)
        cv2.createTrackbar('r1', self.windowname, 255, 255, self.nothing)
        cv2.createTrackbar('r2', self.windowname, 255, 255, self.nothing)
        cv2.createTrackbar('r3', self.windowname, 255, 255, self.nothing)


    def hist(self, image):
        from matplotlib import pyplot as plt
        plt.hist(image.ravel(),256,[0,256])
        #plt.draw()

    def get(self):
        r1 = cv2.getTrackbarPos('r1', self.windowname)
        r2 = cv2.getTrackbarPos('r2', self.windowname)
        r3 = cv2.getTrackbarPos('r3', self.windowname)
        l1 = cv2.getTrackbarPos('l1', self.windowname)
        l2 = cv2.getTrackbarPos('l2', self.windowname)
        l3 = cv2.getTrackbarPos('l3', self.windowname)


        # Remove light     
        h, s, v = cv2.split(self.orig_hsv)
        kernel = np.ones((9*2+1, 9*2+1), np.uint8)
        v_dilated = cv2.dilate(v, kernel, iterations = 1)
        v_out = cv2.subtract(v_dilated, v)

        #ret, v_t = cv2.threshold(v, l3, r3, cv2.THRESH_TRUNC)
        
        # Binarization
        #ret, ots = cv2.threshold(v_out, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        #et, ots2 = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


        #self.hist(v_out)
        #for i in xrange(l1):
        #    ret, mask = cv2.threshold(v_out, l2, 255, cv2.THRESH_TOZERO)
        #    v_out = cv2.bitwise_and(v_out, mask)
        #    v_out = cv2.add(v_out, (v_out/l3))

        v_out = cv2.bitwise_not(v_out)

        th3 = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,l1*2+1,l2)
        th4 = cv2.adaptiveThreshold(v_out, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,l1*2+1,l2)

        return [v_out, th3, th4]



p = Process('/root/Desktop/b.jpg')

img_disp = ImageDisplay('result')
img_disp.spin(p)
