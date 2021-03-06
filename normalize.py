#!/usr/bin/python

import cv2
import numpy as np
import tflow_recognize as rcgn

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


class Normalize:
    def __init__(self, path):
        self.windowname = ''
        self.orig_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        self.orig_bgr = cv2.resize(self.orig_bgr, None, fx=1.0, fy=1.0, interpolation = cv2.INTER_CUBIC)
        self.orig_hsv = cv2.cvtColor(self.orig_bgr, cv2.COLOR_BGR2HSV)
        self.orig_gry = cv2.cvtColor(self.orig_bgr, cv2.COLOR_BGR2GRAY)
        self.orig_blr = cv2.blur(self.orig_gry, (3, 3))


    def init_interface(self, windowname):
        self.windowname = windowname
        cv2.createTrackbar('d1', self.windowname, 1, 30, self.nothing)
        cv2.createTrackbar('S1', self.windowname, 0, 200,  self.nothing)
        cv2.createTrackbar('d2', self.windowname, 1, 30,  self.nothing)
        cv2.createTrackbar('S2', self.windowname, 0, 200, self.nothing)
 
    def get(self):
        normalized, background = self.strengthen(self.orig_bgr.copy())
        me = self.middle_end(normalized, background)
        return [self.orig_gry, normalized] + me

    def strengthen(self, img):        
        dilated_0 = self.dilate(self.orig_blr, 11, 2)
        opened_0 = self.dilate(cv2.bitwise_not(dilated_0), 11, 2)
        ret, mask = cv2.threshold(opened_0, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        img[mask == 255] = (0, 0, 0)        
        gry = self.orig_blr.copy()
        gry[mask == 255] = 0

        dilate_1 = self.dilate(cv2.bitwise_not(gry), 11, 3)
        ret, sure_bg = cv2.threshold(dilate_1, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        gry[sure_bg == 0] = 0
        return (gry, sure_bg)

    def middle_end(self, img, background):
        S2 = cv2.getTrackbarPos('S2', self.windowname)

        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        im2, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cnt = self.filter_cnt(contours)
        hud = self.orig_bgr.copy()

        cv2.drawContours(hud, [cnt[S2]], -1, (0,255,0), 3)
        test = self.cnt_to_img(self.orig_gry.copy(), cnt[S2])
        
        print rcgn.recognize(test)

        return [thresh, hud, test]

    def cnt_to_img(self, img, cnt):
        rect = cv2.minAreaRect(cnt)
        x,y,w,h = cv2.boundingRect(cnt)

        angle = rect[2]
        if (rect[1][0] > rect[1][1]):
            angle = 90 + angle

        mask = np.zeros_like(img)
        cv2.drawContours(mask, [cnt], -1, 255, 3)
        
        s = int(float(h)*0.4)
        roi = mask[y-s/2.0:y+h+s/2.0,x-w/2.0-s/2.0:x+h-w/2.0+s/2.0]

        M = cv2.getRotationMatrix2D(((h+s)/2, (h+s)/2), angle, 1)
        roi = cv2.warpAffine(roi, M, (h+s, h+s))
        return roi

    def filter_cnt(self, contours):
        good_cnt = []
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area == 0.0:
                continue

            perimeter = float(cv2.arcLength(cnt, True))
            if perimeter == 0.0:
                continue
        
            p_a = perimeter/area
            if p_a > 1.0:
                continue
                
            good_cnt.append(cnt)
        return good_cnt

    def postprocess(self, img):
        d1 = cv2.getTrackbarPos('d1', self.windowname)
        S1 = cv2.getTrackbarPos('S1', self.windowname)
        S2 = cv2.getTrackbarPos('S2', self.windowname)

        dilated = self.dilate(cv2.bitwise_not(img), d1*2+1, S1)
        ret,thresh = cv2.threshold(dilated, S2, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh.copy(), 1, 2)
        
        return [dilated, thresh]

    def dilate(self, ary, N, iterations):
        kernel = np.zeros((N,N), dtype=np.uint8)
        kernel[(N-1)/2,:] = 1
        dilated_image = cv2.dilate(ary, kernel, iterations=iterations)
        kernel = np.zeros((N,N), dtype=np.uint8)
        kernel[:,(N-1)/2] = 1
        dilated_image = cv2.dilate(dilated_image, kernel, iterations=iterations)
        return dilated_image

    def nothing(self, x):
        pass


normalizer = Normalize('/root/Desktop/a.jpg')

img_disp = ImageDisplay('result')
img_disp.spin(normalizer)
