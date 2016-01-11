#!/usr/bin/python

import cv2
import numpy as np
from optparse import OptionParser
from os.path import isfile, join
import os

def denoise(image, FM=True):
    if FM:
        return cv2.fastNlMeansDenoisingColored(image, templateWindowSize=7,
                                                      searchWindowSize=21,
                                                      h=3,
                                                      hColor=10)

    return cv2.bilateralFilter(self.orig_hsv, 5, 50, 50)

def ensure_dir(f):
    if not os.path.exists(f):
        os.makedirs(f)

def convert(src, dst):
    bgr = cv2.imread(src, cv2.IMREAD_COLOR)
    denoised = denoise(bgr)
    cv2.imwrite(dst, denoised)

if __name__ == '__main__':
    class MyParser(OptionParser):
        def format_epilog(self, formatter):
            return self.epilog

    examples = ("")
    parser = MyParser(usage="Usage: denoise.py -n [-f]", epilog=examples)

    parser.add_option('-n', '--name', dest='fname',
        help='specify target image name', default="")
    
    parser.add_option('-f', '--folder', dest='folder',
        help='specify folder name (will process all .jpg files in folder)', default="")
   
    (options, args) = parser.parse_args()
    
    if (options.fname == "") and (options.folder == ""):
        print "Nothing to do"
        exit(0)

    # -n option:
    if ('.jpg' in options.fname) or ('.JPG' in options.fname):
        convert(options.fname, os.path.splitext(options.fname)[0] + "_denoised.jpg")

    if (options.folder == ""):
        exit(0)

    files = [f for f in os.listdir(options.folder) if isfile(join(options.folder, f))] 
    jfiles = [f for f in files if ('.jpg' in f) or ('.JPG' in f)]

    print "found " + str(len(jfiles)) + " '.jpg' images"
    
    print os.getcwd()
    ensure_dir(join(options.folder, "denoised"))

    for i, f in enumerate(jfiles):
        print "Processing " + f + " (" + str(i + 1) + "/" + str(len(jfiles)) + ")"
        convert(join(options.folder, f), join(join(options.folder, "denoised"), f))
        
