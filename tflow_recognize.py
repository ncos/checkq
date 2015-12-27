#!/usr/bin/python

import tensorflow as tf
import numpy as np
import cv2
import tflow_model as tm


def print_probabilities(res):
    for i, q in enumerate(res):
        print str(i) + " : " + str(q) 

def recognize(image):
    img = cv2.resize(image, (28, 28), fx=0, fy=0, interpolation = cv2.INTER_CUBIC)

    batch = np.zeros((784), dtype=np.float)
    for i in xrange(28):
        for j in xrange(28):
            batch[i*28+j] = img[i][j] 

    with tf.Session() as sess:
        tm.saver.restore(sess, "./model.ckpt")
        #print("Model restored.")

        result = tm.y_conv.eval(feed_dict={
            tm.x: [batch], tm.keep_prob: 1.0}, session=sess)
   
        #print_probabilities(result[0])
        return result

    #print("test accuracy %g"%tm.accuracy.eval(feed_dict={
    #    tm.x: tm.mnist.test.images, tm.y_: tm.mnist.test.labels, tm.keep_prob: 1.0}, session=sess))























