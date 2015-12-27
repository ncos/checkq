#!/usr/bin/python

import tensorflow as tf
import numpy as np
import cv2
import tflow_model as tm


def print_probabilities(res):
    for i, q in enumerate(res):
        print str(i) + " : " + str(q) 

with tf.Session() as sess:
    tm.saver.restore(sess, "./model.ckpt")
    print("Model restored.")

    result = tm.y_conv.eval(feed_dict={
        tm.x: [tm.mnist.test.images[0]], tm.keep_prob: 1.0}, session=sess)
   
    print_probabilities(result[0])

    #print("test accuracy %g"%tm.accuracy.eval(feed_dict={
    #    tm.x: tm.mnist.test.images, tm.y_: tm.mnist.test.labels, tm.keep_prob: 1.0}, session=sess))























