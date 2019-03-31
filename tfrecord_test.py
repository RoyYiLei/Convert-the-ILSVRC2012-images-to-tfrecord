# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 19:45:06 2019

@author: lei
"""


import tensorflow as tf
import cv2
import numpy as np
import os
from PIL import Image
 
if __name__=='__main__':
    tfrecords_filename = 'D:/ILSVRC2012/data/train_tf/train_1.tfrecord'
    filename_queue = tf.train.string_input_producer([tfrecords_filename],) #读入流中
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           "image": tf.FixedLenFeature([], tf.string, default_value=""),
                "height": tf.FixedLenFeature([1], tf.int64, default_value=[0]),
                "width": tf.FixedLenFeature([1], tf.int64, default_value=[0]),
                "channels": tf.FixedLenFeature([1], tf.int64, default_value=[3]),
                "colorspace": tf.FixedLenFeature([], tf.string, default_value=""),
                "img_format": tf.FixedLenFeature([], tf.string, default_value=""),
                "label": tf.FixedLenFeature([1], tf.int64, default_value=[0]),
                "text": tf.FixedLenFeature([], tf.string, default_value=""),
                                       })  #取出包含image和label的feature对象

#    height = features["height"]
#    width = features["width"]
#    channels = features["channels"]
    image = tf.cast(tf.image.decode_jpeg(features["image"], channels=3), tf.uint8)
    
    label = tf.cast(features['label'], tf.int64)
    with tf.Session() as sess: #开始一个会话
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        coord=tf.train.Coordinator()
        threads= tf.train.start_queue_runners(coord=coord)
        for i in range(20):
            example, l = sess.run([image,label])#在会话中取出image和label
            img=Image.fromarray(example, 'RGB')#这里Image是之前提到的
            img.save('./'+str(i)+'_''Label_'+str(l)+'.jpg')#存下图片
            print(example, l)
 
        coord.request_stop()