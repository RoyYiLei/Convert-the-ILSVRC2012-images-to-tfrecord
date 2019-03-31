# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:50:32 2019

@author: lei
"""

import tensorflow as tf
import cv2
import numpy as np
import os
from multiprocessing import Process, Queue
import time
import random
import math
from scipy.io import loadmat
import caffe_classes
 
max_num = 1000  #max record number in one file
train_path1 = 'F:/train/'  #the folder stroes the train images, and the folder contains 1000 folders.
#train_path2 = 'C:/train/'  #while my PC SSD do not enough space to store the ILSVRC2012 train images,I divide the training images into two copies. 
valid_path = 'F:/ILSVRC2012_img_val/'  #the folder stroes the validation images
meta_path = 'E:/ILSVRC2012/data/meta.mat' #the file download from ILSVRC2012, while contains the information about ['ILSVRC2012_ID', 'WNID', 'words', 'gloss', 'num_children', 'children', 'wordnet_height', 'num_train_images']
cores = 6   #number of CPU cores to process

metadata = loadmat(meta_path, struct_as_record=False)
	
	# ['ILSVRC2012_ID', 'WNID', 'words', 'gloss', 'num_children', 'children', 'wordnet_height', 'num_train_images']
synsets = np.squeeze(metadata['synsets'])
   #ids = np.squeeze(np.array([s.ILSVRC2012_ID for s in synsets]))
wnids = np.squeeze(np.array([s.WNID for s in synsets]))
words = np.squeeze(np.array([s.words for s in synsets]))

 
#Imagenet images are saved in the /data directory, which has 1000 subdirectories and gets the names of these subdirectories.
classes1 = os.listdir(train_path1)
#classes2 = os.listdir(train_path2)
 
#Build a dictionary, Key is the directory name, value is the class name 0-999

labels_dict = {}
for i in range(1000):
    for j in range(1000):
        if  caffe_classes.class_names[i] == words[j]:
            labels_dict[wnids[j]] = i

#Build a list of training set files, each element inside is path name + image file name + class name
images_labels_list = []
for i in range(len(classes1)):
    path = train_path1 + classes1[i]+ '/'
    images_files = os.listdir(path)
    label = str(labels_dict[classes1[i]])
    for image_file in images_files:
        images_labels_list.append(path+','+image_file+','+ label)
"""
for i in range(len(classes2)):
    path2 = train_path2 + classes2[i]+ '/'
    images_files2 = os.listdir(path2)
    label2 = str(labels_dict[classes2[i]])
    for image_file in images_files2:
        images_labels_list.append(path2+','+image_file+','+ label2)
"""

random.shuffle(images_labels_list) #random the order of images list

#Read the description of Labels
labels_text = {}
for i in range(1000):
    labels_text[str(i)] = caffe_classes.class_names[i]
    
 
#Read the class labels file corresponding to the image of the validation set
valid_classes = []
with open('E:/ILSVRC2012/data/ILSVRC2012_validation_ground_truth.txt', 'r') as f:
	  valid_classes = [line.strip() for line in f.readlines()]

val = {}
for i in range(1,1001):
    val[str(i)] = i-1
val_labels = []
for i in range(len(valid_classes)):
    val_labels.append (labels_dict[wnids[val[valid_classes[i]]]])
#Build a list of validation set files, each element inside is path name + image file name + class name
valid_images_labels_list = []
valid_images_files = os.listdir(valid_path)
for file_item in valid_images_files:
    number = int(file_item[15:23])-1
    val_label = str(val_labels[number])
    valid_images_labels_list.append(valid_path+','+file_item+','+ val_label)

#convert the images and labels data to tfrecord format
def make_example(image, height, width, label, text):
    colorspace = b'RGB'
    channels = 3
    img_format = b'JPEG'
    return tf.train.Example(features=tf.train.Features(feature={
        'image' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        'height' : tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'width' : tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'channels' : tf.train.Feature(int64_list=tf.train.Int64List(value=[channels])),
        'colorspace' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[colorspace])),
        'img_format' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_format])),
        'label' : tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        'text' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[text]))
    }))
 

labels ={}
for i in range(1000):
    labels[str(i)] = i

#This function is used to generate the TFRECORD file. 
#The first parameter is a list. Each element is the image file name plus the class label. 
#The second parameter is the written directory name.
#The third parameter is the starting sequence number of the file name, 
#and the fourth parameter is the queue name, which is used to send messages to the parent process.
def gen_tfrecord(trainrecords, targetfolder, startnum, queue):
    tfrecords_file_num = startnum
    file_num = 0
#    total_num = len(trainrecords)
    pid = os.getpid()
    queue.put((pid, file_num))
    writer = tf.python_io.TFRecordWriter(targetfolder+'train_'+str(tfrecords_file_num)+'.tfrecord')
    for record in trainrecords:
        file_num += 1
        fields = record.split(',')
        img = cv2.imread(fields[0]+fields[1])
        height, width, _ = img.shape
        
        img_jpg = cv2.imencode('.jpg', img)[1].tobytes()
        label = labels[fields[2]]
        text = labels_text[fields[2]]
        ex = make_example(img_jpg, height, width, label, text.encode())
        writer.write(ex.SerializeToString())
        #Reports to the parent process every 100 records, reports progress
        if file_num%100==0:
            queue.put((pid, file_num))
        if file_num%max_num==0:
            writer.close()
            tfrecords_file_num += 1
            writer = tf.python_io.TFRecordWriter(targetfolder+'train_'+str(tfrecords_file_num)+'.tfrecord')
    writer.close()        
    queue.put((pid, file_num))

#This function is used to generate TFRECORD files in multiple processes. 
#The first parameter is the list of file names of the images to be processed, and the second parameter is the number of CPU cores needed.
#File directory name written by the third parameter
def process_in_queues(fileslist, cores, targetfolder):
    total_files_num = len(fileslist)
    each_process_files_num = int(total_files_num/cores)
    files_for_process_list = []
    for i in range(cores-1):
        files_for_process_list.append(fileslist[i*each_process_files_num:(i+1)*each_process_files_num])
    files_for_process_list.append(fileslist[(cores-1)*each_process_files_num:])
    files_number_list = [len(l) for l in files_for_process_list]
    
    each_process_tffiles_num = math.ceil(each_process_files_num/max_num)
    
    queues_list = []
    processes_list = []
    for i in range(cores):
        queues_list.append(Queue())
        #queue = Queue()
        processes_list.append(Process(target=gen_tfrecord, 
                                      args=(files_for_process_list[i],targetfolder,
                                      each_process_tffiles_num*i+1,queues_list[i],)))
 
    for p in processes_list:
        Process.start(p)
 
    #The parent process loops through the queue's messages and updates every 0.5 seconds.
    while(True):
        try:
            total = 0
            progress_str=''
            for i in range(cores):
                msg=queues_list[i].get()
                total += msg[1]
                progress_str+='PID'+str(msg[0])+':'+str(msg[1])+'/'+ str(files_number_list[i])+'|'
            progress_str+='\r'
            print(progress_str, end='')
            if total == total_files_num:
                for p in processes_list:
                    p.terminate()
                    p.join()
                break
            time.sleep(0.5)
        except:
            break
    return total



if __name__ == '__main__':

    print('Start processing train data using %i CPU cores:'%cores)
    starttime=time.time()       	  
    total_processed = process_in_queues(images_labels_list, cores, targetfolder='D:/ILSVRC2012/data/train_tf/')
    endtime=time.time()
    print('\nProcess finish, total process %i images in %i seconds'%(total_processed, int(endtime-starttime)))
    """
    print('Start processing validation data using %i CPU cores:'%cores)
    starttime=time.time()  
    total_processed = process_in_queues(valid_images_labels_list, cores, targetfolder='D:/ILSVRC2012/data/valid_tf/')
    endtime=time.time()
    print('\nProcess finish, total process %i images, using %i seconds'%(total_processed, int(endtime-starttime)))
    """