# Convert-the-ILSVRC2012-images-to-tfrecord
Convert the ILSVRC2012 train and validation images to tfrecord format according to the label of caffe_classes
While we train our networks in PC (especially in mechanical hard disk), there is a problem that the efficiency of 
reading small files (images) is too low. In that case, We need convert the images to tfrecord format which is a binary file that stores 
data and label together. It can make better use of memory and copy, move, and read faster in the tensorflow graph. 

In the while,  the labels in meta.mat(download from IMAGENET: http://www.image-net.org/challenges/LSVRC/2012/nonpub-downloads) 
does not match the pre-training model (caffe_classes.py). So the aim is to convert the ILSVRC train and validation images to tfrecord 
format according to the label of caffe_classes. The picture following shows the relationship of each file.

