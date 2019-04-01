# Convert-the-ILSVRC2012-images-to-tfrecord
Convert the ILSVRC2012 train and validation images to tfrecord format according to the label of caffe_classes
While we train our networks in PC (especially in mechanical hard disk), there is a problem that the efficiency of 
reading small files (images) is too low. In that case, We need convert the images to tfrecord format which is a binary file that stores 
data and label together. It can make better use of memory and copy, move, and read faster in the tensorflow graph. 

In the while,  the labels in meta.mat(download from IMAGENET: http://www.image-net.org/challenges/LSVRC/2012/nonpub-downloads) 
does not match the pre-training model (caffe_classes.py). So the aim is to convert the ILSVRC train and validation images to tfrecord 
format according to the label of caffe_classes. The picture following shows the relationship of each file.

![image](https://github.com/RoyYiLei/Convert-the-ILSVRC2012-images-to-tfrecord/blob/master/images/pic%201.png)

Just change the file address(SSD will faster 25x than Mechanical hard disk)  in the code and number of CPU cores to process in the tfrecord_maker.py, I run the code in WIN10, tensorflow1.12, and the tfrecord_test.py will test the tfrecord file work or not by generating 20 images with label. 


Reference：https://blog.csdn.net/gzroy/article/details/85954329.
