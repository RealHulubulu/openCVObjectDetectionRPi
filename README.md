# OpenCV Object Detection without Tensorflow on Raspberry Pi

This is a project using OpenCV’s DNN to detect objects 
trained on the coco data set. The project runs on a 
Raspberry Pi 3b+. The project does not use Tensorflow 
as doing so is too strenuous on the Pi. OpenCV uses the 
Tensorflow Google Protocol Buffer (protobuf / .pb) 
system that defines data objects, writes the data to 
files, and also reads the data. Protobuf is the 
foundation for the TFRecord system. Protobuf files are 
how trained models are stored. Protobuf is a way to 
store massive data in a smaller file size. When using 
.pb files (used by Tensorflow), you need a more powerful 
CPU/GPU than the Pi. The way OpenCV uses .pb files is 
through a .pbtxt file created from the .pb file. The 
.pbtxt file is a text-based representation of the 
serialized graph stored within the .pb file. The .pbtxt 
file avoids the usual computational heavy process of 
creating and using a graph object from the .pb file. 

The classNames file lists out all of the objects 
pre-trained in the MobileNet-SSD v2 model found in the 
documentation for the TensorFlow Object Detection API. 
The model has the recognized objects stored by numbers 
from 1 to 90 but doesn’t have the names of the objects. 
Hence the listing out of the objects in a dictionary. 

The following line of code is the model created using 
OpenCV from the pre-trained .pb and it’s associated 
text-based .pbtxt as discussed above.

model = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'ssd_mobilenet_v2_coco_2018_03_29.pbtxt')





References / Resources for more info:

TFRecords and Protobuf
https://halfbyte.io/33/

TensorFlow Object Detection API  Documentation (with pre-trained models and code to create .pbtxt for OpenCV)
https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API

General Implementation based on TensorFlow API Documentation for OpenCV
https://github.com/rdeepc/ExploreOpencvDnn
