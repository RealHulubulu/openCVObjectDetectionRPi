# OpenCV Object Detection without Tensorflow on Raspberry Pi

-General project description and info on OpenCV DNN-

This project captures images using the standard Pi Camera 
V2.1 and detects objects in the images using the OpenCV 
DNN. The entire project does object detection on the Pi.
The files are stored in a directory with common root
folder as the project itself, but the storage can be
wherever you choose. The only real restriction for this
particular code is you must have the .pb and .pbtxt 
models in the same folder as main.py. You can have these 
two model files in any director if you set the path in 
readNetFromTensorflow('Path/To/File/' + modelName.pb, 
'Path/To/File/' + modelName.pbtxt).

The code is an implementation of the Object Detection
Tensorflow API found at the bottom of this readme.
There is some code taken from rdeepc (Saumya Shovan Roy)
in his overview of how OpenCV DNN works with his repo
and Heartbeat article found at the bottom of this readme.

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

For info about TFRecords and protobuf, see the article
at the bottom of this readme.

For the model, MobileNet-SSD v2 was used. It is used
in the Heartbeat article as the suggested model due to
its popularity. A discussion of what MobileNet and SSD
can be found in the stack overflow article MobileNet vs
SSD found at the bottom of this readme. MobileNet was
created by Cornell, and the summary of the work can be 
found at the bottom of this readme. SSD was created in
a collaboration between UNC Chapel Hill, Zoox, Google,
and University Michigan Ann-Arbor, and the paper about
SSD can be found at the bottom of this readme.

-Things in the code- 

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

The code creates a log file for each run.

-References / Resources for more info-

TFRecords and Protobuf
https://halfbyte.io/33/

TensorFlow Object Detection API  Documentation
(with pre-trained models and code to create 
.pbtxt for OpenCV)
https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API

General Implementation based on TensorFlow API 
Documentation for OpenCV by Saumya Shovan Roy
https://github.com/rdeepc/ExploreOpencvDnn

Heartbeat article discussing OpenCV DNN by
Saumya Shovan Roy
https://heartbeat.fritz.ai/real-time-object-detection-on-raspberry-pi-using-opencv-dnn-98827255fa60

MobileNet vs SSD stack overflow
https://stackoverflow.com/questions/49789001/mobilenet-vs-ssd

Cornell MobileNets paper summary and info
https://arxiv.org/abs/1704.04861

UNC Chapel Hill, Zoox, Google, University
of Michigan Ann-Arbor paper on SSD
https://www.cs.unc.edu/~wliu/papers/ssd.pdf
