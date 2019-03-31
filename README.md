# OpenCV Object Detection without Tensorflow on Raspberry Pi

-General project description and info on OpenCV DNN-

The project runs on a Raspberry Pi 3b+. 

This project is designed to be a solution for low-cost
at-home inventory management. The project itself uses a
Raspberry Pi, Pi Camera, and Python. The camera captures
images of objects pre-trained in data models and counts
how many objects there are. It sends notifications saved 
as text strings to a mobile phone. The notifications 
contain the objects detected and the quantities of said 
objects. The softwares within Python used for the actual
object detection are OpenCV, MobileNet, and SSD. OpenCV 
is the Open Source Computer Vision Library that has a 
Python interface. MobileNet and SSD are combined to 
handle the object recognition and are discussed further 
in this readme. The combination is MobileNet-SSD and is
the software behind how the data models are created and
used.

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

This is a project using OpenCV to detect objects trained 
on models using the coco data set. OpenCV is a library 
that can be used in tandem with Tensorflow but also 
without Tensorflow. OpenCV has its own Deep Neural 
Network (DNN) that supplements the need for Tensorflow
deep learning libraries that are hard for edge devices. 
This project does not use Tensorflow as doing so is too 
strenuous on the Pi. Tensorflow can be installed and used 
within Raspbian but takes a lot of computing. Just using 
OpenCV does analysis on images with much less computing, 
which translates to much less time (few seconds). OpenCV 
uses the Tensorflow Google Protocol Buffer 
(protobuf / .pb) system that defines data objects, writes 
the data to files, and also reads the data. Protobuf is 
the foundation for the TFRecord system. Protobuf files 
are how trained models are stored. Protobuf is a way to 
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
its popularity. MobileNet is a base network that handles
the classification of objects within a model. SSD is the 
detection network used. MobileNet has its own detection
capability however by using SSD it can do multiple object
detection instead of single object detection. A brief
discussion of MobileNet and SSD can be found in the stack 
overflow article MobileNet vs SSD found at the bottom of 
this readme. MobileNet was created by Cornell, and the 
summary of the work can be found at the bottom of this 
readme. SSD was created in a collaboration between UNC 
Chapel Hill, Zoox, Google, and University Michigan 
Ann-Arbor, and the paper about SSD can be found at the 
bottom of this readme.

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

The code creates a log file for each run. The format
lists each picture taken (imag/imageWithBoxes) followd
by a comma then a dictionary of each object detected
with the number detected. If the number is below the
threshold set (objectDict[key] < 5 in code) then
after the number of objects is the a message
(Running Low) in parenthesis. Below this line is a
listing of each object detected with its percentage.
Each log file captures all data per run of the script.
The file itself is created at the start of the
for/while loop that the main code is within. Each pass
through the loop is recorded in the log file. If the 
program crashes between loop iterations it can be
detected in the log file. Also because the image
file names are timestamps, you can know exactly when
the process crashed.

Notifications are sent out using notify_run. Info
on notify_run can be found at the bottom of this 
readme. It is a library that is used to send
notifications to devices registered on a channel.
There is an issue with a channel not working after 
use. Best solution is to just register a new channel.
You can do so from commandline or in Python script.
Doing so in the script doesn't display the channel
info so I don't recommend this method. Using the
commandline shows the channel name web address, and
a QR code for the channel name web address. Any 
devices registered on the channel can see messages
sent over it.

-References / Resources for more info-

OpenCV website and PyPI page
https://opencv.org/
https://pypi.org/project/opencv-python/

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

Notify-Run Website and PyPI page
https://notify.run/
https://pypi.org/project/notify-run/

