# OpenCV Object Detection without Tensorflow on Raspberry Pi

Authors: 

Daniel Karasek (primary) - Kennesaw State University

Linda Vu - Kennesaw State University

Rachel Wendel - Kennesaw State University

######################################################

-Updates-

4/9/2019
Third official logged update!

*Added in Google AI blog link as a reference.

*Corrected the origin of Mobilenets as a Google project
that was published at Cornell. 

*Corrected some of the readme as well as added in some
notes about some of the code including the algorithm 
that handles detection box overlap. The corrections 
were all on what portions of the code does.

*Updated sample of log file and console output to have
the runtime displayed.

*Forgot to mention in the last update that the 
notifications sent out from objectDetection now only
track objects that are calibrated. All objects are
recorded in log files to keep all data.

*Added in user input error handling in calibration and 
objectDetection code.

------------------------------------------------------

4/8/209
Second official logged update!

*The code now logs runtimes for calibration and object
detection. Both are recorded in seconds. The runtimes 
are also written to console.

*The names of objects put in the boxes for the images
that are labeled ...WithBoxes have been changed. The
font size has been made smaller to better fit in the
box for each respective object. Also, the rounded 
detected percentage is printed next to the name of the
object. Before the text was too large and would be 
hard to read when multiple objects were being detected.

*Regarding the fixed boxes, the sample images have been
updated as well.

*Fixed the issue with iterations not being logged when
not objects were detected. Now all iterations are 
logged in log file and in console.

*Set object detection count threshold to 5 to better
reflect what we have written for a paper on this 
project.

*We have submitted a draft of paper for this project.
If we move to get it published, we will put a copy of
it in this Github.

------------------------------------------------------

4/7/2019
First official logged update! Many changes made since 
last commit. 

*The code now handles double counting and false second 
counting of objects. It looks at the bounding boxes of 
each detected object and checks for overlap. 

*The code now sends proper notifications when objects
all are depleted and skip the depletion threshold 
amount. For example, the objects go from 10 being
counted to 0 being counted. Notification now reflects
that said object is depleted.

*The log files now record all objects that are 
detected in each captured image. Before it would only
list out the objects above threshold. Adding in a list
of the output from openCV to log all things that are
detected.

*Calibration now adjusts for a 35% variance in its 
threshold. Through some testing 30% was too low.

*Did some testing of calibration and object detection
code. Found out that where the items are in the image
affects its calculated percentage. It has to do with
how SSD works for detecting objects. Best advice is 
to have objects close to the center of the image and
have objects organized in some fashion. For example,
detect donuts by using the box they came in. Found out
some objects are harder to detect than others. Bananas 
are a challenge if they have spots. Also, the objects
you detect must be similar to objects trained on in 
COCO dataset. For example, it does not count plastic
forks only real metal forks.

*In testing found out multiple object calibration does
work but has issues with placement within the image as
mentioned above.

*Some changes above have been added into the readme 
below. Only really things relating to understanding the
project itself.

######################################################

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
detection network used. MobileNet has its own
classification capability however by using SSD it can do 
object detection. A brief
discussion of MobileNet and SSD can be found in the stack 
overflow article MobileNet vs SSD found at the bottom of 
this readme. MobileNet was created by Google, and the 
summary of the work can be found at the bottom of this 
readme in the Google AI post and in the paper summary 
published at Cornell. SSD was created in a collaboration 
between UNC Chapel Hill, Zoox, Google, and University 
Michigan Ann-Arbor, and the paper about SSD can be found 
at the bottom of this readme.

-Things in the code- 

The code is broken into a very readable file structure.
Originally the project was one lengthy .py file that
was harder to follow. Now, there is a main,
calibration, objectDetection, and objectIdToName file.
Each file is modular for easy use. There are comments
in each to inform how to use each file. Additionally,
there is a file sampleOfConsoleOutput.txt that shows
a sample output in console for this project. The only
particular thing is the yes or no input for both the
calibration and objectDetection file. Look at code
for details as it doesn't do input error handling.

The calibration portion of the code creates a
detection threshold through an algorithm. If that 
threshold is not used, a default threshold is then
used instead. It is set to .2 in the code as that is
a reasonable minimum threshold for our purposes. The 
.2 eliminates many of the false positives that occur.

When running the object detection portion of the code
found within the objectDetection.py file, it currently
runs with a while loop for running continuously. 
There is commented out code for a for loop that can be
set with a certain amount of iterations for tests. A 
two iteration for loop is the simplest example of use 
for our inventory management idea as it can track 
change in two different images.

Within the while loop is an algorithm that detects any
bounding box overlap. False positives can show up even 
with high detection thresholds. These false positives
are double counts of an already detected object, with
the double count either being a second count of that 
object or a completely different object. For example,
it may correctly detect a banana but double count the 
banana or count it as a banana and an apple. The 
algorithm looks for other bounding boxes that are 
similar in position and shape to any exisitng boxes 
and ignores them if so. This greatly reduced these
false positive errors.

The classNames file lists out all of the objects 
pre-trained in the MobileNet-SSD v2 model found in the 
documentation for the TensorFlow Object Detection API. 
The model has the recognized objects stored by numbers 
from 1 to 90 but doesn’t have the names of the objects. 
Hence the listing out of the objects in a dictionary.
The objects are from the COCO Data Set. Find a link
to COCO data set at the bottom of this readme.

The following line of code is the model created using 
OpenCV from the pre-trained .pb and it’s associated 
text-based .pbtxt as discussed above.

model = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'ssd_mobilenet_v2_coco_2018_03_29.pbtxt')

Past calibration, each image is double saved as the raw
image and as the image with bounding boxes. The images 
with bounding boxes have labels that say what is 
detected with a rounded detection percentage next to the
label.

The code creates a log file for each run. At the top is
the detection threshold used. Then the next line lists
out each picture taken (imag/imageWithBoxes) followed
by a comma then a dictionary of each object detected
with the number detected. If the number is below the
threshold set (objectDict[key] < 5 in code) then
after the number of objects is a message that says
(Running Low) in parenthesis. Next in the log is an
array that shows all objects detected including false
positives. The dimensions of the array are 100 rows by 7
columns. The rows are for each object detected. It is 
capped at 100. This is the from the output listed in the
objectDetection code. The first entry for each row of
the array is empty. The second entry is the object that
is detected represented by its ID number. The third entry 
is the detection percentage. The last four entries are 
the bounding box X and Y coordinates for each object. 
From this array, only objects above the calibrated 
detection threshold and those objects selected for 
calibration will be used in the code. However, it is 
important to have a log of everything that is detected.
Below the array is the runtime for detecting all objects
found in the array in seconds. Below this line is a
listing of each object detected with its percentage.
Each log file captures all data per run of the script.
The file itself is created at the start of the
for/while loop that the main code is within. Each pass
through the loop is recorded in the log file. If the 
program crashes between loop iterations it can be
detected in the log file. Also because the image
file names are timestamps, you can know exactly when
the process crashed.

There is also a log file made for calibration. It first
lists the image used for calibration. Then it lists out
each object used for calibration with the inputed count
of objects that are in the calibration image. Following
this it runs through iterations of calibration starting
with a threshold of 0.9. For each iteration if it is 
still missing objects that were inputed for calibration, 
the threshold is reduced by .05. This repeats until all 
objects are detected. There is no extra code here to 
handle box overlap because those false positives are 
only realized after all real objects are detected as 
discovered through testing. If an iteration over counts
the number of objects the threshold is increased by .01 
until the correct number is reached. At the end of this 
log is the runtime of calibration which is measured in
seconds.

Notifications are sent out using Notify-Run. Info
on Notify-Run can be found at the bottom of this 
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

COCO Data Set
http://cocodataset.org/

Google AI Blog Post Introducing Mobilenets
https://ai.googleblog.com/2017/06/mobilenets-open-source-models-for.html

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

