import cv2
from picamera import PiCamera
from time import sleep
from datetime import datetime
from notify_run import Notify # https://notify.run/c/Hz16GGFB5LyehlRh
from collections import Counter
import json
import objectIdToName
import time

#this is the timestamp for saving the log file and calibration file
currentTimeFile = datetime.now().strftime('%Y%m%d%H%M%S')
#this creates camera object
camera = PiCamera()
#below loads the model using the text-graph representation for OpenCV 
model = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb',
                                      'ssd_mobilenet_v2_coco_2018_03_29.pbtxt')

#for sending calibration as only object(s) to notification
onlyThisObject = list()
def getOnlyThisObject():
    return onlyThisObject

#this is to call the camera in objectDetection.py file
def getCamera():
    return camera

def calibrateThreshold():
    """This uses a captured image to calibrate the threshold of confidence.
    The capture image must have known counts of objects as they are inputed
    by the user for calibration. The output is the calibrated confidence."""
    
    #this creates log file for calibration
    f=open('/home/pi/openCVData/CalibrationFiles/' + currentTimeFile + "Log.txt", "a+") #changed to append from write
    
    #this is the camera code, 5 seconds is to balance image before taking picture
    currentTime = datetime.now().strftime('%Y%m%d%H%M%S')
    camera.rotation = 180
    camera.start_preview()
    sleep(5)
    #creates filename with timestamp
    capturedImage = 'capturedImage' + str(currentTime) 
    #this needs to capture an image with a known object count
    camera.capture('/home/pi/openCVData/CalibrationFiles/' + currentTimeFile + 'ForCalibration.jpeg')
    f.write('Calibrate Image: /home/pi/openCVData/CalibrationFiles/' + currentTimeFile + 'ForCalibration.jpeg\n')
    camera.stop_preview()
    
    print("----Calibrating----")
    inputBoolean = 0
    inputObjectWithCount = {}
    #this takes in the user input for calibration and writes it to log file
    while inputBoolean == 0:
        #this is the object(s) in image for calibration
        objectBeingCounted = ""
        while objectBeingCounted not in objectIdToName.getClassNames().values():
            objectBeingCounted = str(input("What object is being counted?: "))
            if objectBeingCounted not in objectIdToName.getClassNames().values():
                print("Misspelled the object. Try again.")
        objectBeingCounted = objectBeingCounted.lower()
        #this is so that only calibrated objects are sent in notifications in objectDetection
        onlyThisObject.append(objectBeingCounted) ###
        f.write("Object used for calibration: " + objectBeingCounted + "\n")
        #this is the count of object(s) in image for calibration
        realCountForCalibration = 0
        while realCountForCalibration == 0:
            try:
                realCountForCalibration = int(input("How many " +objectBeingCounted+  " are there actually?: "))
            except ValueError:
                print("Not an integer. Try again.")
        f.write("Number of " +objectBeingCounted + ": " + str(realCountForCalibration) + "\n")        
        #this is the dictionary of all test objects with their counts
        inputObjectWithCount[objectBeingCounted] = realCountForCalibration
        #this is for inputing more than one object for calibration
        #recommend limitting how many objects are used for calibration to one, maybe two
        moreObjects = ""
        while moreObjects == "":
            moreObjects = input("Are there more objects [y/n]?: ")
            if moreObjects.lower() == 'n':
                inputBoolean = 1
            elif moreObjects.lower() != 'y':
                moreObjects = ""
                print("Did not input y or n. Try again.")
    
    #this is where openCV reads in the image and does object detection
    image = cv2.imread('/home/pi/openCVData/CalibrationFiles/' +currentTimeFile + "ForCalibration.jpeg")
    image_height, image_width, _ = image.shape
    model.setInput(cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True))
    output = model.forward()
    
    whileLoop = 0
    actualObjectNumber = len(inputObjectWithCount.keys())
    testingConfidence = 0.9
    iterationNumber = 1
    print("Calibrating each object...")
    f.write("Calibrating each object...\n")
    #this does the calibration
    alreadyCalibrated = list()
    
    #start of timer for calibration
    start = time.time()
    while whileLoop == 0:
        #this checks for exit condition of all input objects being detected
        if actualObjectNumber == 0:
            #this 35% variance is for detecting objects not tested (65% of threshold)
            #this 35% is from observations, however an official survey of error may change this
            testingConfidence = testingConfidence*.65
            print("Final threshold (65% of lowest object threshold): " + str(testingConfidence))
            f.write("Final threshold (65% of lowest object threshold): " + str(testingConfidence) + "\n")
            
            #end of timer for calibration
            end = time.time()
            #prints time to do calibration
            print("Time to do calibration: " + str(end - start))
            #writes to file time it takes for calibration
            f.write("Time to do calibration: " + str(end - start) + "\n")
            
            return float(testingConfidence)
            #whileLoop = 1 #this is if you don't need a return value
        
        print("Iteration: " + str(iterationNumber))
        f.write("Iteration: " + str(iterationNumber)+ "\n")
        listOfDetected = list()
        for detection in output[0, 0, :, :]:
            confidence = detection[2]
            if confidence > testingConfidence:
                class_id = detection[1]
                class_name=objectIdToName.id_class_name(class_id,objectIdToName.getClassNames())
                if class_name not in alreadyCalibrated:
                    if class_name in inputObjectWithCount: #.keys() alternative
                        listOfDetected.append(class_name)
                #this is for when object is detected and prints object and percentage confidence
                print("Object detected")
                print(str(str(class_id) + " " + str(detection[2])  + " " + class_name))
                f.write("Object detected\n"
                        + str(class_id) + " " + str(detection[2])  + " " + class_name + "\n")
                    
        objectCount = Counter(listOfDetected)
        #this is for when threshold is too high and no input objects are detected
        if len(listOfDetected) == 0:
            print("Threshold too high for object. Threshold -.05")
            f.write("Threshold too high for object. Threshold -.05\n")
            testingConfidence -= .05
        #this is where the threshold is changed based on input objects being detected
        else: 
            for key in inputObjectWithCount: #.keys() as alternative
                if key in objectCount: #.keys() as alternative
                    numberCountedByCamera = int(objectCount[key])
                    #this is when not all input objects counted, threshold reduced
                    if numberCountedByCamera < inputObjectWithCount[key]:
                        print("Threshold too high for " +key+  ". Threshold -.05")
                        f.write("Threshold too high for " +key+  ". Threshold -.05\n")
                        testingConfidence -= .05
                    #this is for false positives for input objects, threshold increased
                    elif numberCountedByCamera > inputObjectWithCount[key]:
                        print("Threshold too low for " +key+  ". Threshold +.01")
                        f.write("Threshold too low for " +key+  ". Threshold +.01\n")
                        testingConfidence += .01
                    #this is for correct count of input objects
                    else:
                        print("Calibration has correct count of " + key)
                        print("Calibration for " + key + " has threshold: " + str(testingConfidence))
                        f.write("Calibration has correct count of " + key + "\n" +
                                "Calibration for " + key + " has threshold: " + str(testingConfidence)+ "\n")
                        alreadyCalibrated.append(key)
                        #this reduces for each object (key) that is calibrated
                        actualObjectNumber -= 1
        
        #this is for when threshold confidence goes to 0 or negative, exits script
        if testingConfidence <= 0:
            print("Testing Confidence <= 0. Ending script")
            f.write("Testing Confidence <= 0. Ending script\n")
            exit()
        #this increments to keep track of loop iterations
        iterationNumber += 1
    
        
        

