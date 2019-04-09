import cv2
from picamera import PiCamera
from time import sleep
from datetime import datetime
from notify_run import Notify # https://notify.run/c/Hz16GGFB5LyehlRh
from collections import Counter
import json
import objectIdToName
import calibration
import time

#creates a notify object so notifications can be sent to phones/other devices
notify = Notify()
#this is the timestamp for saving the log file and calibration file
currentTimeFile = datetime.now().strftime('%Y%m%d%H%M%S')
#this creates camera object, has to pull it from calibration.py file, otherwise use comment below it
camera = calibration.getCamera()
#camera = PiCamera()
#set to keep track of depleting items, also for outputing when objects run out
setOfDepletingItems = set()
#this is for when all items are depleted and skip over threshold amount
setSkipsDepletingButRunsOut = set()

#below loads the model using the text-graph representation for OpenCV 
model = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb',
                                      'ssd_mobilenet_v2_coco_2018_03_29.pbtxt')

def objectDetection(thresholdConfidence):
    """Takes in a confidence between 0 and 1. At start of function, user
    prompted to use inputed confidence or default of 0.2 if user selects
    not to use the inputed confidence"""
    print("----Object Detection----")
    print("Default threshold 0.2 and will be set if not using calibration threshold")
    print("Recommend using default for un-organized objects or multi-object calibration")
    userCalibration = input("Use the calibration threshold: "+ str(thresholdConfidence)+"? [y/n] ")
    if userCalibration.lower() == "y":
        runningThreshold = thresholdConfidence
    else:
        runningThreshold = 0.2
    """THIS IS WHERE TO MODIFY CODE FOR EITHER FOR LOOP FOR TESTING OR WHILE LOOP FOR DEPLOYMENT"""
    i = 1 #this is for while loop instance to log iterations of loop
    while True: #while loop
    #for i in range(2): #for loop
        #creates file that can be appended to so it records data each iteration
        f=open('/home/pi/openCVData/LogFiles/' + currentTimeFile + "Log.txt", "a+") 
        #logs threshold used for log file
        f.write("Threshold: " + str(runningThreshold) + "\n")
        
        #this is the camera code, 5 second sleep is for balancing image before taking picture
        currentTime = datetime.now().strftime('%Y%m%d%H%M%S')
        camera.rotation = 180
        camera.start_preview()
        sleep(5)
        capturedImage = 'capturedImage' + str(currentTime) #creates filename with timestamp
        camera.capture('/home/pi/openCVData/Images/' + capturedImage + '.jpeg')
        camera.stop_preview()
        
        #this is the openCV code for reading in image
        image = cv2.imread('/home/pi/openCVData/Images/' +capturedImage + ".jpeg")
        image_height, image_width, _ = image.shape
        model.setInput(cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True))
        output = model.forward()
        #print(output[0,0,:,:].shape) #array of detected objects, max (100,7)
        #writes all detected objects to log file
        f.write("" + str(output[0,0,:,:]) + "\n")
        
        #creates a list to print objects to log file
        listOfObjectsForLog = list()
        #creates a list to write detected objects to
        listOfDetectedObjects = list()
        #below record all bounding boxes to check for false positive bound box overlap
        #X1, X2 are vertical left right edges, Y1, Y2 are horizontal bottom top edges
        overlapDetectedX1 = list() ###
        overlapDetectedX2 = list() ###
        overlapDetectedY1 = list() ###
        overlapDetectedY2 = list() ###
        
        #start of timer for detection
        start = time.time()
        #this is the loop that runs thgrough all objects detected
        for detection in output[0, 0, :, :]:
            confidence = detection[2]
            if confidence > runningThreshold:
                #these are the edges for each detected over threshold to check for overlap
                testEdgeX1 = detection[3] ###
                testEdgeX2 = detection[5] ###
                testEdgeY1 = detection[4] ###
                testEdgeY2 = detection[6] ###
                overlap = 0 ###
                #checks each box corner for overlap, overlap counts total corners overlapped
                for index in range(len(overlapDetectedX1)): #arbitrary which you choose
                    if testEdgeX1 < overlapDetectedX1[index] + .03 and testEdgeX1 > overlapDetectedX1[index] - .03:
                        if testEdgeY1 < overlapDetectedY1[index] + .03 and testEdgeY1 > overlapDetectedY1[index] - .03:
                            overlap += 1
                        if testEdgeY2 < overlapDetectedY2[index] + .03 and testEdgeY2 > overlapDetectedY2[index] - .03:
                            overlap += 1
                    if testEdgeX2 < overlapDetectedX2[index] + .03 and testEdgeX2 > overlapDetectedX2[index] - .03:
                        if testEdgeY1 < overlapDetectedY1[index] + .03 and testEdgeY1 > overlapDetectedY1[index] - .03:
                            overlap += 1
                        if testEdgeY2 < overlapDetectedY2[index] + .03 and testEdgeY2 > overlapDetectedY2[index] - .03:
                            overlap += 1
                #this runs if no overlap occured meaning no double count or secondary false count
                if overlap == 0:
                    #keeps track of what iteration, no iteration printed if no objects detected
                    print("Iteration: "+str(i))
                    class_id = detection[1]
                    class_name=objectIdToName.id_class_name(class_id,objectIdToName.getClassNames())
                    print(str(str(class_id) + " " + str(detection[2])  + " " + class_name))
                    
                    #this ensures all detected objects are only what is calibrated against
                    #if not using calibration code, comment out next two lines
                    listFromCalibration = calibration.getOnlyThisObject() ###
                    if class_name in listFromCalibration: ###
                        listOfDetectedObjects.append(class_name) #un-indent if not using calibration
                    
                    #this is the list for writing to the log file
                    listOfObjectsForLog.append(str(i) + " - " + class_name + " - " + str(detection[2]))
                    #this creates each bounding edge for boxes, adds edges for overlap detection
                    box_x = detection[3] * image_width
                    overlapDetectedX1.append(detection[3])###               
                    box_y = detection[4] * image_height                   
                    overlapDetectedY1.append(detection[4])###                   
                    box_width = detection[5] * image_width                    
                    overlapDetectedX2.append(detection[5]) ###
                    box_height = detection[6] * image_height
                    overlapDetectedY2.append(detection[6]) ###
                    #this is the openCV code that creates boxes with names
                    cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), (23, 230, 210), thickness=1)
                    cv2.putText(image,class_name + " " + str(round(confidence, 3)),(int(box_x), int(box_y+.05*image_height)),cv2.FONT_HERSHEY_SIMPLEX,(1),(0, 0, 255))
        if not listOfDetectedObjects:
            print("Iteration: " + str(i) + " ")
            f.write("Iteration: " + str(i) + " ")
        #end of timer for detection
        end = time.time()
        #prints detection time for all objects including false positives
        print("Time to detect all objects above: " + str(end - start))
        #writes to file the time to detect all objects including false positives
        f.write("Time to detect all objects above: " + str(end - start) + "\n")
        #saves the image with detection boxes to specified folder /ImagesWithBoxes
        cv2.imwrite('/home/pi/openCVData/ImagesWithBoxes/' + capturedImage + "withBoxes.jpeg",image)
        
        #dictionary with count of objects
        objectDict = Counter(listOfDetectedObjects)
        #output dictionary to be sent as notification and written to file
        notifySendString = {}
        #this uses count of items to create output for notifications and files
        #if no objects detected, objectDict is empty and this code doesn't run
        for key in objectDict:
            if objectDict[key] < 5: #threshold
                notifySendString[key] = str(objectDict[key]) + " (Running Low)" 
                setOfDepletingItems.add(key)
                #this is for when objects all get removed and skip above threshold
                if objectDict[key] > 0 and objectDict[key] < 3: ####
                    if key in setSkipsDepletingButRunsOut: ###
                        setSkipsDepletingButRunsOut.remove(key) ###
            else:
                notifySendString[key] = str(objectDict[key])
                setSkipsDepletingButRunsOut.add(key) ###
                #below removes item from setOfDepletingItems as it is above threshold
                if key in setOfDepletingItems:
                    setOfDepletingItems.remove(key)             
        #this checks if detected, hit threshold, runs out and adds to output dictionary           
        for item in setOfDepletingItems:
            if item not in list(objectDict.keys()):
                notifySendString[item] = "Out of: " + str(item)
        #this checks if detected, skips threshold, runs out and adds to output dictionary
        for item in setSkipsDepletingButRunsOut: ###
            if item not in list(objectDict.keys()): ###
                notifySendString[item] = "Out of: " + str(item) ###
            
        #this sends output dictionary of items, count to notify-run and writes them to log        
        if not listOfDetectedObjects and not setSkipsDepletingButRunsOut:
            notify.send("Out of: " +str(setOfDepletingItems)) #prints Out of set() if starting with nothing
            f.write(capturedImage + ' , ' + "{Out of: " + str(setOfDepletingItems) + "} \n")
        elif not listOfDetectedObjects and setSkipsDepletingButRunsOut: ###
            notify.send("Out of: " +str(setSkipsDepletingButRunsOut)) #prints Out of set() if starting with nothing
            f.write(capturedImage + ' , ' + "{Out of: " + str(setSkipsDepletingButRunsOut) + "} \n")
        else:
            notify.send(json.dumps(notifySendString))
            f.write(capturedImage + ' , ' + json.dumps(notifySendString) + " \n")
        #this writes each item and its probability to the log
        for objectAndProb in listOfObjectsForLog:
            f.write(objectAndProb + " \n")
        #closes writing to file for this iteration
        f.close()
    
        #saves the image with detection boxes to specified folder /ImagesWithBoxes
        #cv2.imwrite('/home/pi/openCVData/ImagesWithBoxes/' + capturedImage + "withBoxes.jpeg",image)
        
        #sleeps between taking pictures, measured in seconds
        sleep(10)
        i+=1 #this increments iteration for while loop
       