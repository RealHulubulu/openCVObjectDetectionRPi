import cv2
from picamera import PiCamera
from time import sleep
from datetime import datetime
from notify_run import Notify # https://notify.run/c/Hz16GGFB5LyehlRh
from collections import Counter
import json
import objectIdToName
import calibration

#creates a notify object so notifications can be sent to phones/other devices
notify = Notify()
#this is the timestamp for saving the log file and calibration file
currentTimeFile = datetime.now().strftime('%Y%m%d%H%M%S')
#this creates camera object, has to pull it from calibration.py file, other use comment below it
camera = calibration.getCamera()
#camera = PiCamera()
#set to keep track of depleting items, also for outputing when objects run out
setOfDepletingItems = set()

#below loads the model using the text-graph representation for OpenCV 
model = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb',
                                      'ssd_mobilenet_v2_coco_2018_03_29.pbtxt')

def objectDetection(thresholdConfidence):
    """Takes in a confidence between 0 and 1. At start of function, user
    prompted to use inputed confidence or default of 0.4 if user selects
    not to use the inputed confidence"""
    print("----Object Detection----")
    userCalibration = input("Use the calibration threshold: "+ str(thresholdConfidence)+"? [y/n] ")
    if userCalibration.lower() == "y":
        runningThreshold = thresholdConfidence
    else:
        runningThreshold = 0.4
    #while True:
    for i in range(2):
        #i = 0 #this is for while loop instance to log iterations of loop
        f=open('/home/pi/openCVData/LogFiles/' + currentTimeFile + "Log.txt", "a+") #changed to append from write
        currentTime = datetime.now().strftime('%Y%m%d%H%M%S')
        camera.rotation = 180
        camera.start_preview()
        sleep(5)
        capturedImage = 'capturedImage' + str(currentTime) #creates filename with timestamp
        camera.capture('/home/pi/openCVData/Images/' + capturedImage + '.jpeg')
        camera.stop_preview()
    
        image = cv2.imread('/home/pi/openCVData/Images/' +capturedImage + ".jpeg")
        image_height, image_width, _ = image.shape
        model.setInput(cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True))
        output = model.forward()
        #print(output[0,0,:,:].shape) #this is array of detected objects, max (100,7) 
        #print(output[0,0,:,:]) #this prints array output of all detected objects (incl false pos)
        listOfObjectsForLog = list() #creates a list to print objects to log file
        listOfDetectedObjects = list()
        for detection in output[0, 0, :, :]:
            confidence = detection[2]
            if confidence > runningThreshold:
                print("Iteration: "+str(i))
                class_id = detection[1]
                class_name=objectIdToName.id_class_name(class_id,objectIdToName.getClassNames())
                #this just keeps track of which iteration of the loop the objects were detected in
                print(str(str(class_id) + " " + str(detection[2])  + " " + class_name))
                listOfDetectedObjects.append(class_name)
                #this is the list for writing to the log file
                listOfObjectsForLog.append(str(i) + " - " + class_name + " - " + str(detection[2]))
                #this creates the boxes around objects detected in the image with text for each
                box_x = detection[3] * image_width
                box_y = detection[4] * image_height
                box_width = detection[5] * image_width
                box_height = detection[6] * image_height
                cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), (23, 230, 210), thickness=1)
                cv2.putText(image,class_name ,(int(box_x), int(box_y+.05*image_height)),cv2.FONT_HERSHEY_SIMPLEX,(.005*image_width),(0, 0, 255))
    
        #dictionary with count of objects
        objectDict = Counter(listOfDetectedObjects)
        #output dictionary to be sent as notification and written to file
        notifySendString = {}
    
        #this uses count of items to create output for notifications and files
        for key in objectDict:
            if objectDict[key] < 5: #threshold
                notifySendString[key] = str(objectDict[key]) + " (Running Low)" 
                setOfDepletingItems.add(key)
            else:
                notifySendString[key] = str(objectDict[key])
                #below removes item from setOfDepletingItems as it is above threshold
                if key in setOfDepletingItems:
                    setOfDepletingItems.remove(key)
                
        #this checks if item was detected but has run out, adds it to output dictionary           
        for item in setOfDepletingItems:
            if item not in list(objectDict.keys()):
                notifySendString[item] = "Out of: " + str(item)
            
        #this sends output dictionary of items and count to notify-run and writes them to the log        
        if not listOfDetectedObjects:
            notify.send("Out of: " +str(setOfDepletingItems)) #prints Out of set() if starting with nothing
            f.write(capturedImage + ' , ' + "{Out of: " + str(setOfDepletingItems) + "} \n")
        else:
            notify.send(json.dumps(notifySendString))
            f.write(capturedImage + ' , ' + json.dumps(notifySendString) + " \n")
        
        #this writes each item and its probability to the log
        for objectAndProb in listOfObjectsForLog:
            f.write(objectAndProb + " \n")
        f.close()
    
        #saves the image with detection boxes to specified folder /ImagesWithBoxes
        cv2.imwrite('/home/pi/openCVData/ImagesWithBoxes/' + capturedImage + "withBoxes.jpeg",image)

        sleep(10)
        #i+=1 #this increments iteration for while loop
       