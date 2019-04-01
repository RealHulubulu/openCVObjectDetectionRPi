import cv2
from picamera import PiCamera
from time import sleep
from datetime import datetime
from notify_run import Notify # https://notify.run/c/Hz16GGFB5LyehlRh
from collections import Counter
import json

#creates a notify object so notifications can be sent to phones/other devices
notify = Notify()
#this is the timestamp for saving the log file and calibration file
currentTimeFile = datetime.now().strftime('%Y%m%d%H%M%S')
#this creates camera object
camera = PiCamera()
#set to keep track of depleting items, also for outputing when objects run out
setOfDepletingItems = set()

runningThreshold = 0

#below loads the model using the text-graph representation for OpenCV 
model = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb',
                                      'ssd_mobilenet_v2_coco_2018_03_29.pbtxt')

#pretrained classes in the model
classNames = {0: 'background',
              1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
              7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
              13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
              18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
              24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
              32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
              37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
              41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
              46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
              51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
              56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
              61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
              67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
              75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
              80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
              86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

#this function returns object names as the data models only have their IDs and not the names
def id_class_name(class_id, classes):
    for key, value in classes.items():
        if class_id == key:
            return value

#calibrates the threshold based on current environment settings like lighting
def calibrateThreshold():
    f=open('/home/pi/openCVData/CalibrationFiles/' + currentTimeFile + "Log.txt", "a+") #changed to append from write
    currentTime = datetime.now().strftime('%Y%m%d%H%M%S')
    camera.rotation = 180
    camera.start_preview()
    sleep(5)
    #this needs to capture an image with a known object count
    capturedImage = 'capturedImage' + str(currentTime) #creates filename with timestamp
    camera.capture('/home/pi/openCVData/CalibrationFiles/' + currentTimeFile + 'ForCalibration.jpeg')
    f.write('Calibrate Image: /home/pi/openCVData/CalibrationFiles/' + currentTimeFile + 'ForCalibration.jpeg\n')
    camera.stop_preview()
    
    #need to know the exact object count
    print("----Calibrating----")
    inputBoolean = 0
    inputObjectWithCount = {}
    while inputBoolean == 0:
        objectBeingCounted = str(input("What object is being counted?: "))
        objectBeingCounted = objectBeingCounted.lower()
        f.write("Object used for calibration: " + objectBeingCounted + "\n")       
        realCountForCalibration = int(input("How many " +objectBeingCounted+  " are there actually?: "))
        f.write("Number of " +objectBeingCounted + ": " + str(realCountForCalibration) + "\n")
        realCountForCalibration = int(realCountForCalibration)
        inputObjectWithCount[objectBeingCounted] = realCountForCalibration
        moreObjects = input("Are there more objects [y/n]?: ")
        if moreObjects.lower() == 'n':
            inputBoolean = 1
        
    image = cv2.imread('/home/pi/openCVData/CalibrationFiles/' +currentTimeFile + "ForCalibration.jpeg")
    image_height, image_width, _ = image.shape
    model.setInput(cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True))
    output = model.forward()
    x = 0
    actualObjectNumber = len(inputObjectWithCount.keys())
    objectAndPercent = list()
    #print(actualObjectNumber)
    testingConfidence = .9
    iterationNumber = 1
    print("Calibrating each object...")
    f.write("Calibrating each object...\n")
    while x == 0:
        print("Iteration: " + str(iterationNumber))
        f.write("Iteration: " + str(iterationNumber)+ "\n")
        listOfDetected = list()
        for detection in output[0, 0, :, :]:
            confidence = detection[2]
            if confidence > testingConfidence:
                class_id = detection[1]
                class_name=id_class_name(class_id,classNames)
                listOfDetected.append(class_name)
                if len(listOfDetected) == 0:
                    print("Nothing detected")
                    f.write("Nothing detected\n")
                else:
                    print("Object detected")
                    print(str(str(class_id) + " " + str(detection[2])  + " " + class_name))
                    f.write("Object detected\n"
                            + str(class_id) + " " + str(detection[2])  + " " + class_name + "\n")
                #objectAndPercent.append(class_name + ", " + str(confidence))
        #print(x) #prints infinite 0s       
        objectCount = Counter(listOfDetected)
        """FOUND PROBLEM!!! the below print objectCount prints Counter(), not dict"""
        #print(objectCount) 
        #if len(inputObjectWithCount.keys()) > 0:
        """some reason never reaches below, the print x below loops inf tho,
        for loop not reached, i think because dict format?"""
        #print(inputObjectWithCount) #infinitely prints, does print dict with person : 1
        #print(listOfDetected)
        if actualObjectNumber == 0:
            print("Final threshold: " + str(testingConfidence))
            f.write("Final threshold: " + str(testingConfidence) + "\n")
            return float(testingConfidence)
            x = 1
            break
        #print(x) #infinitely printed
        elif len(listOfDetected) == 0:
            print("Threshold too high for any object. Threshold -.05")
            f.write("Threshold too high for any object. Threshold -.05\n")
            testingConfidence -= .05
        else: #for loop below not reached, no clue why
            #print("hi")
            for key in inputObjectWithCount: #.keys()
                if key in objectCount: #.keys()
                    numberCountedByCamera = int(objectCount[key])
                    #print(numberCountedByCamera)
                    if numberCountedByCamera < inputObjectWithCount[key]:
                        print("Threshold too high for " +key+  ". Threshold -.05")
                        f.write("Threshold too high for " +key+  ". Threshold -.05\n")
                        testingConfidence -= .05
                    elif numberCountedByCamera > inputObjectWithCount[key]:
                        print("Threshold too low for " +key+  ". Threshold +.01")
                        f.write("Threshold too low for " +key+  ". Threshold +.01\n")
                        testingConfidence += .01
                    else:
                        print("Camera has correct count of " + key)
                        print("Calibration for " + key + " has threshold: " + str(testingConfidence))
                        f.write("Camera has correct count of " + key + "\n" +
                                "Calibration for " + key + " has threshold: " + str(testingConfidence)+ "\n")
                        actualObjectNumber -= 1
        if testingConfidence <= 0:
            print("Testing Confidence <= 0. Ending script")
            f.write("Testing Confidence <= 0. Ending script\n")
            exit()
        iterationNumber += 1
        #this subtraction is to handle error for detecting objects not tested (75% of threshold)
        testingConfidence = testingConfidence - (testingConfidence*.25)
        #return float(testingConfidence)
    #have it run analysis again using final threshold
    #final threshold should be lowest
    #some cases may have too low threshold where another object gets over counted

testingConf = calibrateThreshold()
sleep(10)
print()
userCalibration = input("Use the calibration threshold: "+ str(testingConf)+"? [y/n] ")
if userCalibration.lower() == "y":
    runningThreshold = testingConf
else:
    runningThreshold = 0.4
    
    
#the loop below is the bulk of the code, while True is for running continuous, for for fixed time
#i is updated in for loop but needs to be += for while loop, code is commented out below
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
    #print(output[0,0,:,:].shape) #this is the (100,7) array of detected objects
    #print(output[0,0,:,:]) #this prints out the array output of detected objects
    listOfObjectsForLog = list() #creates a list to print objects to log file
    listOfDetectedObjects = list()
    for detection in output[0, 0, :, :]:
        confidence = detection[2]
        if confidence > runningThreshold:
        #if confidence > 0.4:
            class_id = detection[1]
            class_name=id_class_name(class_id,classNames)
            #this just keeps track of which iteration of the loop the objects were detected in
            print(i)
            print(str(str(class_id) + " " + str(detection[2])  + " " + class_name))
            listOfDetectedObjects.append(class_name)
            #this is the list for writing to the log file
            listOfObjectsForLog.append(str(i) + " - " + class_name + " - " + str(detection[2]))
            #below creates the boxes around objects detected in the image with text for each
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
    
    #below uses count of items to create output for notifications and files
    for key in objectDict:
        if objectDict[key] < 5: #threshold
            notifySendString[key] = str(objectDict[key]) + " (Running Low)" 
            setOfDepletingItems.add(key)
        else:
            notifySendString[key] = str(objectDict[key])
            #below removes item from setOfDepletingItems as it is above threshold
            if key in setOfDepletingItems:
                setOfDepletingItems.remove(key)
                
    #below checks if item was detected but has run out, adds it to output dictionary           
    for item in setOfDepletingItems:
        if item not in list(objectDict.keys()):
            notifySendString[item] = "Out of: " + str(item)
            
    #below sends output dictionary of items and count to notify-run and writes them to the log        
    if not listOfDetectedObjects:
        notify.send("Out of: " +str(setOfDepletingItems)) #prints Out of set() if starting with nothing
        f.write(capturedImage + ' , ' + "{Out of: " + str(setOfDepletingItems) + "} \n")
    else:
        notify.send(json.dumps(notifySendString))
        f.write(capturedImage + ' , ' + json.dumps(notifySendString) + " \n")
        
    #below writes each item and its probability to the log
    for objectAndProb in listOfObjectsForLog:
        f.write(objectAndProb + " \n")
    f.close()
    
    #saves the image with detection boxes to specified folder /ImagesWithBoxes
    cv2.imwrite('/home/pi/openCVData/ImagesWithBoxes/' + capturedImage + "withBoxes.jpeg",image)

    sleep(10)
    #i+=1