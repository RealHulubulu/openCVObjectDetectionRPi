#import cv2
#from picamera import PiCamera
#from time import sleep
#from datetime import datetime
#from notify_run import Notify # https://notify.run/c/Hz16GGFB5LyehlRh
#from collections import Counter
#import json
import calibration
import objectDetection
import objectIdToName

calibratedThreshold = calibration.calibrateThreshold()

objectDetection.objectDetection(calibratedThreshold)

