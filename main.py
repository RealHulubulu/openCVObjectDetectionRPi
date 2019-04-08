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

"""CURRENTLY HAVE WHILE LOOP SET FOR DEPLOYMENT at 10 seconds"""
#bananas have to be horizontal

"""[ 0.00000000e+00  5.20000000e+01  9.59541261e-01  4.51447248e-01
   5.34423292e-01  8.04373264e-01  7.56225288e-01]
 [ 0.00000000e+00  5.20000000e+01  7.73660719e-01  1.10421598e-01
   3.44388306e-01  4.00141060e-01  5.63168108e-01]
 [ 0.00000000e+00  5.20000000e+01  5.69080234e-01  1.02199093e-01
   6.09039903e-01  4.16821241e-01  8.41133237e-01]
   
   resolved issue with detectin donuts, threshold set at 3
"""
calibratedThreshold = calibration.calibrateThreshold()

objectDetection.objectDetection(calibratedThreshold)

