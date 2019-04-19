import calibration
import objectDetection
import objectIdToName

"""CURRENTLY HAVE WHILE LOOP SET FOR DEPLOYMENT at 10 seconds"""

calibratedThreshold = calibration.calibrateThreshold()

objectDetection.objectDetection(calibratedThreshold)

