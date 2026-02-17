import numpy as np
import time
from voltageSetting import voltageSetter, voltageSetterNoDuration, disconnect

## driver routine to perform open loop experiments as needed on 2/16/26. 
#  each line is an experiment, and we should image for 2-4 hours after. 

#H+ delivery 
#+2  volts for 5 min, a -2 for 30s to reset the device (30min)
#+1  volts for 5 min, a -2 for 30s to reset the device (30min)
#+0.5 volts for 5 min, a -2 for 30s to reset the device (30min)

#OH- delivery 
#+2  volts for 5 min, a -2 for 30s to reset the device (30min)
#+1  volts for 5 min, a -2 for 30s to reset the device (30min)
#+0.5 volts for 5 min, a -2 for 30s to reset the device (30min)




# all experiments have same stim duration
totalDurationStim = 30 # minutes
totalDurationStimSeconds = totalDurationStim*60
def experiment1():
    #is_first_timepoint = True
    voltageSetterNoDuration(2, True)
    time.sleep(300)
    # reset device
    voltageSetterNoDuration(-2, False)
    time.sleep(30)
    
    # repeat above, but already connected. 
    voltageSetterNoDuration(2, False)
    time.sleep(300)
    voltageSetterNoDuration(-2, False)
    time.sleep(30) # 660 seconds have passed

    # repeat above, but already connected. 
    voltageSetterNoDuration(2, False)
    time.sleep(300)
    voltageSetterNoDuration(-2, False)
    time.sleep(30) # 990 seconds... 

    # repeat above, but already connected. 
    voltageSetterNoDuration(2, False)
    time.sleep(300)
    voltageSetterNoDuration(-2, False)
    time.sleep(30)

    # repeat above, but already connected. 
    voltageSetterNoDuration(2, False)
    time.sleep(300)
    voltageSetterNoDuration(-2, False)
    time.sleep(30)

    # repeat above, but already connected. 
    voltageSetterNoDuration(2, False)
    time.sleep(300)
    voltageSetterNoDuration(-2, False)
    time.sleep(30) # 6*330 = 1980 seconds (or 33 minutes) have passed. 

    disconnect()

def experiment1():
    #is_first_timepoint = True
    voltageSetterNoDuration(0.5, True)
    time.sleep(300)
    # reset device
    voltageSetterNoDuration(1, False)
    time.sleep(30)
    
    # repeat above, but already connected. 
    voltageSetterNoDuration(2, False)
    time.sleep(300)
    voltageSetterNoDuration(-2, False)
    time.sleep(30) # 660 seconds have passed

    # repeat above, but already connected. 
    voltageSetterNoDuration(2, False)
    time.sleep(300)
    voltageSetterNoDuration(-2, False)
    time.sleep(30) # 990 seconds... 

    # repeat above, but already connected. 
    voltageSetterNoDuration(2, False)
    time.sleep(300)
    voltageSetterNoDuration(-2, False)
    time.sleep(30)

    # repeat above, but already connected. 
    voltageSetterNoDuration(2, False)
    time.sleep(300)
    voltageSetterNoDuration(-2, False)
    time.sleep(30)

    # repeat above, but already connected. 
    voltageSetterNoDuration(2, False)
    time.sleep(300)
    voltageSetterNoDuration(-2, False)
    time.sleep(30) # 6*330 = 1980 seconds (or 33 minutes) have passed. 

    disconnect()


if __name__=="__main__":
    #experiment1()
    voltageSetter(2, 300)

