# purpose: driver script, which sends to command line execution of other scripts for real time control of spore device.
# author:  Nick Rondoni

import time
from datetime import datetime, timezone
import pytz
import os
import numpy as np
from fileChecker import hasNewFiles
from desiredTrajectory import desiredTraj



### define changable parameters
# working directory that is watched for new files
folder = "/home/nicho/workspace/sporesLocal"
# how many seconds to check for new files
pollTime = 5
# final frame number. Multiples of 5 minutes. May want to iterate indefinitely.
frameFinal = 6
# how long voltage, decided by controller, will be supplied for. Should be less than duration of a frame. 
duration = 270 

### initializations that shouldn't be changed
i = 0 # starting frame index
found = False
known = None
startTimeUnix = time.time()
pacific = pytz.timezone("US/Pacific")
startTime = datetime.fromtimestamp(startTimeUnix, pacific)
vMin, vMax = -2.5, 2.5
###

print("##################################################################")
print("Beginning driver for real time control of spore germination rates!")
print("------------------------------------------------------------------")
print("Starting time:", startTime.strftime("%Y-%m-%d %H:%M:%S %Z"))
print("------------------------------------------------------------------")
while i <= frameFinal:
    print("Polling for new images every", pollTime, "seconds")
    print("current frame number:", i)
    while found == False: # will check every pollTime seconds until found becomes true. 
        found, known, newfpaths = hasNewFiles(folder, known)
        #print(found, known)
        if found:
            print("New files detected!", newfpaths, "has been found.")
            i = i + 1
        else:
            print("No new files.")
            time.sleep(pollTime)     
    # once found becomes True, there is a new image (or at least file), thus we are on the next frame, and should iterate i. Done in "if found" check.
    # select least blurry file, newfpaths is a list of strings, which are fnames. Pass these into blurry decider. 

#### ============ SEGMENTATION
    ### Question: Will we only receive one image at a time? Or will this be a czi stack for each timepoint? 
    ### Question: What will be the identifier between PhC and ThT when we start doing that? 
            
    ### Select focused image functions (czi stack path)
    ### Image preprocessing - background subtraction, etc. (focused image path)
    ### if t == 0: 
    ###        run mask 
    ### apply t0 mask to current image, write spot properties to csv for timepoint (one csv for each timepoint)
    ### if imaging == PhC: 
    ###     determine germination from spot properties (fluorescence)
    ### then estimate germination % currently 
#### ============ CONTROLS
    # estimate germination % at current frame. Temporarily loading in experimental data, should be replaced with call that computes in real time. 
    print("Estimating current germination rate and computing control action...")
    tempDataLocation = "/home/nicho/workspace/sporesLocal/data/generatedData/frame_vs_rate.npy"
    frame_v_rate = np.load(tempDataLocation)
    frame = frame_v_rate[i, 0] # good gut check to compare this to i. Maybe shouldn't be used? 
    rate = frame_v_rate[i, 1]

    ## compute control action as a function of this, e.g., suppose germ% is 0.5 currently. Compare to desired traj.
    # Eventually compute PID error between desired trajectory, and actual current rate
    setPoint = desiredTraj(i)
    ep = (setPoint - rate)
    ## supply voltage to potentiostat as a function of this error. Hold for 5 minutes? 4.9 minutes? What is safe.
    #                                                              consider: will not poll for new files until done. 
    voltOut = ep*vMax
    print("Next voltage", voltOut, "computed, beginning potentiostat communication")

    # connect to device, set voltage for duration seconds
    #voltageSet(voltOut, duration) uncomment on windows machine

    # have OS code execute new file, with appropriate sleep time
    #               potentially have a seperate driver checking for new methodscripts? potentiostat execution seperate?


    # set found back to false; re-initializes while loop checking for new files. 
    found = False


