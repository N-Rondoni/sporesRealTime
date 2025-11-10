# purpose: driver script, which sends to command line execution of other scripts for real time control of spore device.
# author:  Nick Rondoni

import time
from datetime import datetime, timezone
import pytz
import os
import numpy as np
from fileChecker import hasNewFiles
from desiredTrajectory import desiredTraj
from voltageSetting import voltageSetter
# begin Alex's imports
from aicspylibczi import CziFile
from tifffile import imwrite
from SelectFocusedImage import focused_image_selection, best_brenner_score
from ComputeCellposeMask import write_mask
from ApplyCellposeMask import apply_cellpose_mask
from Preprocessing import preprocess_image
from WriteGerminationStatus import write_germination_status
from CalculatePercentageGerminated import calculate_percentage_germinated
import pandas as pd




### define changable parameters
# working directory that is watched for new files
folder = r"C:\palmsens\proj\rawData"
# "/home/nicho/workspace/sporesLocal"
# how many seconds to check for new files
pollTime = 5
# final frame number. Multiples of 5 minutes. May want to iterate indefinitely.
frameFinal = 6
# how long voltage, decided by controller, will be supplied for. Should be less than duration of a frame. 

duration = 270 
duration = 10 # just for testing


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
    #output_dir =  r"C:\palmsens\proj\genData\\"
    output_dir =  r"C:/palmsens/proj/genData/"

    #print(output_dir)

    ### FOR T IN TIMEPOINTS:
    #czi_path = '/Users/alexandra/Library/CloudStorage/Dropbox/ARO-Files/Device-Segmentation/8-20-2025/Test.czi' # point to new CZI
    czi_path = str(newfpaths[0]) #"/Users/nrondoni/Workspace/spore/sporeHome-main_saveStateOct15/sporesLocal/data/9_30"# point to newfpaths during real time 
    spore_data_output_path = f"{output_dir}spore_data.csv"
    print(czi_path)


    focused_image_path: str = focused_image_selection(czi_path, output_dir) # pass back path of focused image for this timepoint
    #preprocessed_image_path: str = preprocess_image(focused_image_path)

    print("focused image path:", focused_image_path)
    preprocessed_image_path = focused_image_path #output_dir + 'focused_t=000_z=028.tiff')

    #timepoint = preprocessed_image_path.split('t=')[1].split('_')[0] # determine timepoints from image path
    #timepoint = i
    #print("Gut Check: should we pull timepoint from name?", timepoint == i)
    timepoint = i
    imaging = "PhC" # determine imaging from image path 

# produce mask at first timepoint, than apply to all timepoints
    if int(timepoint) == 1: 
      print("new print:", preprocessed_image_path, output_dir)
      mask_path = write_mask(preprocessed_image_path, output_dir)
    print("made it through write_mask")
    data_time_t = apply_cellpose_mask(preprocessed_image_path, mask_path) 
    

# write spore data to csv
    if int(timepoint) == 1:
      data_time_t.to_csv(spore_data_output_path.format(imaging), index=False)
    else:
      data_time_t.to_csv(spore_data_output_path.format(imaging), mode='a', header=False, index=False)

    data_all_time = pd.read_csv(spore_data_output_path.format(imaging))
    data_all_time_fpath = str(spore_data_output_path.format(imaging))

    data_all_time_with_germ_status = write_germination_status(data_all_time_fpath, timepoint) # goes through each spore and add germination status column value to current timepoint

# calculate percentage germinated
    currently_germinated_percentage = calculate_percentage_germinated(data_all_time_with_germ_status["timepoint"] == int(timepoint))



#### ============ CONTROLS
    # estimate germination % at current frame. Temporarily loading in experimental data, should be replaced with call that computes in real time. 
    print("Estimating current germination rate and computing control action...")
    tempDataLocation = r"C:\palmsens\proj\sporeHome-main\sporesLocal\data\generatedData\frame_vs_rate.npy" #"/home/nicho/workspace/sporesLocal/data/generatedData/frame_vs_rate.npy"
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
    print("voltageSetter executes here, when connected to windows machine and potentiostat uncomment. ")
    #voltageSetter(voltOut, duration) #uncomment on windows machine

    # have OS code execute new file, with appropriate sleep time
    #               potentially have a seperate driver checking for new methodscripts? potentiostat execution seperate?


    # set found back to false; re-initializes while loop checking for new files. 
    found = False


