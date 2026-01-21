# author: Alexandra Nava
# date: 2025-09-24
# description: Short description of the file

#IMPORTS 
from aicspylibczi import CziFile
from tifffile import imwrite
import numpy as np
from SelectFocusedImage import focused_image_selection, best_brenner_score
from ComputeCellposeMask import write_mask
from ApplyCellposeMask import apply_cellpose_mask
from Preprocessing import preprocess_image
from WriteGerminationStatus import write_germination_status
from CalculatePercentageGerminated import calculate_percentage_germinated
import pandas as pd
#==================================================================
### WILL WORK BEST IF WAIT A FEW TIMEPOINTS TO BEGIN EXPERIMENT to determine phase trace stability 
### ASSUMES ALL SPORES ARE DORMANT

base_dir = '/Users/alexandra/Library/CloudStorage/Dropbox/ARO-Files/Device-Segmentation/012026-testing/' # base directory contains folder of images and output folder
image_input_dir = f"{base_dir}M4576_s2_PhC_crop/" #directory to images
output_dir = f'{base_dir}output/' #directory to output masks, images, and data

init_timepoint_index, n_timepoint_index = 0, 50 # incase imaging starts with timepoint 1, final_timepoitn_index is for testing only 
mask_name = "cellpose_mask_t=000.tiff" # where to save cellpose mask 
live_segmentation = True # keep True for device
image_naming_convention = "M4576_s2_PhC_t={t}.tif" #naming convention for images


### FOR T IN TIMEPOINTS:

for t in range(init_timepoint_index, n_timepoint_index + 1):
  image_path = f'{image_input_dir}{image_naming_convention.format(t=str(t).zfill(4))}' #FUTURE: read in path from Nick's algorithm 
  t = int(t)  # convert back to integer for processing functions
  spore_data_output_path = output_dir + "spore_data.csv" #where to write spore data including physiological features and germination statuses for all timepoints 

  imaging = "PhC" # FUTURE: determine imaging from image path 
  print(f"processing {imaging} imaging, timepoint {t}...")

  ### FOCUSED IMAGE SELECTION AND PREPROCESSING
  #focused_image_path: str = focused_image_selection(image_path, output_dir, live_segmentation) # pass back path of focused image for this timepoint
  #preprocessed_image_path: str = preprocess_image(focused_image_path, timepoint, [0, 100], [0, 200], focused_image_path)
  preprocessed_image_path = image_path #** TESTING WITHOUT FOCUSING AND PREPROCESSING

  # produce mask at first timepoint, than apply to all timepoints
  if int(t) == init_timepoint_index:  
    write_mask(preprocessed_image_path, output_dir, mask_name)
  data_time_t = apply_cellpose_mask(preprocessed_image_path, output_dir + mask_name, t) 

  # write spore data to csv
  if int(t) == init_timepoint_index:
    data_time_t.to_csv(spore_data_output_path.format(imaging), index=False)
  else:
    data_time_t.to_csv(spore_data_output_path.format(imaging), mode='a', header=False, index=False)

  # add germination status and overwrite csv
  data_all_time_with_germ_status: pd.DataFrame = write_germination_status(spore_data_output_path.format(imaging), t) # goes through each spore and add germination status column value to current timepoint
  data_all_time_with_germ_status.to_csv(spore_data_output_path.format(imaging), index=False) # overwrite with germination status included

  # calculate percentage germinated
  germinated_percentage_over_time: np.array = calculate_percentage_germinated(data_all_time_with_germ_status)
  print(f'germinated percentage over time: {germinated_percentage_over_time}')


