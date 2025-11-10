# author: Alexandra Nava
# date: 2025-09-24
# description: Short description of the file


import numpy as np
import skimage.io as skio

def preprocess_image(unprocessed_path, timestep, x: list[int, int], y: list[int, int],  processed_image_path: str) -> str:
    """
    Crop and normalize an image array based on given x and y coordinates.

    :x: list containing the start and end x-coordinates for cropping
    :y: list containing the start and end y-coordinates for cropping
    :return: path to processed image 
    """
    unprocessed_path_tlast = unprocessed_path.replace(f"t={timestep}", f"t={timestep-1}")
    unprocessed_image = skio.imread(unprocessed_path) # current image 
    image_aligned = None # align to last timepoint
    image_array = np.squeeze(image_aligned)
    cropped_image = image_array[y[0]:y[1], x[0]:x[1]]
    background_subtracted_image = None #rolling ball
    skio.imsave(processed_image_path, cropped_image.astype(np.uint16))
    processed_image_path = unprocessed_path # FOR TESTING REMOVE LATER
    return processed_image_path