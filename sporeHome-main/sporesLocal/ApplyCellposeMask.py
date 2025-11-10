# author: Alexandra Nava
# date: 2025-09-25
# description: Short description of the file



from pathlib import Path
import pandas as pd
from skimage import io, measure

def apply_cellpose_mask(image_path: str, mask_path: str) -> pd.DataFrame:
    """
    Apply a Cellpose-generated mask to an image and save the masked image.

    :image_path: Path to the image file to be segmented
    :mask_path: Path to the Cellpose-generated mask file.
    :data_output_path: path to spot properties csv file
    """

    img = io.imread(image_path)
    mask = io.imread(mask_path)

    # measure region properties with the mask applied
    props = measure.regionprops_table(
        mask,
        intensity_image=img,
        properties=('label', 'mean_intensity', 'centroid', 'area', 
                    'perimeter', 'major_axis_length', 'minor_axis_length')
    )
    df = pd.DataFrame(props)

    return df