# author: Alexandra Nava
# date: 2025-09-25
# description: Short description of the file


import pandas as pd
import numpy as np

def calculate_percentage_germinated(data: pd.DataFrame) -> float:
    """
    Calculate the percentage of germinated spores in the dataset.

    :data: DataFrame containing spore data with a 'germination_status' column.
    :return: np.array with percentage of germinated spores over timepoints
    """
    total_spores = data['label'].nunique()
    percentage_germinated_array = np.zeros(len(data['timepoint'].unique()))

    for timepoint in sorted(data['timepoint'].unique()):
        data_timepoint = data[data['timepoint'] == timepoint]
        germinated_spores = data_timepoint[data_timepoint['germination_status'] == 1]['label'].nunique()
        percentage_germinated_t = (germinated_spores / total_spores) * 100 if total_spores > 0 else 0
        percentage_germinated_array[int(timepoint)] = percentage_germinated_t

    return percentage_germinated_array