# author: Alexandra Nava
# date: 2025-09-24
# description: Short description of the file



import pandas as pd
import numpy as np

def smooth_trace(trace: np.array, alpha: float = 0.3) -> np.array: 
  '''
  :trace: a 1D numpy array with values of the trace intensity over time 
  :return: a smoothed version of the trace 
  '''
  smoothed_trace = np.zeros_like(trace)
  smoothed_trace[0] = trace[0]
  for t in range(1, len(trace)):
    smoothed_trace[t] = alpha * trace[t] + (1 - alpha) * smoothed_trace[t-1]
  
  return smoothed_trace

def check_if_germination_occurs_now(phase_trace: np.array, drop_threshold: float = 0.1) -> int:
  baseline = np.mean(phase_trace[:5])  # average phase value of first 5 frames
  threshold = baseline * (1 - drop_threshold)    # drop_threshold% drop

  if phase_trace[-1] < threshold:
    germination_occurs = 1
  else:
    germination_occurs = 0

  return germination_occurs

def plot_and_save_phase_trace(phase_traces: pd.DataFrame, save_path: str) -> None:
  sns.lineplot(x = 'timepoint', y = 'mean_intensity', hue = 'label', data = phase_traces)
  df_during_germination = phase_traces[phase_traces['germination_status'] == 1]
  df_during_dormant = phase_traces[phase_traces['germination_status'] == 0]
  sns.scatterplot(x = 'timepoint', y = 'mean_intensity', color = "black", data = df_during_germination, marker='o', s=50, legend=False)
  sns.scatterplot(x = 'timepoint', y = 'mean_intensity', color = "lightgrey", data = df_during_dormant, marker='o', s=50, legend=False)
  plt.savefig(save_path)
  plt.close()

def write_germination_status(property_data_csv, t, plot_and_save_phase_trace = False) -> pd.DataFrame:
  '''
  :property_data_csv: path to the csv file with spore properties over all timepoints
  :t: current timepoint integer to add germination status for
  :return: None, but updates the csv file "germination_status" at time t with 0 or 1 for each spore
  '''
  spore_id_column = "label"
  phase_trace_column = "mean_intensity"
  germination_status_column = "germination_status"
  time_column = "timepoint"
  all_spore_data_all_time = pd.read_csv(property_data_csv)
  for spore_id in all_spore_data_all_time[spore_id_column].unique():
    spore_data: pd.DataFrame = all_spore_data_all_time[all_spore_data_all_time[spore_id_column] == spore_id]
    
    germination_status_all: np.array = spore_data[germination_status_column].values
    
    phase_trace: np.array = spore_data[phase_trace_column].values
    smoothed_phase_trace: np.array = smooth_trace(phase_trace)
    
    # determine if germination occurs NOW, or if germination had already occurred

    # if already occured, skip and add 1 to germination status
    if germination_status_all.values.any() == 1:
      germination_status_time_t: int = 1

    if germination_status_all.values.any() != 1:
      germination_status_time_t = check_if_germination_occurs_now(smoothed_phase_trace)

    all_spore_data_all_time.loc[
        (all_spore_data_all_time[spore_id_column] == spore_id) &
        (all_spore_data_all_time[time_column] == t),
        germination_status_column
    ] = germination_status_time_t
    
  if plot_and_save_phase_trace:
    PHASE_SAVE_PATH = property_data_csv.replace('.csv', '_phase_traces.png')
    plot_and_save_phase_trace(all_spore_data_all_time, PHASE_SAVE_PATH)
    print(f"saved phase traces to {PHASE_SAVE_PATH}")
  return all_spore_data_all_time