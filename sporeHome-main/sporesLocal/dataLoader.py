import seaborn as sns 
sns.set_theme(style="darkgrid")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, Bounds
import pandas as pd
import time

def computeRates(csv):
    ##
    # Alex's plotting routine, computes rates also. 
    # takes: CSV name as a string. CSV file must be in data_dir. 
    ##
    rates_data = []

    df = pd.read_csv(data_dir + csv)
    df = df[["Frame", "Track_PhC", "Status_PhC", "Germinant", "Germination_Index_PhC"]]
    max_frames = max(df["Frame"].values)
    germinant_concentration = df.loc[~df["Germinant"].isin([0]), "Germinant"].unique()
    germinant_exposure_frames = df.loc[~df["Germinant"].isin([0]), "Frame"].values
    total_spores = df["Track_PhC"].nunique()

    for frame_i in range(max_frames + 1 - window_size):

        if frame_i in germinant_exposure_frames:
          germinant_value = germinant_concentration.max()
        else:
          germinant_value = 0

        frame_window = list(range(frame_i, frame_i + window_size))
        spores_germinated = df[df["Germination_Index_PhC"].isin(frame_window)]
        num_germinated = spores_germinated["Track_PhC"].nunique()
        norm_num_germinated = num_germinated / total_spores
        rates_data.append([frame_i, norm_num_germinated, num_germinated, germinant_value])

    df_germination_rates = pd.DataFrame(rates_data, columns = ["Frame", "Normalized_Rate", "Rate", "Germinant_Concentration"])

    # plotting routine, commented out cause a little slow and can vary from machine to machine.  
    #for germinant in germinant_exposure_frames:
    #    plt.axvline(germinant, alpha = 0.5, color = "lightgrey")
    #    exp = csv.replace(data_dir, "").replace("_Data.csv","")
    #sns.lineplot(x = "Frame", y = f"Normalized_Rate", data = df_germination_rates, linewidth = 4)

    #title = exp + f" {germinant_concentration.max()}mM"
    #plt.title(title)
    #plt.savefig(data_save_dir + f"Germination_Rate_{exp}.jpg")
    #plt.show()
    #df_germination_rates.to_csv(data_save_dir + f"Germination_Rate_{exp}.csv", index = False)
    # end plotting

    return df_germination_rates


if __name__=="__main__":
    
    # load in csv to compute rates
    window_size = 3 # unit: frames. Each frame is 5 minutes. 
    data_dir = "/home/nicho/workspace/sporesLocal/data/"   # contains raw CSVs
    data_save_dir = "/home/nicho/workspace/sporesLocal/data/generatedData/"  # where I save generated files to
    # two sets of data, currently point to one or the other in the
    #below funtion call. Will overwrite the other .npy in generated data, should be easy to fix if desired.
    fpath1 = "M6813_s4_Data.csv"
    fpath2 = "M6813_s1_Data.csv" 

    df_germination_rates = computeRates(fpath1)

    rawDat = df_germination_rates[["Frame", "Normalized_Rate"]].to_numpy()
    # rawDat has dim (140, 2), organized by (frame, normalized_rate) 
    np.save(data_save_dir + "frame_vs_rate", rawDat)
    
    zDat = rawDat[:, 1]
    timeDat = rawDat[:, 0]

    #can also load like:
    a = np.load(data_save_dir + "frame_vs_rate.npy")
    #print(a)


    print(np.trapz(zDat, timeDat))
